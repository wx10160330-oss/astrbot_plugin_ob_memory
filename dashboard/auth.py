"""Dashboard authentication: password hashing, session tokens, middleware.

Security model:
- Password stored as SHA-256(salt + password) in dashboard_auth.json
- Environment variable MEMORY_DASHBOARD_PASSWORD overrides file password
- Sessions are in-memory tokens with 7-day expiry (lost on restart)
- Cookie: httpOnly, SameSite=Lax
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import secrets
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("astrbot_plugin_ob_memory.dashboard.auth")

SESSION_EXPIRY_SECONDS: int = 7 * 24 * 3600  # 7 days
MIN_PASSWORD_LENGTH: int = 4


def _hash_password(password: str, salt: str) -> str:
    """SHA-256(salt + password)."""
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()


def verify_password(password: str, salt: str, expected_hash: str) -> bool:
    """Constant-time comparison."""
    computed = _hash_password(password, salt)
    return hmac.compare_digest(computed, expected_hash)


class AuthStore:
    """Manages password persistence and session tokens."""

    def __init__(self, data_dir: Path):
        self._auth_file = data_dir / "dashboard_auth.json"
        self._sessions: dict[str, float] = {}  # token -> expiry timestamp
        self._salt: str | None = None
        self._hash: str | None = None
        self._env_locked: bool = False
        self._load()

    def _load(self) -> None:
        """Load password from env var or file."""
        env_pwd = os.environ.get("MEMORY_DASHBOARD_PASSWORD", "").strip()
        if env_pwd:
            self._env_locked = True
            salt = secrets.token_hex(16)
            self._salt = salt
            self._hash = _hash_password(env_pwd, salt)
            return

        if self._auth_file.exists():
            try:
                data = json.loads(self._auth_file.read_text(encoding="utf-8"))
                self._salt = data.get("salt", "")
                self._hash = data.get("hash", "")
            except Exception as e:
                logger.warning("failed to load dashboard_auth.json: %s", e)

    @property
    def setup_needed(self) -> bool:
        """True if no password has been configured yet."""
        return not self._env_locked and (not self._salt or not self._hash)

    @property
    def env_locked(self) -> bool:
        """True if password is set via environment variable."""
        return self._env_locked

    def setup_password(self, password: str) -> bool:
        """Set initial password. Returns False if already configured."""
        if not self.setup_needed:
            return False
        if len(password) < MIN_PASSWORD_LENGTH:
            return False
        salt = secrets.token_hex(16)
        h = _hash_password(password, salt)
        self._salt = salt
        self._hash = h
        self._persist(salt, h)
        return True

    def check_password(self, password: str) -> bool:
        """Verify a password attempt."""
        if not self._salt or not self._hash:
            return False
        return verify_password(password, self._salt, self._hash)

    def change_password(self, current: str, new_password: str) -> bool:
        """Change password. Returns False on wrong current or env-locked."""
        if self._env_locked:
            return False
        if not self.check_password(current):
            return False
        if len(new_password) < MIN_PASSWORD_LENGTH:
            return False
        salt = secrets.token_hex(16)
        h = _hash_password(new_password, salt)
        self._salt = salt
        self._hash = h
        self._persist(salt, h)
        self._sessions.clear()
        return True

    def _persist(self, salt: str, h: str) -> None:
        """Write auth data to file."""
        try:
            self._auth_file.parent.mkdir(parents=True, exist_ok=True)
            data = {"salt": salt, "hash": h, "created_at": time.time()}
            self._auth_file.write_text(
                json.dumps(data, ensure_ascii=False), encoding="utf-8"
            )
            # Best-effort chmod 600 on POSIX
            try:
                os.chmod(self._auth_file, 0o600)
            except (OSError, AttributeError):
                pass
        except Exception as e:
            logger.warning("failed to persist dashboard_auth.json: %s", e)

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------
    def create_session(self) -> str:
        """Issue a new session token."""
        token = secrets.token_urlsafe(32)
        self._sessions[token] = time.time() + SESSION_EXPIRY_SECONDS
        return token

    def validate_session(self, token: str | None) -> bool:
        """Check if a session token is valid and not expired."""
        if not token:
            return False
        expiry = self._sessions.get(token)
        if expiry is None:
            return False
        if time.time() > expiry:
            self._sessions.pop(token, None)
            return False
        return True

    def revoke_session(self, token: str | None) -> None:
        """Invalidate a session token."""
        if token:
            self._sessions.pop(token, None)

    def to_status_dict(self) -> dict[str, Any]:
        """Public status for /auth/status endpoint."""
        return {
            "setup_needed": self.setup_needed,
            "env_locked": self.env_locked,
        }
