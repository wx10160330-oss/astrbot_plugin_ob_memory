"""Pytest configuration for the memory plugin test suite.

Tests import the plugin via its full package path::

    from astrbot_plugin_ob_memory.core.X import Y

For that to resolve we put ``data/plugins/`` (the parent of the plugin
folder) onto ``sys.path``. We do **not** put the plugin's own root onto
``sys.path``, because doing so would let ``from storage import ...``
work in tests but fail in production — AstrBot only adds the parent
directory of the plugin, never the plugin itself, so source modules
must use *relative* imports for cross-subpackage references.

Keeping the test environment aligned with production is what catches
import-shape regressions before they hit users.
"""

from __future__ import annotations

import sys
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
PLUGINS_DIR = PLUGIN_ROOT.parent  # data/plugins/

if str(PLUGINS_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGINS_DIR))
