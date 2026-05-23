"""AstrBot Memory Plugin entry point.

Wires the full plugin lifecycle (initialize / terminate) together: storage,
core engines (decay, embedding, search, surface, writer), LLM hooks/tools,
``/memory`` commands, and the embedded dashboard.
"""

from __future__ import annotations

from pathlib import Path

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.api.star import Context, Star, StarTools, register
from astrbot.core.config.astrbot_config import AstrBotConfig

from .core.decay_engine import DecayConfig, DecayEngine
from .core.dehydrator import Dehydrator
from .core.embedding_service import EmbeddingService
from .core.memory_manager import MemoryManager
from .core.memory_writer import MemoryWriter
from .core.search_service import SearchService
from .core.session_resolver import SessionResolver
from .core.surface_strategy import SurfaceStrategy
from .core.tagger import Tagger
from .dashboard.server import DashboardServer
from .handlers.commands import MemoryCommandsMixin
from .handlers.llm_hooks import MemoryHooksMixin
from .handlers.llm_tools import MemoryToolsMixin
from .storage.db import Database
from .storage.schema import apply_migrations


@register(
    "astrbot_plugin_ob_memory",
    "You",
    "这是关于你们的点滴",
    "v1.1.0",
    "https://github.com/L1ke40oz/astrbot_plugin_ob_memory",
)
class MemoryPlugin(
    MemoryToolsMixin,
    MemoryHooksMixin,
    MemoryCommandsMixin,
    Star,
):
    """Top-level Star wrapper for the memory plugin.

    Composes the runtime components:

    - SQLite storage (``storage.db`` + ``storage.schema``)
    - ``MemoryManager`` — CRUD + touch + time ripple
    - ``EmbeddingService`` + ``Tagger`` — vector search & LLM analysis
    - ``DecayEngine`` — background scoring + archival loop
    - ``SearchService`` + ``SurfaceStrategy`` — dual-channel retrieval
    - ``MemoryWriter`` — high-level ``hold`` / ``hold_feel``
    - Dashboard server — embedded HTTP UI
    - LLM tool / hook / command mixins for AstrBot integration
    """

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.context = context
        # Read UI config from _conf_schema.json (passed by AstrBot framework)
        self.config: dict = dict(config) if config else {}

        # Resolve plugin data directory (managed by AstrBot, survives upgrades)
        self.data_dir: Path = StarTools.get_data_dir()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path: Path = self.data_dir / "memory.db"

        # Components are created in ``initialize`` so async setup happens
        # inside the AstrBot event loop.
        self.db: Database | None = None
        self.manager: MemoryManager | None = None
        self.embedding: EmbeddingService | None = None
        self.tagger: Tagger | None = None
        self.writer: MemoryWriter | None = None
        self.search: SearchService | None = None
        self.surface: SurfaceStrategy | None = None
        self.decay: DecayEngine | None = None
        self.dashboard: DashboardServer | None = None
        # Session resolver is cheap and stateless beyond ``self`` —
        # build it eagerly so every handler can call it (some handlers
        # may fire before ``initialize`` finishes wiring storage).
        self.session_resolver: SessionResolver = SessionResolver(self)
        self._dehydrator: Dehydrator | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def initialize(self) -> None:
        """Open the database, build core components, start decay engine."""
        try:
            self.db = Database(self.db_path)
            await self.db.connect()
            await apply_migrations(self.db)
            logger.info(
                f"[memory] storage ready at {self.db_path} (schema migrations applied)"
            )

            self.manager = MemoryManager(self.db)

            # ---------- Embedding ----------
            embedding_provider = EmbeddingService.resolve_provider(
                self.context,
                provider_id=str(self.config.get("embedding_provider_id", "")).strip(),
            )
            self.embedding = EmbeddingService(
                self.db,
                provider=embedding_provider,
                context=self.context,
                provider_id=str(self.config.get("embedding_provider_id", "")).strip(),
            )
            if not self.embedding.enabled:
                logger.info(
                    "[memory] embedding provider not available; "
                    "search will run keyword-only and merge detection disabled"
                )

            # ---------- Tagger ----------
            analyze_provider_id = str(self.config.get("analyze_provider_id", "")).strip()
            self.tagger = Tagger(self.context, analyze_provider_id=analyze_provider_id)

            # ---------- Dehydrator ----------
            self._dehydrator = Dehydrator(self.tagger)

            # ---------- Writer ----------
            tagging_enabled = bool(self.config.get("tagging_enabled", True))
            merge_threshold = float(self.config.get("merge_threshold", 0.85))
            digest_prompt = str(self.config.get("digest_prompt", "")).strip()
            self.writer = MemoryWriter(
                self.manager,
                tagger=self.tagger,
                embedding=self.embedding,
                merge_threshold=merge_threshold,
                tagging_enabled=tagging_enabled,
                merge_enabled=tagging_enabled,
                digest_prompt=digest_prompt,
            )

            # ---------- Search + Surface ----------
            self.search = SearchService(self.manager, embedding=self.embedding)

            decay_cfg = self._build_decay_config()
            self.surface = SurfaceStrategy(self.manager, decay_config=decay_cfg)

            # ---------- Decay engine ----------
            self.decay = DecayEngine(self.manager, decay_cfg)
            await self.decay.start()

            # ---------- Dashboard ----------
            webui_config = self.config.get("webui", {})
            if not isinstance(webui_config, dict):
                webui_config = {}
            enabled = bool(
                webui_config.get("enabled", self.config.get("enabled", True))
            )
            if enabled:
                try:
                    host = str(
                        webui_config.get("host", self.config.get("host", "127.0.0.1"))
                    )
                    port = int(webui_config.get("port", self.config.get("port", 2140)))
                    self.dashboard = DashboardServer(self, self.data_dir)
                    await self.dashboard.start(host=host, port=port)
                    # Log the access URL
                    display_host = "localhost" if host == "127.0.0.1" else host
                    logger.info(f"[memory] Dashboard: http://{display_host}:{port}/")
                except Exception as e:
                    logger.warning(
                        f"[memory] Dashboard 启动失败，插件功能不受影响: {e}"
                    )
                    self.dashboard = None

        except Exception as e:
            logger.error(f"[memory] failed to initialize: {e}")
            await self._safe_terminate()
            raise

    async def terminate(self) -> None:
        """Stop background tasks and close the database connection."""
        await self._safe_terminate()

    async def _safe_terminate(self) -> None:
        """Idempotent shutdown — safe to call from initialize() failure path."""
        if self.dashboard is not None:
            try:
                await self.dashboard.stop()
            except Exception as e:
                logger.warning(f"[memory] dashboard.stop raised: {e}")
            finally:
                self.dashboard = None

        if self.decay is not None:
            try:
                await self.decay.stop()
            except Exception as e:
                logger.warning(f"[memory] decay.stop raised: {e}")
            finally:
                self.decay = None

        if self.db is not None:
            try:
                await self.db.close()
                logger.info("[memory] storage closed cleanly")
            except Exception as e:
                logger.warning(f"[memory] error while closing storage: {e}")
            finally:
                self.db = None

        self.manager = None
        self.embedding = None
        self.tagger = None
        self._dehydrator = None
        self.writer = None
        self.search = None
        self.surface = None

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    def _build_decay_config(self) -> DecayConfig:
        """Build the decay engine config from UI config overrides."""
        return DecayConfig(
            decay_lambda=float(self.config.get("decay_lambda", 0.05)),
            archive_threshold=float(self.config.get("decay_archive_threshold", 0.3)),
            check_interval_hours=float(
                self.config.get("decay_check_interval_hours", 24.0)
            ),
            emotion_base=float(self.config.get("decay_emotion_base", 1.0)),
            arousal_boost=float(self.config.get("decay_arousal_boost", 0.8)),
        )

    # ==================================================================
    # LLM Tools — defined here so __module__ matches the plugin path.
    # Implementation logic lives in MemoryToolsMixin; these thin wrappers
    # just delegate to the mixin methods with the correct decorator.
    # ==================================================================
    @filter.llm_tool(name="record_memory")
    async def record_memory(
        self,
        event: AstrMessageEvent,
        content: str,
        importance: int = 5,
        tags: str = "",
        pinned: bool = False,
    ) -> str:
        """记住一件事。AstrBot 会把这条记忆和当前会话绑定，并在以后相关对话里再带出来。

        Args:
            content(string): 要记住的内容，越具体越好。包含人物、时间、感受、待办等具体信息。
            importance(number): 1-10 的整数，1 表示水话别记、10 表示核心准则永不忘。默认 5。
            tags(string): 逗号分隔的关键词，方便日后检索；可以留空让系统自动生成。
            pinned(boolean): 是否钉选为永久核心准则；钉选后永不衰减。默认 false。
        """
        return await MemoryToolsMixin.record_memory(
            self, event, content, importance, tags, pinned
        )

    @filter.llm_tool(name="record_feel")
    async def record_feel(
        self,
        event: AstrMessageEvent,
        content: str,
        source_bucket_id: str = "",
        valence: float = -1.0,
    ) -> str:
        """记下你（模型）从某段记忆里带走的感受。这与事件本身是分开的。

        Args:
            content(string): 你想记的第一人称感受、领悟或未解的疑问。例如「我从中看到了她的成长」。
            source_bucket_id(string): 这段感受对应的源事件桶 id；提供后会把源事件标为已消化。可留空。
            valence(number): 你对这段感受的效价 0.0-1.0；0 极负、0.5 中性、1 极正。-1 表示不指定。
        """
        return await MemoryToolsMixin.record_feel(
            self, event, content, source_bucket_id, valence
        )

    @filter.llm_tool(name="recall_memory")
    async def recall_memory(
        self,
        event: AstrMessageEvent,
        query: str = "",
        domain: str = "",
        limit: int = 10,
        importance_min: int = 0,
    ) -> str:
        """主动从记忆里检索相关内容。返回 top 命中条目供你接续回应。

        Args:
            query(string): 关键词或自然语言查询；可关于人物、事件、感受。可留空。
            domain(string): 可选的主题域过滤，例如「求职」「内心」。
                特别地传 "feel" 会进入 feel 独立通道，按时间倒序返回所有「感受」记忆。
            limit(number): 最多返回几条，默认 10，上限 20。
            importance_min(number): 1-10。设为 ≥1 时进入「批量拉重要记忆」模式：
                跳过语义检索，按 importance 降序返回 importance≥此值的桶。默认 0 表示关闭。
        """
        return await MemoryToolsMixin.recall_memory(
            self, event, query, domain, limit, importance_min
        )

    @filter.llm_tool(name="forget_memory")
    async def forget_memory(
        self,
        event: AstrMessageEvent,
        bucket_id: str,
        mode: str = "resolve",
    ) -> str:
        """让一段记忆退场。默认是「沉底」（仍可被关键词唤醒），传 mode=delete 才彻底删除。

        Args:
            bucket_id(string): 要操作的记忆桶 id；可以从 recall_memory 返回里拿到。
            mode(string): 处理方式：resolve 表示标记已解决（推荐），delete 表示永久删除（不可恢复）。
        """
        return await MemoryToolsMixin.forget_memory(self, event, bucket_id, mode)

    @filter.llm_tool(name="record_diary")
    async def record_diary(self, event: AstrMessageEvent, content: str) -> str:
        """把一大段日记/长文本拆分为多条独立记忆。适合用户一次倾诉很多内容时。

        系统会自动识别其中的事件、感受、决定、未完结的事，分别作为独立记忆存储。
        每条独立走一遍合并检测，相似话题会自动合并到已有桶。

        Args:
            content(string): 一段较长的内容；典型场景是用户当日的多件事汇总，或一段反思。
        """
        return await MemoryToolsMixin.record_diary(self, event, content)

    @filter.llm_tool(name="reflect_memory")
    async def reflect_memory(self, event: AstrMessageEvent, limit: int = 10) -> str:
        """自省/做梦：读取最近几条记忆，引导你用第一人称想想哪些事还有重量、哪些可以放下。

        系统会返回最近创建的记忆 + 自省引导词；你看完后可以：
        - 觉得可以放下的：调用 forget_memory(bucket_id, mode='resolve') 让它沉底
        - 有沉淀的：调用 record_feel(content="...你的感受...", source_bucket_id="...") 写下感受
        - 没有沉淀就不写，不强迫产出

        典型用法：每次新对话开头调一次，把过去几天的事消化一下，然后开始正常对话。

        Args:
            limit(number): 最多读取几条最近记忆，默认 10，上限 20。
        """
        return await MemoryToolsMixin.reflect_memory(self, event, limit)

    # ==================================================================
    # LLM Hooks — defined here so __module__ matches the plugin path.
    # ==================================================================
    @filter.on_llm_request()
    async def memory_on_llm_request(
        self, event: AstrMessageEvent, req: ProviderRequest
    ) -> None:
        """Augment the next LLM call with relevant memory context."""
        await MemoryHooksMixin.memory_on_llm_request(self, event, req)

    @filter.on_llm_response()
    async def memory_on_llm_response(
        self, event: AstrMessageEvent, response: LLMResponse
    ) -> None:
        """Optionally schedule a background auto-record task."""
        await MemoryHooksMixin.memory_on_llm_response(self, event, response)

    # ==================================================================
    # Commands — defined here so __module__ matches the plugin path.
    # ==================================================================
    @filter.command_group("memory")
    def _memory_group(self):
        """长期记忆管理。/memory help 查看子指令。"""

    @_memory_group.command("list")
    async def cmd_memory_list(self, event: AstrMessageEvent, limit: int = 10):
        """列出当前会话最近活跃的记忆桶。"""
        async for result in MemoryCommandsMixin.cmd_memory_list(self, event, limit):
            yield result

    @_memory_group.command("search")
    async def cmd_memory_search(self, event: AstrMessageEvent, query: str = ""):
        """关键词或语义搜索。/memory search 实习"""
        async for result in MemoryCommandsMixin.cmd_memory_search(self, event, query):
            yield result

    @_memory_group.command("pin")
    async def cmd_memory_pin(self, event: AstrMessageEvent, bucket_id: str = ""):
        """切换钉选状态。钉选的桶不衰减不合并。"""
        async for result in MemoryCommandsMixin.cmd_memory_pin(self, event, bucket_id):
            yield result

    @_memory_group.command("forget")
    async def cmd_memory_forget(self, event: AstrMessageEvent, bucket_id: str = ""):
        """让一段记忆沉底（仍可被关键词唤醒）。"""
        async for result in MemoryCommandsMixin.cmd_memory_forget(
            self, event, bucket_id
        ):
            yield result

    @_memory_group.command("delete")
    async def cmd_memory_delete(
        self, event: AstrMessageEvent, bucket_id: str = "", confirm: str = ""
    ):
        """永久删除一段记忆（不可恢复，需二次确认）。"""
        async for result in MemoryCommandsMixin.cmd_memory_delete(
            self, event, bucket_id, confirm
        ):
            yield result

    @_memory_group.command("clear")
    async def cmd_memory_clear(self, event: AstrMessageEvent, confirm: str = ""):
        """清空当前会话的所有记忆（管理员，需二次确认）。"""
        async for result in MemoryCommandsMixin.cmd_memory_clear(self, event, confirm):
            yield result

    @_memory_group.command("stats")
    async def cmd_memory_stats(self, event: AstrMessageEvent):
        """显示当前会话的记忆系统状态。"""
        async for result in MemoryCommandsMixin.cmd_memory_stats(self, event):
            yield result

    @_memory_group.command("help")
    async def cmd_memory_help(self, event: AstrMessageEvent):
        """显示子指令列表。"""
        async for result in MemoryCommandsMixin.cmd_memory_help(self, event):
            yield result

    @_memory_group.command("import_astrbot")
    async def cmd_memory_import_astrbot(
        self,
        event: AstrMessageEvent,
        file_path: str = "",
        max_pairs: int = 30,
    ):
        """从 AstrBot 导出的 JSONL 历史中提取记忆。"""
        async for result in MemoryCommandsMixin.cmd_memory_import_astrbot(
            self, event, file_path, max_pairs
        ):
            yield result

    @_memory_group.command("summarize")
    async def cmd_memory_summarize(self, event: AstrMessageEvent, rounds: int = 0):
        """从当前对话上下文中提取记忆。"""
        async for result in MemoryCommandsMixin.cmd_memory_summarize(
            self, event, rounds
        ):
            yield result
