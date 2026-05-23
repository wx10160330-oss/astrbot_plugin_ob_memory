# AstrBot Memory Plugin Handoff

## 目标
继续基于旧插件 `D:\Savedfiles\00 files\AI\astrbot\plugin\memory\1\astrbot_plugin_ob_memory` 开发，不重写。

核心目标有两个：
1. **修复自动记忆逻辑**：不要再把 `用户 + 助手` 原始对话直接写入记忆，而是整理/总结后再入库。
2. **替换 dashboard 前端**：复用当前目录里 Ombre Brain 的 WebUI，优先使用 `ombre-brain-update-2026-05-20_163329\app.py` 那套前端，并给插件补 API 兼容层。

---

## 当前状态（2026-05-23，本窗口已实际修改）

### 已完成 1：auto-record 不再写 raw transcript
已修改文件：
- `handlers/llm_hooks.py`
- `handlers/commands.py`
- `tests/test_llm_hooks.py`
- `tests/test_commands.py`

实际改动：
- `handlers/llm_hooks.py:_auto_record_task()` 已不再执行：
  ```python
  content = f"用户: {user_msg}\n助手: {assistant_msg}"
  await self.writer.hold(...)
  ```
- 现在改为复用整理格式，把单轮对话转成：
  - `对方(用户)说: ...`
  - `我(AI)回应: ...`
- 然后走 `writer.hold_diary()`，也就是摘要/拆分路径。
- `handlers/commands.py` 中把原来的 `_format_digest_pairs()` 提成了可复用的 `format_digest_pairs()`，并保留 `_format_digest_pairs()` 作为兼容包装。

当前意义：
- 自动记忆已从“原始对话直存”切到“整理后再入库”，这是本项目最核心的修复。

### 已完成 2：dashboard 后端兼容 API 已基本补齐
已修改文件：
- `dashboard/server.py`
- `tests/test_dashboard_smoke.py`

此前已补上的兼容路由：
- `GET /api/memories`
- `POST /api/memories`
- `PUT /api/memories/{bucket_id}`
- `DELETE /api/memories/{bucket_id}`
- `POST /api/analyze`
- `POST /api/grow`
- `GET /api/status`
- `GET /api/config`
- `PUT /api/config`
- `GET /api/session_status`
- `PUT /api/password`
- `POST /api/login`
- `POST /api/logout`
- `GET /manifest.json`
- `GET /icons/{filename}`

本窗口继续补上的兼容路由：
- `POST /api/backfill_embeddings`
- `GET /api/export/config`
- `GET /api/export/memories`
- `POST /api/import/config`
- `POST /api/import/memories`
- `GET /api/backups/list`
- `POST /api/backups/delete`

本窗口实际落地语义：
- `backfill_embeddings`
  - 支持 `dry_run`
  - 可扫描缺失向量并执行补建
- `export/config`
  - 导出插件版 **runtime-only** 兼容配置快照
- `export/memories`
  - 导出插件自有 zip 格式：`manifest.json + memories.json + embeddings.json`
- `import/config`
  - 只映射可识别字段回 `plugin.config` 与 `writer` 运行时参数
- `import/memories`
  - 支持 `dry_run / merge / replace`
  - 非 dry-run 会先自动生成 dashboard 级备份
- `backups/list`
  - 只列白名单前缀的 dashboard 备份文件
- `backups/delete`
  - 拒绝 `..` 与路径分隔符，避免路径穿越

额外已做：
- `dashboard/server.py` 的静态文件返回从“按文本读取”改成了 `FileResponse`，避免 png/svg/manifest 被当文本发出。
- `/icons/{filename}` 做了简化兼容：会优先从 `dashboard/static/` 找文件，并对 `icon_light_192.png` 这类请求尝试回退到同名 `.svg`。
- `/api/config` 仍然是**运行时兼容版**，不是 Ombre Brain 原版那种 `config.yaml` 完整持久化语义。

当前意义：
- **后端兼容层已经不是第一批“核心版”了，而是把 Ombre update 前端依赖的那批缺口接口基本补齐了。**
- 现在 dashboard 的主要剩余工作点已经从“补后端 API”转向“前端可见入口与交互是否完整恢复”。

### 已完成 3：新补 API 的 smoke test 已补并通过
已修改文件：
- `tests/test_dashboard_smoke.py`

本窗口补的验证覆盖：
- 路由注册存在性检查
- `export/config`
- `export/memories`
- `backfill_embeddings` dry-run / execute
- `import/config`
- `import/memories` dry-run / merge
- `backups/list`
- `backups/delete`
- 非法备份名拒绝
- `manifest.json`
- `/icons/{filename}`

验证结果：
- 定向 smoke：`15 passed`
- `dashboard/server.py` 编译检查已过（`python -m compileall`）

### 已完成 4：脉搏页可见入口已恢复，旧说明文案已清掉
已修改文件：
- `dashboard/static/index.html`
- `tests/test_dashboard_smoke.py`

本窗口实际改动：
- `dashboard/static/index.html:renderPulse()` 已新增可见操作入口：
  - 导出配置
  - 导出记忆
  - 导入配置
  - 合并记忆
  - 替换记忆
  - 补建向量
  - 备份列表
- 同一区域已加入隐藏 file input，直接绑定：
  - `importConfigFile()`
  - `importMergeMemoryFile()`
  - `restoreReplaceFile()`
- 已删除旧提示文案：
  - `当前插件版先开放核心记忆管理、分析和配置功能；备份、导入恢复、缺失向量补建等入口后续再接。`
- `tests/test_dashboard_smoke.py` 新增静态 smoke 断言：
  - 检查 `btn_export_config`
  - 检查 `btn_export_memories`
  - 检查 `btn_fix_emb`
  - 检查导入/恢复/备份相关函数入口字符串仍存在
  - 检查 `后续再接` 文案已不存在

当前意义：
- **“后端接口已在、前端 helper 已在、但入口没露出来” 这个阶段已经过去。**
- 现在 dashboard 剩余主问题进一步收敛为：
  - 这些入口虽然已经可见，但是否全部真的能点击走通，还没有实际人工验证
  - 因此还不能宣称 Ombre 前端移植彻底完成

### 已完成 5：登录后 memories 排序不再因超久远记忆溢出
已修改文件：
- `core/decay_engine.py`
- `tests/test_decay_engine.py`

本窗口实际改动：
- `core/decay_engine.py` 中给短期/长期 crossfade 的 sigmoid 计算补了数值稳定写法
- 不再直接在 `alpha = 1 / (1 + exp(...))` 上对超大正数做 `exp()`
- 极久远记忆现在会自然收敛到 long-term 分支，而不是在 `/api/memories` 排序时报：
  - `OverflowError: math range error`
- `tests/test_decay_engine.py` 已补回归用例，覆盖“超久远记忆不会 overflow”

当前意义：
- 登录后 dashboard 拉取 `GET /api/memories` 时，不会再因为某条旧记忆把整个页面炸掉。
- 这属于 dashboard 可用性的阻断级修复。

### 已完成 6：dashboard 时间显示与编辑日期保存已修正
已修改文件：
- `dashboard/static/index.html`
- `dashboard/server.py`

本窗口实际改动：
- 前端新增 `normalizeDateValue()`，统一兼容：
  - 秒级 unix 时间戳
  - 毫秒级 unix 时间戳
  - ISO 时间字符串
- `fmtDate()` 不再把秒级时间戳当毫秒解析，因此列表时间不再落到 `1970-01-21` 一类错误日期
- 编辑面板里的 `datetime-local` 初始值已改为用本地时间正确格式化，而不是直接对原始字段 `slice(0,19)`
- 后端 `PUT /api/memories/{bucket_id}` 已补 `created` / `created_at` 的更新支持
- 当编辑面板提交 `datetime-local` 字符串时，后端会转成时间戳再写回数据库

当前意义：
- recall 卡片里的日期显示现在和真实记忆时间一致。
- 点“编辑”后修改记忆日期，现在会真正保存生效，而不是前端改了但后端忽略。

### 已完成 7：点击编辑时不再重播整张卡片的入场动画
已修改文件：
- `dashboard/static/index.html`

本窗口实际改动：
- 新增 `_suppressCardAnim` 标记
- 点击 `编辑 / 收起` 时会禁用当前次渲染的卡片 `slideUp` 动画
- 渲染结束后自动清掉该标记，保持首次加载、分页切换等场景仍可继续使用原有入场动画

当前意义：
- recall 页点击“编辑”时，卡片不会再像刷新列表一样重新向上弹一次
- 视觉效果会更接近“原位展开编辑面板”，减少打断感

### 已完成 8：设置/脉搏页统计卡片改为以后端统计为准
已修改文件：
- `dashboard/server.py`
- `dashboard/static/index.html`
- `dashboard/static/ombre_frontend_extracted.html`
- `API.md`
- `README.md`
- `PROMPTS_GUIDE.md`

本窗口实际改动：
- `/api/stats` 已补齐前端统计卡片需要的字段：
  - `today_new`
  - `week_new`
  - `max_activation`
- 前端 `loadMemories()` 会同步拉取 `/api/stats`，设置/脉搏页优先使用后端统计值。
- 当前前端兜底仍保留本地计算，但 `max_activation` 兜底已改为 `Number(...)`，避免字符串拼接或异常类型导致显示超大数字。
- 已清理本次改动中出现的无用 helper / 无用导入。
- `tests/test_dashboard_smoke.py` 定向验证通过：`18 passed`。

当前意义：
- 本日新增、本周新增不再依赖前端列表局部数据。
- 最高回想以全局后端聚合为准，不再受前端字段类型影响。

### 当前未完成：Ombre 前端主体已能用，但仍缺完整 UI 全流程人工验证
已修改文件：
- `dashboard/static/index.html`
- `dashboard/static/ombre_frontend_extracted.html`

真实现状要分开说：

1. **底层 JS helper 已基本在页面里**
   - 当前前端文件里已经能看到这些函数：
     - `fixEmbeddings()`
     - `exportConfig()`
     - `exportMemories()`
     - `importConfigFile()`
     - `importMergeMemoryFile()`
     - `restoreReplaceFile()`
     - `showBackupsList()`
     - `deleteBackup()`

2. **可见 UI 入口现在也已经补回来了**
   - `renderPulse()` 已经能看到：
     - 导出配置
     - 导出记忆
     - 导入配置
     - 合并记忆
     - 替换记忆
     - 补建向量
     - 备份列表
   - 且“后续再接”的旧提示已经移除

3. **但完整交互流仍未实机验证**
   - 也就是说：
     - 后端接口已到位
     - 前端 helper 已在
     - 前端可见入口也已补回
     - **但还没有跑 UI 去确认每个入口都真的可用**

当前意义：
- **不能诚实地说“全部移植完成了”。**
- 更准确的说法是：
  - 后端兼容 API 这一层已基本完成
  - 前端主体已接入 Ombre 风格代码
  - 前端可见入口已恢复
  - 但“完整 UI 全流程人工验证 + 必要的小修补”还没做完

---

## 用户最新决定
用户一开始接受了“先做核心版”的推荐；随后又明确同意：
- **继续把 Ombre Brain 前端尽量完整照搬过来**
- 也就是下一窗口应继续补齐缺失后端 API，而不是长期停留在裁剪版前端

另外用户还明确说过：
- **暂时不考虑群组记忆**

这意味着下一窗口在设计 dashboard 接口和默认 session 选择时，可以继续按单会话 / 单用户语义优先，不需要先解群组作用域问题。

---

## 为什么继续基于旧插件
旧插件已经不是壳子，而是完整的 AstrBot 原生实现，具备：
- 插件入口和生命周期：`main.py`
- LLM tools：`handlers/llm_tools.py`
- 自动注入与自动记录 hooks：`handlers/llm_hooks.py`
- dashboard 服务：`dashboard/server.py`
- SQLite 存储、衰减、检索、embedding、tagger、writer
- 一套现成测试

因此更适合续做，而不是回到 MCP 方案或重写插件。

---

## 前端来源选择

### 推荐来源
优先复用：
- `D:\Savedfiles\00 files\AI\astrbot\OB\ombre-brain-update-2026-05-20_163329\app.py`

参考对照：
- `D:\Savedfiles\00 files\AI\astrbot\OB\Ombre-Brain-改版\app.py`

### 选择原因
`update` 基本是改版 WebUI 的累计增强版：
- 保留原有主界面
- 多了更完整的认证/配置/备份/安全增强
- 更新说明见：
  - `D:\Savedfiles\00 files\AI\astrbot\OB\ombre-brain-update-2026-05-20_163329\更新说明.md`

### 注意
不要直接把单文件 Flask app 搬进插件。
应当：
1. 抽取其中前端 HTML/CSS/JS 到插件的 `dashboard/static/`
2. 保留插件自己的 Starlette dashboard 宿主
3. 在插件后端补前端所需 API

---

## 自动记忆问题定位

### 原始 bug 根因（已修）
文件：
- `handlers/llm_hooks.py`

关键函数：
- `_auto_record_task()`

之前逻辑是：
```python
content = f"用户: {user_msg}\n助手: {assistant_msg}".strip()
result = await self.writer.hold(session_id, content)
```

现在已经改成：
- 构造整理后的 digest text
- 通过 `format_digest_pairs()` 统一格式
- 调 `writer.hold_diary()` 入库

### 已复用的 helper
- `handlers/commands.py: format_digest_pairs`
- `handlers/commands.py: _format_digest_pairs`（兼容包装）

---

## 现在 dashboard 还差什么

### 1) 后端接口主缺口已补齐，前端入口也已恢复，但还差 UI 实测
这一步现在要拆开看：

- **后端 API 缺口**：本来 handoff 里列出的 7 个核心缺失接口，现在都已经补上。
- **前端入口恢复**：`renderPulse()` 中导出 / 导入 / 备份 / backfill 相关按钮现在都已可见。
- **最近新修的阻断交互问题**：
  - 超久远记忆导致 `/api/memories` 排序 overflow 的问题已修
  - recall 列表显示 `1970` 错误日期的问题已修
  - 编辑日期不生效的问题已修
  - 点击“编辑”时整张卡重复播放入场动画的问题已修
- **剩余问题**：不再是“有没有接口/有没有按钮”，而是：
  - 前端每个入口是否都真的点得通
  - 下载、上传、弹窗、toast、刷新这些交互是否与实际返回一致
  - 是否还存在 Ombre 原版前端依赖、但插件版尚未对齐的小细节

### 2) /api/config 仍然只是 runtime 兼容层
现状：
- `dashboard/server.py` 里的 `/api/config`、`/api/export/config`、`/api/import/config` 都已经可用
- 但它们本质上仍是让插件版前端读写一部分字段
- 目前更偏“运行时改 `plugin.config` 和 `writer` 参数”
- 还没有建立一套稳定的“持久化配置映射”策略

这意味着：
- 现在已经足够支撑 Ombre 风格前端的配置交互
- 但仍不能宣称与 Ombre 原版 `config.yaml` 语义完全等价

### 3) /api/status 语义仍偏简化
当前仍只是告诉前端：
- `ai_available`
- `model`

但 Ombre 原版 `/api/status` 更接近“测试 dehy 配置是否可用”。
如果要更像原版，后续仍可决定是否：
- 用当前 `Tagger` / provider 做真正探测
- 还是保持 lightweight 兼容

### 4) 缺的已经主要是 UI 实测与交互收尾
当前状态是：
- 页面里已经有导出 / 导入 / 备份 / backfill 对应 JS 函数
- `renderPulse()` 中相关按钮/入口也已经补回
- `后续再接` 的旧说明文案已删除

因此真正剩余工作更像是：
- 实际跑 dashboard UI，把这些入口逐条点通
- 确认导出下载文件名、导入确认弹窗、完成后的 toast / alert 是否符合预期
- 确认操作后页面状态是否会正确刷新
- 如有必要，再补少量前端文案或交互细节

建议重点验证路径：
- 登录
- 导出配置
- 导出记忆
- 导入配置
- 合并记忆
- 替换记忆
- 补建缺失向量
- 备份列表 / 删除备份

---

## 下一窗口最该继续做什么

### 第一优先：跑 UI，把已恢复的入口逐条验证成“可用完成态”
现在最优先的已经不是继续补 API，也不是继续补按钮，而是实际验证 `dashboard/static/index.html` 里的已恢复入口：
- 登录
- memories CRUD
- analyze / grow
- 导出配置 / 导出记忆
- 导入配置
- 合并记忆 / 替换记忆
- 补建缺失向量
- 备份列表 / 删除备份

验证时重点看：
- 下载是否真的触发
- 上传 zip 后的 dry-run / confirm / execute 是否顺畅
- 成功/失败提示是否合理
- 操作后页面数据是否刷新一致

### 第二优先：根据 UI 实测结果做小修补
如果实测中发现问题，下一窗口应优先做小范围修补，例如：
- 按钮状态文案
- toast / alert 提示
- 某些操作后未刷新
- 某些字段映射不完全对齐

注意：
- 目前 smoke test 证明的是接口存在性和基本行为，再加上静态前端入口检查。
- 这仍然不等于完整前端交互已经全部通了。
- 用户要的是“尽量完整照搬 Ombre 前端体验”，所以最终仍需要 UI 侧确认。

### 第三优先：按需要再补更像 Ombre 原版的细节语义
如果 UI 收尾后还想继续对齐原版，可再考虑：
- `/api/status` 是否做更真实的模型可用性探测
- `/api/grow` 返回字段是否再逐项比对 Ombre 原版
- `/api/config` 是否需要更清晰地区分 runtime-only 与持久化能力

### 第四优先：如果要对外宣称“移植完成”，必须补一次真实结项检查
至少满足以下条件后，才适合说“全部移植完成”：
1. 页面中相关入口都可见且可点
2. 关键导入/导出/备份/backfill 流程已实际跑过
3. 新增 API 有 smoke 覆盖
4. 前端入口存在性有静态 smoke 覆盖
5. 页面里不再保留“后续再接”的旧文案

---

## 当前代码位置速查

### 已改过的关键文件
- `handlers/llm_hooks.py`
- `handlers/commands.py`
- `dashboard/server.py`
- `dashboard/static/index.html`
- `dashboard/static/ombre_frontend_extracted.html`
- `tests/test_llm_hooks.py`
- `tests/test_commands.py`
- `tests/test_dashboard_smoke.py`

### 插件核心
- `main.py`
- `handlers/llm_tools.py`
- `core/memory_writer.py`
- `core/tagger.py`
- `core/memory_manager.py`
- `storage/db.py`

### Ombre Brain WebUI 源
- `D:\Savedfiles\00 files\AI\astrbot\OB\ombre-brain-update-2026-05-20_163329\app.py`
- `D:\Savedfiles\00 files\AI\astrbot\OB\ombre-brain-update-2026-05-20_163329\更新说明.md`
- `D:\Savedfiles\00 files\AI\astrbot\OB\Ombre-Brain-改版\app.py`

---

## 测试现状

### 已做过的低成本验证
- `dashboard/server.py` 跑过 `python -m compileall`，至少语法层面可过
- `tests/test_dashboard_smoke.py` 新补了前端静态入口 smoke：
  - 校验脉搏页相关按钮 ID / 入口函数字符串存在
  - 校验 `后续再接` 文案已移除
- `core/decay_engine.py` 的 overflow 修复已补回归测试：
  - 覆盖超久远记忆评分时不会再触发 `math range error`
- `dashboard/static/index.html`、`dashboard/server.py` 本窗口改动后已再次跑过：
  - `python -m compileall dashboard core tests`

### 还没做的验证
- 没有完整跑 dashboard UI
- 没有实际手点过新恢复的导入 / 导出 / 备份 / backfill 流程
- 没有完整跑 pytest（本地环境缺 `astrbot` 导致直接收集测试会失败）
- 新窗口如果要继续验证，最好优先做：
  - compileall / 定向 smoke
  - 再跑 dashboard UI 手测
  - 再看是否需要补测试桩或在 AstrBot 环境内跑

---

## 一句话总结给接手模型
**不要重写插件。auto-record 已修成 hold_diary/summary 路径；dashboard 已接入 Ombre update 前端，并补齐 export/import/backups/backfill 等兼容 API，脉搏页可见入口也已恢复。最近又修掉了超旧记忆导致的排序溢出、1970 时间显示、编辑日期不生效、编辑时卡片重复入场动画等问题。下一步应优先实际跑 UI，把现有入口逐条点通并做交互收尾。**
