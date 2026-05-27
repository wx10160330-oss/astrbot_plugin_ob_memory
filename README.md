# AstrBot Memory — 类人长期记忆插件

> 让 AI 伴侣的记忆像人一样：会衰减、会主动想起、能区分事件与感受、能识别"未完结的事"并在合适的时机重新浮现。

---

## 它在做什么

传统记忆插件的套路是「累积对话历史 → 周期性总结 → 检索时拿 top-K → 拼回 prompt」。这种做法解决了"信息能否被找回"，但解决不了 AI 伴侣最常见的痛点：**「说完一个话题就一直说这个话题」、「过去 resolved 的事还在反复浮现」、「重要的未完结的事却没有被惦记」**。

本插件的核心思路是给记忆**生命周期**：

- **会衰减**：基于改进版艾宾浩斯曲线，重要 / 高情感强度的记忆衰减更慢
- **会归档**：长期低活跃的记忆自动 archive，不会无限堆积
- **会沉底**：标记 `resolved=True` 的记忆权重 ×0.05，仍可被关键词唤醒但不会主动浮上来
- **会浮现**：未完结的高 arousal 记忆获得 ×1.5 紧迫度加成，自然在 system_prompt 里出现
- **会联想**：检索结果不足时，40% 概率漂浮一条「忽然想起」的旧低权重记忆，模拟人类随机联想
- **能消化**：模型可以为某段事件写下自己的「感受 (feel)」，源记忆被标记为 digested 加速淡化，feel 本身永不衰减
- **能整理**：长段倾诉（日记、回顾）可以一次性拆分为多条独立记忆
- **能自省**：模型可以在新对话开头读最近记忆，用第一人称想想哪些事还有重量、哪些可以放下
- **会想起**：每次 LLM 请求前，钩子自动按 Russell 情感坐标 + 时间近邻 + 重要度做四维加权检索 + 主动浮现，注入 system_prompt

设计灵感来自 [Ombre Brain](https://github.com/P0luz/Ombre-Brain) 项目（一个给 Claude 用的 MCP 长期情绪记忆系统），但重写成纯 AstrBot 插件：用 SQLite + AstrBot Provider 体系替代 Markdown + MCP + 独立服务，零外部部署。

---

## 与 AstrBot 内置 / 其他记忆插件的边界

| 角色 | 它在做什么 | 跟本插件冲突吗 |
|---|---|---|
| **AstrBot 内置 LongTermMemory** | 把当前群聊 / 会话最近 N 条消息塞进 system_prompt | 不冲突，关注的层不同（短期上下文 vs 跨对话情感记忆） |
| **astrbot_plugin_livingmemory** | BM25 + Faiss 混合检索 + LLM 总结 + 图谱记忆 | 功能有重叠（都做"长期记忆"），建议二选一启用 |
| **本插件** | Russell 情感坐标 + 衰减 + 主动浮现 + 事件/感受双层 | — |

简单选型：
- 想要 **大规模检索增强**（重视召回率、需要图谱）→ LivingMemory
- 想要 **AI 伴侣式情感记忆**（重视"像人"的浮现行为、未完结惦记）→ 本插件
- 两个同时开会让两份独立的记忆池互相打架，**不推荐共存**

---

## 快速上手

1. **复制插件目录**到 `AstrBot/data/plugins/astrbot_plugin_ob_memory/`
2. **安装额外依赖**（首次使用需要）：
   ```bash
   pip install -r data/plugins/astrbot_plugin_ob_memory/requirements.txt
   ```
   只装了 `rapidfuzz`（关键词检索）和 `starlette` / `uvicorn`（仪表盘）。
3. **重启 AstrBot**——首次加载时插件会在 `data/plugin_data/astrbot_plugin_ob_memory/` 下创建 `memory.db`（SQLite）。
4. **在 AstrBot Web 管理面板找到本插件的配置**，按需调整以下常用选项（默认面板里 9 项 + 仪表盘 2 项）：

| 配置项 | 默认值 | 含义 |
|---|---|---|
| `scope_mode` | `conversation` | 隔离粒度：`conversation` 每个对话窗口独立 / `user` 同一用户跨窗口/平台共享（群聊会按说话人切碎） / `origin` 同一群/私聊全员共享 / `hybrid` 私聊用 user 语义（同一用户跨窗口共享）+ 群聊用 origin 语义（一个群一份） |
| `unify_groups_into_user` | （留空） | **群聊⇄私聊记忆互通开关**。填一个用户 ID（一般你自己 QQ 号），群聊里所有人的对话都被重定向到 `user:{这个 ID}` 池，跟该用户的私聊**完全共用一份记忆**。留空 = 关闭（各自独立）。仅 `hybrid` / `user` / `origin` 模式下生效 |
| `embedding_provider_id` | （留空） | 用于向量检索的 Embedding 模型 ID。留空使用 AstrBot 默认；没配 embedding provider 时只走关键词检索 |
| `tagging_enabled` | `true` | 写入时是否调 LLM 自动打标 + 智能合并相似桶。关闭可省每次写入 1 次 LLM 调用 |
| `inject_memory_persona` | `true` | 是否在每次 LLM 请求前自动把「记忆行为」YAML 指引拼到 `system_prompt` 末尾，让模型像真人一样自然地用记忆功能（不必再粘到自己的人设里） |
| `memory_persona_text` | （留空） | 自定义上一项注入的内容。留空 = 用插件内置默认（YAML，介绍长期记忆 + 怎么记 / 怎么用）。要换语气、加例外、换风格直接在文本框里改，保存后下次对话即生效 |
| `auto_record_enabled` | `true` | 模型没主动调 `record_memory` 时，插件是否启发式 / 周期性兜底把对话存为记忆 |
| `auto_record_mode` | `every_n_turns` | 兜底模式：`every_n_turns`（推荐）每攒够 N 轮对话整体总结一次 / `per_turn` 每轮单独判定 / `disabled` 完全关掉兜底只信模型自己 `record_memory` |
| `auto_record_every_n_turns` | `20` | `every_n_turns` 模式下每 N 轮触发一次自动总结。计数器按**对话窗口（cid）独立累**——开新对话 = 从 0 重数，私聊群聊各自一份，跟 `scope_mode` 解耦。计数器持久化在 `memory.db` 的 `session_state` 表里，插件 / AstrBot 重启都不会清零；模型自己主动 `record_memory` 那一轮不计数（已攒进度保留）。推荐 15-30 |
| `auto_record_use_judge` | `true` | 仅 `per_turn` 模式生效：自动记录是否再调一次 LLM 判定「这值得记吗」。关闭可省 1 次 LLM 调用 |
| `dashboard_enabled` | `true` | 是否启用浏览器可视化管理面板 |
| `dashboard_host` | `127.0.0.1` | 仪表盘监听地址：`127.0.0.1` 仅本机 / `0.0.0.0` 局域网可访问 |
| `dashboard_port` | `2140` | 仪表盘端口 |

5. **触发记忆**：直接和 AI 对话即可。模型支持 function calling 时，它会在合适时机自己调用 `record_memory` 工具；不支持时，`auto_record` 会兜底。

---

## 新手必读：仪表盘上那堆按钮到底是啥

> 装完了打开仪表盘一脸懵？这节用大白话讲清楚每条「记忆」上那些图标和按钮，以及为啥有些事它会自动忘、有些事它就是不忘。

### 一条记忆有五种「状态」

打开仪表盘，每条记忆旁边都有几个小按钮和一个状态标签。可以把记忆当成 AI 脑子里的一张张便签——便签会有不同状态：

| 图标 | 状态 | 怎么进入这个状态 | 啥意思 |
|---|---|---|---|
| ✨ | **活跃中** | 默认状态，新写的都是这个 | AI 现在脑子里能用、能想起、能搜到的便签 |
| 💎 | **核心准则**（钉选 `pinned`） | 你点「💎 钉选」/ AI 听到你说「记住 XX」自己钉 | **永远忘不掉**。适合你的身份、长期约定、重要承诺 |
| 💭 | **感受** (`feel`) | **只有 AI 自己能写**——它对某件事产生情绪反应时记一条 | 跟普通记忆并行的"AI 的内心反应日记"（详见下面） |
| 💤 | **已沉底**（`resolved` 或 `archived`） | ① 你点「💤 沉底」 ② 这条便签太久没用、自动衰减下来 | AI 不再主动想起，但**没删**，关键词还能搜到 |
| 😶‍🌫️ | **已隐藏**（`digested`） | 你点「😶‍🌫️ 隐藏」 | 纯粹 UI 折叠，你看着不烦，AI 那边正常用 |

> **沉底 ≠ 删除**。沉底只是这条便签不再"自己飘到眼前"，但还在抽屉里。要彻底删点「👻 删除」。

### 为什么会自动沉底？谁决定的？

AI 每聊一轮，背后有个"**衰减引擎**"在跑（默认每天一次）。它会算每条便签的**当前权重**，公式大概长这样：

```
权重 = 重要度 × 时间衰减 × 激活次数加成
```

- **重要度**（写入时打的 1–10 分）越高，衰得越慢
- **时间衰减**：上次被用到越久，权重越低
- **激活次数**：越常被翻出来用，权重越高

每天一轮检查，权重低于某个值（默认 2.0）的就自动 💤 沉底——AI 自然就"淡忘了"。

**哪些便签不会被自动沉底：**
- 💎 钉选过的（pinned=True）：永不衰减
- 💭 感受（bucket_type=feel）：单独曲线，衰得更慢
- 任何 `permanent` 类型的：永不衰减

### 💭 感受（feel）到底是啥？为啥从来没出现过？

**普通记忆是 AI 记你的事，感受是 AI 记自己的反应。**

举个例子。如果你跟它说：

> "我今天被裁员了"

它可能会：
1. 调 `record_memory` 工具 → 记一条普通便签：「小明 今天被裁员了」
2. **同时**调 `record_feel` 工具 → 记一条感受便签：「听到 小明 被裁，心里咯噔一下，又心疼又担心」

第二条就是 feel。它会出现在仪表盘的「💭 感受」tab 里，每条都关联到那个事件便签（`source_id`）。

**为什么你的没触发过？**

`core/prompts.py` 里的提示词写得很克制：
> feel.when: 这件事让你心里**真的有具体反应**；一轮最多一条；没感受就不写。

如果你跟它聊的多是日常 / 工作 / 问问题，没啥强情绪刺激，AI 就压根不会调那个工具。它"没真感受到"，就不假装记一条。

**想强制看看效果？** 发一句情绪强烈的话试试：
- "我刚被裁了"
- "我妈住院了"
- "今天考试过了！🎉"
- "我和 xx 分手了"

下次回复后打开「💭 感受」tab 应该多一条。

**想让它更频繁一点？** 配置面板里找到 **「记忆行为」提示词内容** (`memory_persona_text`)，编辑那段 YAML，把 `feel.when` 改宽松：
```yaml
feel:
  when: 对方提到自己的事 / 心情 / 关系，就顺手记一条感受
```
保存即生效，下次对话开始 AI 就更愿意记 feel 了。

### 💎 钉选 / 💤 沉底 / 😶‍🌫️ 隐藏 怎么选？

| 我想做的事 | 应该用 |
|---|---|
| "AI 你给我记牢，这是我的底线 / 身份 / 重要承诺" | 💎 **钉选** |
| "这条记忆已经没意义了，不要再总扯到这事上来了" | 💤 **沉底** |
| "这条挺有用，但仪表盘看着烦，眼不见心不烦" | 😶‍🌫️ **隐藏** |
| "彻底不要了" | 👻 **删除** |
| "刚沉底的我后悔了，能找回来吗" | 沉底的便签上点「✨ 激活」 |

### 还有些你可能没注意的小东西

**🌟 权重数字**：每张活跃中的便签右侧那个 `🌟 权重 5.2`，就是上面那个衰减公式的当前值。看着权重慢慢往下走 = 这条在淡。突然涨 = AI 又用到了它。

**域 / domain 标签**：每条记忆都会自动打 1-2 个标签（家庭/工作/恋爱/情绪…），AI 根据内容自己分类。仪表盘上能按域筛选。

**时间涟漪**：一条记忆被用到时，**附近时间（±48h）创建的其他记忆也会被小幅加分**。模拟"一件事让你想起那阵子别的事"。这就是为啥有些便签明明没被直接提到，权重也在慢慢涨。

**情绪平面（Pulse）**：仪表盘有个圆形散点图，按 Russell 情感坐标（积极程度 × 唤醒度）把每条便签画在一张图上——你能直观看到你跟 AI 聊过的事**整体情绪分布**。

**「忽然想起」效应**：每次 AI 回复你前，会有 **40% 概率**从权重低的旧便签里漂浮一条上来——像人类突然走神想起某件旧事。所以聊着聊着 AI 偶尔会主动提到一件你以为它早忘的事。

### 💌 怎么在「私聊 / 群聊」之间切换

每个对话窗口（一个私聊、一个群聊）默认是**独立的记忆库**——这是 `scope_mode = conversation` 模式下的设计：私聊聊过的事 AI 不会带到群里说，反过来也一样。

仪表盘顶部有个**「💌 当前会话」下拉框**，下拉里列出所有有记忆的 session（私聊、各个群聊…），每条显示**条数 + 最近活跃时间**，你能直接点哪条记忆面板就显示哪条。

> 只有一个 session 的时候这个下拉框会自动隐藏（没东西可切换）。
>
> 切换后浏览器会记住选择，下次打开仪表盘还是停在你选的那个 session 上。

如果你想**永久锁定**到某个 session（比如只想看私聊的，群聊那个永远不显示），可以在插件配置 `dashboard_session_id` 里填那个 session_id（SSH 上服务器 `sqlite3 memory.db "SELECT DISTINCT session_id FROM memories;"` 能列出所有 session_id）。配置一旦填上，仪表盘永远以它为准，无视下拉框。

如果你想让**多个聊天窗口共用同一份记忆库**，把 `scope_mode` 改成：
- `user`：同一个用户在哪聊都共用记忆。⚠️ **群聊里会按每个说话人切碎**（每个人各一份 `user:他的QQ`，不是群共享）
- `origin`：群聊全员共用一份（同一个 QQ 群里所有人都能让 AI 想起群里发生过的事）。但私聊里**新开对话会换 session_id**
- **`hybrid`（推荐）**：私聊用 `user` 语义（同一用户跨窗口共享），群聊用 `origin` 语义（一个群一份全员共享，不按说话人切碎）。「私聊不掉记忆 + 群里全员共享」就选这个

### 🧮 自动总结计数器是怎么算的？（per-window）

`every_n_turns` 那个 N 轮一次的总结计数器**始终按对话窗口（cid）单独累**，跟 `scope_mode` 无关：

- 你**开新对话窗口** → 计数器从 0 开始（旧窗口的进度跟它无关）
- **私聊**和**群聊** → 各自独立的计数器
- 同一个群里的多次对话 → 各自独立

这条规则跟 `scope_mode` **解耦**——比如你 `scope_mode = user`，私聊里聊了 7 轮、切到群聊，群聊的计数器是 0 起步（不会继承私聊那 7 轮的进度）；但记忆库还是共用的，群聊的 AI 能调用私聊聊过的事。

> 即使 AstrBot 对某个适配器（比如某些群聊）不分配独立 cid，插件也会**按 origin 分开**计数（私聊一份、每个群一份），不会全部混到一起。

`/memory stats` 那行 `every_n_turns X/N（… · 本对话）` 就是当前对话窗口的进度。

### 三种「沉底」的细微差别

容易混的三个概念，一次讲清：

| 内部字段 | 谁能改 | 触发方式 | 仪表盘显示 |
|---|---|---|---|
| `resolved = True` | 你 + AI（AI 听到你说"算了"会自己改）| 「💤 沉底」按钮 | 💤 已沉底 |
| `bucket_type = "archived"` | 衰减引擎 | 权重低于阈值自动归档 | 💤 已沉底 |
| `bucket_type = "permanent"` | 模型主动写 / 你升级 | 不动 | 💎 核心准则 |

前两个**仪表盘合并显示在「💤 已沉底」tab**——你不用区分到底是手动还是自动。

---

## 进阶配置（高级用户）

普通用户用上面常用项就够了。要精调衰减曲线、检索权重、token 预算，**在 UI 配置里把 `advanced_mode` 开关打开**——12 个高级参数会立即出现在配置面板里（基于 AstrBot 的 `condition` 字段实现，无需任何额外文件）。

### `advanced_mode = true` 后多出来的参数

| 类别 | 参数 | 默认 | 说明 |
|---|---|---|---|
| 衰减引擎 | `decay_lambda` | 0.05 | 指数衰减速率（每天） |
| 衰减引擎 | `decay_archive_threshold` | 0.3 | 低于此值的桶下次扫描时归档 |
| 衰减引擎 | `decay_check_interval_hours` | 24 | 后台扫描间隔（0 = 禁用） |
| 衰减引擎 | `decay_emotion_base` / `decay_arousal_boost` | 1.0 / 0.8 | 情感权重计算系数 |
| 检索注入 | `max_search_results` | 3 | 关键词+向量召回的注入条数 |
| 检索注入 | `max_surface_results` | 2 | 主动浮现的注入条数 |
| 检索注入 | `injection_token_budget` | 1500 | system_prompt 里记忆块的 token 上限 |
| 检索注入 | `merge_threshold` | 0.85 | 向量相似度高于此值合并到现有桶 |
| 检索注入 | `random_drift_enabled` | true | 注入候选 < 3 条时 40% 概率漂浮一条「忽然想起」的旧低权重记忆，模拟人类联想 |
| 自动记录 | `auto_record_min_chars` | 30 | 用户消息少于此长度跳过自动记录 |
| 手动总结 | `summarize_default_rounds` | 0 | `/memory summarize` 不带参数时默认总结几轮（0 = 全部上下文） |
| 提示词 | `digest_prompt` | （留空） | 自定义日记/总结/导入的 LLM 提示词。留空使用内置默认（AI 第一人称视角） |
| 会话控制 | `disabled_sessions` | `[]` | session_id 在此列表则禁用注入与自动记录 |

每项 hint 会直接显示在 UI 上，按需调整即可。**所有配置改动需重启插件生效。**

### 自定义记忆提示词

本插件有两个可在 UI 直接编辑的提示词文本框，留空都使用内置默认值，修改后下次对话即生效，无需重启：

- **`memory_persona_text`（常用配置）**——「记忆行为」指引，每次跟模型对话前自动拼到 `system_prompt` 末尾。内置默认是一段简短的 YAML，告诉模型：「你拥有长期记忆 / 哪些该记 / 哪些不该记 / 怎么自然地用记忆工具」。**升级到本版本后建议把人设里那段同义内容删掉**，省下来的字数留给人物刻画。如果你的角色很特殊，可以直接在文本框里改语气、加例外、换格式。
- **`digest_prompt`（高级配置）**——`/memory summarize`、`/memory import_astrbot` 和 `record_diary` 用到的拆分总结提示词。默认以 **AI 第一人称视角**记忆（"我"=AI，"你"=用户），例如产出的记忆会是：
  > 你告诉我你拿到了 offer，我能感受到你的激动

  想改成其他视角或风格，文本框里直接改即可。

---

## 仪表盘（Dashboard）

启动 AstrBot 后，浏览器打开 `http://<dashboard_host>:<dashboard_port>/`（例如 `http://127.0.0.1:2140/`）即可进入。`/dashboard` 也指向同一个页面。

- **首次访问**会要求设置一个登录密码（最少 4 位）。也可以通过环境变量 `MEMORY_DASHBOARD_PASSWORD=your_password` 强制覆盖（此时 UI 不再允许改密码）。
- **能做什么**：浏览所有 session 下的记忆桶、按 session/类型/关键词过滤、查看详情、编辑（name/content/记忆日期/tags/domain/valence/arousal/importance/pinned/resolved/digested）、删除、全文搜索、看运行状态；脉搏页会展示本日新增、近 7 天新增、最高回想次数、领域分布、情感分布，并提供导出配置、导出记忆、导入配置、合并/替换导入记忆、补建缺失向量、备份列表与删除备份等入口。
- **前端交互**：当前仪表盘前端已经统一朝 Ombre Brain / Uluru Star 风格靠拢：普通提示逐步切到胶囊提示，`hold` 页的“分析并预览 → 确认保存”链路已使用胶囊风格提示，`grow` 页的“拆分预览 → 确认保存”交互也正在对齐同一套体验（提示、按钮位置、保存反馈保持一致）。
- **兼容状态**：当前插件版已接入 Ombre Brain 风格前端，并补齐该前端依赖的主要兼容 API；最近还修掉了超久远记忆导致的排序溢出、列表时间误显示为 1970、编辑日期不生效、点击“编辑”时卡片重复播放入场动画等问题，前端提示系统也在从旧 toast / modal 迁移到统一的胶囊体系。
- **不会做什么**：不接管 AstrBot 自身的认证体系，仅做记忆的 CRUD；不允许直接编辑 vector / session_id 等结构性字段。
- **安全性**：默认监听 `127.0.0.1`，仅本机可访问。改成 `0.0.0.0` 暴露到局域网时**务必**设置密码，并自行评估防火墙规则。会话 cookie 是 httpOnly + SameSite=Lax，重启会失效（密码持久化在 `data/plugin_data/astrbot_plugin_ob_memory/dashboard_auth.json`，文件权限 600）。
- **当前边界**：后端兼容层和主要前端入口已经到位，核心 CRUD、hold/grow、导入导出、备份、向量补建等路径都可用；但如果要对外宣称“前端完全定稿”，仍建议按真实 UI 流程把 grow 预览、批量操作确认、备份列表、导入替换确认等交互逐条手测一遍，确认它们都与新的胶囊式交互保持一致。

---

## 它到底怎么记的——架构速览

```
用户消息
    │
    ▼
on_llm_request 钩子 ──┐
    │                 │
    │   ┌─────────────▼─────────────┐
    │   │ Memory Persona 注入        │  把「记忆行为」YAML 指引拼到 system_prompt 末尾
    │   │ SearchService             │  关键词 (rapidfuzz) + 向量 (cosine) 双通道检索
    │   │ SurfaceStrategy           │  按 ActivationScore 主动浮现 pinned + 高权重未完结桶
    │   │ Random Drift              │  结果 < 3 时 40% 几率漂一条「忽然想起」
    │   └─────────────┬─────────────┘
    │                 │ 拼成 [persona] + [=== 长期记忆 ===] 块
    │                 ▼
    │             LLM Provider ←─── 模型看到记忆，可调用：
    │                 │              record_memory / record_feel / record_diary
    │                 │              recall_memory / reflect_memory / forget_memory
    │                 ▼
    │      回复 + 可能的工具调用
    │                 │
    └──── on_llm_response 钩子 ──┐
                                  │
              ┌───────────────────▼─────────────────────────────┐
              │  模型已主动 record_memory？                     │
              │  ├─ 是 → 这轮不计数（已攒进度保留）            │
              │  └─ 否 → 按 auto_record_mode 兜底：             │
              │       • every_n_turns: 计数+1，满 N 轮一次总结 │
              │       • per_turn:      启发式 + judge LLM       │
              │       • disabled:      不兜底                   │
              │                                                 │
              │  计数器键 = conv:{cid}（或 origin 兜底），跟    │
              │  scope_mode 解耦：每个窗口独立从 0 累起。       │
              └───────────────────┬─────────────────────────────┘
                                  │
                       MemoryWriter.hold()       MemoryWriter.hold_diary()
                                  │              （模型显式调 record_diary 时）
                ┌─────────────────┼─────────────────┐
                ▼                 ▼                 ▼
           Tagger.analyze   EmbeddingService    MemoryManager
           (auto-tag)       (embed + 找合并候选) (写 SQLite)

后台周期：DecayEngine 每 24h 跑一遍
    └─ 长期未活跃 + 低重要度 → auto-resolve（沉底 ×0.05）
    └─ ActivationScore < 0.3 → archive
```

### 核心模块（`core/`）

| 文件 | 职责 |
|---|---|
| `models.py` | `MemoryBucket` dataclass + `clamp_bucket` 钳制工具 |
| `memory_manager.py` | 纯 CRUD（带 session 隔离）+ touch + 时间涟漪 |
| `decay_engine.py` | `calculate_score` 纯函数 + 后台衰减循环 + auto-resolve |
| `embedding_service.py` | 向量生成（走 AstrBot Embedding Provider）+ BLOB 序列化 + cosine 搜索 |
| `tagger.py` | LLM 调用封装：`analyze` / `merge_content` / `judge_worth_recording` / `digest` |
| `dehydrator.py` | 记忆脱水压缩：注入前用 LLM 将长内容压缩为高密度摘要（带 LRU 缓存） |
| `search_service.py` | 关键词+向量双通道检索 + 四维加权排序 |
| `surface_strategy.py` | 主动浮现：pinned + 冷启动 + 衰减分排序 |
| `memory_writer.py` | 高层写入流程：`hold`（单条）+ `hold_feel`（感受）+ `hold_diary`（拆分） |
| `session_resolver.py` | 把 event 映射为 session_id（按 scope_mode） |
| `prompts.py` | LLM 模板：ANALYZE / MERGE / JUDGE / DIGEST + 自动注入的 MEMORY_PERSONA_PROMPT |

### 接入层（`handlers/` + `dashboard/`）

| 文件 | 职责 |
|---|---|
| `handlers/llm_hooks.py` | `@on_llm_request` 注入「记忆行为」persona + 记忆（含脱水压缩）+ 随机漂流；`@on_llm_response` 兜底自动记录（per_turn / every_n_turns / disabled 三模式） |
| `handlers/llm_tools.py` | 6 个 `@filter.llm_tool`：record_memory / record_feel / record_diary / recall_memory / reflect_memory / forget_memory |
| `handlers/commands.py` | `/memory` 指令组：list / search / summarize / import_astrbot / pin / forget / delete / clear / stats / help |
| `dashboard/server.py` | Starlette + uvicorn 嵌入式 HTTP 服务，提供 REST API + 单页前端 |
| `dashboard/auth.py` | 密码哈希 (SHA-256 + salt) + 内存会话 token + 环境变量覆盖 |

### 存储层（`storage/`）

`memory.db` 单 SQLite 文件，两张表：
- `memories`：主记忆桶
- `embeddings`：向量（BLOB 存 packed float32）+ 外键级联删除

升级时通过 `schema_version` 表做版本化迁移。

---

## 用户能用的指令

| 指令 | 用途 |
|---|---|
| `/memory list [N]` | 列出当前会话最近活跃的 N 条记忆（默认 10，最多 50） |
| `/memory search <关键词>` | 双通道搜索（关键词 + 向量） |
| `/memory summarize [N]` | 手动总结最近 N 轮对话为记忆（不传 N 则总结全部上下文） |
| `/memory import_astrbot <文件路径> [N]` | 从 AstrBot 导出的 JSONL 历史中提取记忆，`N` 为最多导入轮数（默认 30） |
| `/memory pin <id>` | 切换钉选状态。钉选的桶 importance 锁 10、永不衰减、永不合并 |
| `/memory forget <id>` | 沉底（标记 resolved=True，关键词仍可唤醒） |
| `/memory delete <id> [confirm]` | 永久删除（二次确认） |
| `/memory clear [confirm]` | 清空当前会话所有记忆（管理员，二次确认） |
| `/memory stats` | 当前会话状态 + 衰减引擎运行情况 |
| `/memory help` | 子指令清单 |

---

## LLM 工具（模型自主调用）

模型在对话中会看到 6 个工具，下面是它们的简要含义。普通用户不需要管这些，模型会按需调用。

| 工具 | 含义 |
|---|---|
| `record_memory(content, importance, tags, pinned)` | 记住一件事。`importance` 1-10，`pinned=True` 创建核心准则永不忘 |
| `record_feel(content, source_bucket_id, valence)` | 记下模型自己的感受。`source_bucket_id` 指向被消化的源记忆，会被标记 `digested` |
| `record_diary(content)` | 把一大段日记/长文本拆分为多条独立记忆。适合用户一次倾诉很多内容时 |
| `recall_memory(query, domain, limit, importance_min)` | 主动检索相关记忆。`domain="feel"` 进入感受独立通道；`importance_min>=1` 进入「批量拉重要记忆」模式 |
| `reflect_memory(limit)` | 自省/做梦：读最近几条记忆 + 引导自省 + embedding 连接提示 + feel 结晶提示。典型用法是新对话开头调一次 |
| `forget_memory(bucket_id, mode)` | mode='resolve' 沉底 / 'delete' 永久删除 |

**模型一定要主动调用这些吗？** 不一定。`auto_record_enabled=True` 时，即使模型没调 `record_memory`，插件后台也会启发式判断要不要存。但模型主动调用的质量更高，因为它带着对当前对话的理解。

**对话启动建议序列**（参考 Ombre Brain 的设计）：
1. `reflect_memory()` — 消化最近记忆，看有什么沉淀，决定要不要 `record_feel` / `forget_memory`
2. （可选）`recall_memory(domain="feel")` — 读以前留下的感受
3. 开始和用户说话；记忆注入由 `on_llm_request` 钩子自动完成

### 与 Ombre Brain 工具的映射关系

如果你之前用过 Ombre Brain，下面这张表帮你快速对应：

| Ombre Brain | 本插件 | 备注 |
|---|---|---|
| `breath()` 浮现 | 由 `on_llm_request` 钩子自动完成 | 不再需要模型主动调用，注入透明 |
| `breath(query=...)` | `recall_memory(query=...)` | 双通道搜索 |
| `breath(domain="feel")` | `recall_memory(domain="feel")` | 感受独立通道 |
| `breath(importance_min=N)` | `recall_memory(importance_min=N)` | 批量拉重要记忆 |
| `hold(content=...)` | `record_memory(content=...)` | 普通事件记忆 |
| `hold(feel=True, ...)` | `record_feel(...)` | 模型感受 |
| `grow(content=...)` | `record_diary(content=...)` | 长文本拆分 |
| `dream()` | `reflect_memory()` | 自省/做梦 |
| `trace(resolved=1)` | `forget_memory(bucket_id, mode='resolve')` | 沉底 |
| `trace(delete=True)` | `forget_memory(bucket_id, mode='delete')` | 永久删除 |
| `trace` 修改元数据 | 通过仪表盘编辑 | 不再走工具 |
| `pulse` | `/memory stats` 指令 + 仪表盘 | 拆分到用户层 |

差异说明：
- **浮现自动化**：Ombre Brain 要求模型在每次对话开头主动调 `breath()`；本插件改成由 AstrBot 钩子自动注入到 `system_prompt`，模型零负担
- **元数据编辑**：Ombre Brain 用 `trace` 工具改字段，本插件改用浏览器仪表盘点击编辑（更直观，且业务规则会经过 `MemoryManager.update` 统一钳制）
- **状态查看**：Ombre Brain 用 `pulse` 让模型读，本插件拆为用户的 `/memory stats` 指令 + 仪表盘统计栏，避免占用模型上下文

---

## 手动总结与历史导入

### `/memory summarize [N]`

手动触发对当前对话上下文的记忆提取。适合在一段重要对话结束后使用，确保关键信息不被遗漏。

- 不传 `N`：总结当前对话的全部上下文（或配置的 `summarize_default_rounds` 轮）
- 传 `N`：只总结最近 N 轮对话

流程：从 AstrBot 对话历史中提取 user-assistant 对话对 → 喂给 LLM 拆分为独立记忆条目 → 每条走 merge 检测（相似内容自动合并到已有桶）。

### `/memory import_astrbot <文件路径> [N]`

从 AstrBot 导出的 `.jsonl` 对话历史文件中批量提取记忆。

- 在 AstrBot Web 管理面板导出对话历史（JSONL 格式）
- 执行 `/memory import_astrbot D:\path\to\export.jsonl 50`（最多导入 50 轮）
- 系统会自动过滤系统提示、RAG 注入块等非用户内容
- 记忆以 AI 第一人称视角记录（"我"=AI，"你"=用户）

---

## 调用成本调优

每个值得记的对话回合，本插件相比传统记忆插件**多 1~2 次 LLM 调用 + 1 次 embedding**。如果你重视成本，按需关闭：

| 场景 | 关哪个 | 损失什么 |
|---|---|---|
| 完全信任模型自主决定 | `auto_record_mode = disabled` 或 `auto_record_enabled = false` | 模型没主动 `record_memory` 时不会兜底（可能漏记） |
| 不要每轮单独判定，省 LLM 调用 | `auto_record_mode = every_n_turns`（默认）| 单轮的细节可能合并进 N 轮总结里 |
| 用 `per_turn` 但不要 LLM 复核 | `auto_record_use_judge = false` | 启发式通过即记，误记率略升 |
| 不需要情感坐标 / 智能合并 | `tagging_enabled = false` | 元数据全用默认（domain="未分类" 等），相似话题会重复建桶 |
| 不需要随机联想 | `random_drift_enabled = false`（高级配置）| 检索结果不足时不会主动漂浮旧记忆，输出更可预测 |

三个核心 toggle 全关 + `auto_record_enabled=false` 时，每个回合的 LLM 调用回到「主对话 + tool 后续 = 2 次」，跟最朴素的 RAG 插件持平。但同时也失去了"像人一样记忆"的几乎所有特性。

**典型回合的 LLM 调用次数**（默认 `auto_record_mode=every_n_turns`）：
- 注入阶段：0 次（搜索是纯 embedding/keyword）
- 主对话：1 次
- 模型主动调 `record_memory`：触发 1 次 analyze + 可能 1 次 merge = 1~2 次（同时把 every_n_turns 计数器归零，避免重复）
- 模型没调 `record_memory`、计数器尚未达阈值：本轮 0 次额外调用，计数器 +1
- 模型没调 `record_memory`、计数器满 N 轮：1 次 digest + 每条 merge 检测（跟 `/memory summarize N` 同一套）

切到 `per_turn` 时每轮另起 1 次 judge + 通过后 1 次 analyze + 可能 1 次 merge = 每轮 2~3 次（旧行为，调用最多）。
- 模型调 `record_diary`：1 次 digest + N 次 merge（每条目独立合并检测，无 analyze 重复）

随机漂流不调 LLM；reflect_memory 不调 LLM（连接提示和结晶提示都基于已有 embedding）；recall_memory 在 `domain="feel"` / `importance_min` 模式下也不调 LLM。

---

## 数据存储

```
data/plugin_data/astrbot_plugin_ob_memory/
├── memory.db                  # 主数据库（不要直接编辑）
├── memory.db-wal              # SQLite WAL 模式临时文件
├── memory.db-shm              #
└── dashboard_auth.json        # 仪表盘密码（盐 + 哈希），文件权限 600
```

**备份**：把整个目录拷走即可，所有记忆 + 向量 + 仪表盘密码都在里面。**迁移**：把目录复制到新机器对应位置即可，无需任何特殊步骤。

**不要直接编辑 `memory.db`**——通过 `/memory delete` / `/memory clear` 指令、或仪表盘的删除按钮来管理。

---

## 隔离粒度（scope_mode）

这是插件最核心的语义之一。用 `scope_mode` 决定「记忆按什么粒度隔离」：

| 模式 | session_id 形式 | 适合谁 |
|---|---|---|
| `conversation` (默认) | `conv:{cid}` | 每个对话窗口独立。AI 伴侣场景推荐——跟 ChatGPT/Claude 心理模型一致 |
| `user` | `user:{sender_id}` | 同一用户跨窗口、跨平台共享。想要"无论我在哪聊都是同一个 AI"的用户选这个。⚠️ **在群聊里会按每个说话人切碎**——群里每个人有自己的 `user:` session，不是全员共享 |
| `origin` | `unified_msg_origin` | 同一群/私聊共享。群聊场景下需要全员看到同一份记忆 |
| `hybrid` | 私聊 → `user:{sender_id}` / 群聊 → `unified_msg_origin` | **私聊跨窗口共享（同一用户开新对话不丢记忆）+ 群聊一个群一份（全员共享，不按说话人切碎）**。同时要这两种语义时选这个 |

**重要不变量：切换 `scope_mode` 不会迁移已有数据。** 在 `conversation` 模式下记的东西，切到 `user` 后会"看不见"——切回去就能再看到。这避免了"切一下模式就把所有记忆搅乱"的事故。

> **`hybrid` vs `user` 在群聊的区别**：`user` 下群里每个说话人各自一个 `user:他的QQ` session；`hybrid` 下整个群共用一个 `aiocqhttp:GroupMessage:群号` session。私聊侧两者行为一致，都是 `user:你QQ` 跨窗口共享。

### 🔗 让群聊和私聊记忆**互通**（`unify_groups_into_user`）

默认情况下，`hybrid` / `user` / `origin` 模式下私聊池和群聊池是**独立**的——AI 在群里不知道你私聊聊过啥，反之亦然。

如果想让两边互通，在配置面板填 `unify_groups_into_user = 你的QQ号`：

| 事件 | 关闭（默认） | 开启（填了你 QQ） |
|---|---|---|
| 你私聊 → bot | `user:你QQ` | `user:你QQ` |
| 你在群里说话 | 按 `scope_mode` 决定 | **`user:你QQ`**（重定向） |
| 群友 A 在群里说话 | 按 `scope_mode` 决定 | **`user:你QQ`**（重定向，**合并进你池子**） |
| 其他用户 B 跟 bot 私聊 | `user:B的QQ` | `user:B的QQ`（**不动**） |

效果：
- ✅ AI 在群里能想起你私聊聊过的事
- ✅ AI 在私聊能想起群里发生过的事（包括群友说的话、AI 在群里的回复）
- ✅ 其他人单独跟 bot 的私聊不受影响（不会污染你的池子）

> ⚠️ 用这个的前提是你**接受群友的发言被记进你的记忆库**。如果群友说了你不想记的，可以在仪表盘里手动「💤 沉底」或「删除」掉那条。

---

## 测试与开发

```bash
# 跑测试套件（不需要 AstrBot 在运行）
python -m pytest data/plugins/astrbot_plugin_ob_memory/tests -q

# 格式化 + 静态检查
ruff format data/plugins/astrbot_plugin_ob_memory
ruff check data/plugins/astrbot_plugin_ob_memory
```

当前测试覆盖：**297 个用例**，包括：
- `test_models.py` — clamp 行为 + bucket id 唯一性
- `test_storage.py` — schema 迁移 + 外键级联
- `test_serialization.py` — bucket 与 SQL row 双向转换
- `test_memory_manager.py` — CRUD + session 隔离 + touch + 时间涟漪
- `test_decay_engine.py` — 衰减公式各分支 + 短/长期边界连续性 + 超久远记忆不再 overflow + 后台循环
- `test_search_service.py` — 关键词权重 + 向量 fallback + resolved 降权
- `test_surface_strategy.py` — pinned 优先 + 冷启动 + token 预算
- `test_memory_writer.py` — hold / hold_feel + 合并阈值确定性
- `test_embedding_service.py` — pack/unpack + cosine + session 隔离
- `test_tagger.py` — analyze / merge / judge 容错路径
- `test_llm_tools.py` — record_memory / record_feel / recall_memory / forget_memory 端到端
- `test_llm_tools_extra.py` — record_diary / reflect_memory + recall_memory 增强模式
- `test_llm_hooks.py` — 注入 + auto-record 完整流程
- `test_commands.py` — `/memory` 子指令含二段确认
- `test_session_resolver.py` — 三种模式 + fallback
- `test_scope_mode_integration.py` — scope 切换不污染数据
- `test_cost_control_toggles.py` — 成本控制 toggle 真的跳过 LLM 调用
- `test_dashboard_smoke.py` — 仪表盘鉴权流程 + API 401 行为 + 路由注册 + 脉搏页关键入口静态 smoke

## 项目结构

```
astrbot_plugin_ob_memory/
├── _conf_schema.json              # AstrBot UI 配置（11 项常用 + advanced_mode 解锁 14 项高级）
├── metadata.yaml                  # 插件元数据
├── main.py                        # MemoryPlugin 主类，注册 hooks/tools/commands/dashboard
├── core/                          # 核心引擎
│   ├── models.py
│   ├── memory_manager.py
│   ├── memory_writer.py
│   ├── decay_engine.py
│   ├── dehydrator.py
│   ├── embedding_service.py
│   ├── tagger.py
│   ├── search_service.py
│   ├── surface_strategy.py
│   ├── session_resolver.py
│   └── prompts.py
├── handlers/                      # AstrBot 接入层
│   ├── llm_hooks.py
│   ├── llm_tools.py
│   └── commands.py
├── dashboard/                     # 嵌入式可视化管理面板
│   ├── server.py
│   ├── auth.py
│   └── static/
├── storage/                       # SQLite 包装
│   ├── db.py
│   └── schema.py
└── tests/                         # 297 个测试用例
```

---

## 已知限制 / 路线图

- **scope_mode 数据迁移**：切换模式后已有桶不会自动按新规则重新分组。如果有需要可以加 `/memory migrate-scope` 指令
- **Russell 坐标外的情感模型**：目前只支持 valence + arousal 二维。未来可能加上 dominance 形成 PAD 三维
- **记忆重构（valence ±0.1 偏移）尚未在注入层应用**：`SearchService` 已支持 `query_valence` 参数，但 `on_llm_request` 钩子目前不传当前情绪上下文。需要时可基于会话最近几轮的情感分布做估计

---

## 致谢

- [Ombre Brain](https://github.com/P0luz/Ombre-Brain) by P0luz —— 衰减公式、Russell 坐标、感受/事件双层这些核心机制的灵感来源
- [LivingMemory](https://github.com/lxfight/astrbot_plugin_livingmemory) by lxfight —— 成熟的 AstrBot 记忆插件参考实现，配置文件结构与 WebUI 接入方式参考了它的做法

## 许可证

AGPL-3.0
