# OB Memory 插件提示词指南

## 📍 提示词位置总览

### 1. 核心提示词文件
**位置**: `data/plugins/astrbot_plugin_ob_memory/core/prompts.py`

这个文件包含所有 LLM 调用的提示词模板：

#### `ANALYZE_PROMPT` - 记忆分析提示词
**用途**: 分析单条记忆，提取元数据（主题、情感、标签、重要度等）

**何时调用**: 
- 调用 `record_memory` 工具时
- 自动记录触发时
- 需要为记忆打标签时

**输出格式**: JSON
```json
{
  "domain": ["主题1", "主题2"],
  "valence": 0.0-1.0,  // 情感效价
  "arousal": 0.0-1.0,  // 唤醒度
  "tags": ["标签1", "标签2", ...],
  "suggested_name": "记忆名称",
  "importance": 1-10
}
```

#### `MERGE_PROMPT` - 记忆合并提示词
**用途**: 将两条相似的记忆合并为一条

**何时调用**: 
- 新记忆与已有记忆相似度 > `merge_threshold`（默认 0.85）时
- 智能合并功能开启时（`tagging_enabled: true`）

**输出格式**: 纯文本（合并后的记忆内容）

#### `JUDGE_PROMPT` - 判断是否值得记录
**用途**: 判断一轮对话是否值得自动记录

**何时调用**: 
- `auto_record_enabled: true` 且 `auto_record_use_judge: true` 时
- 模型没有主动调用 `record_memory` 工具
- 用户消息通过启发式检查（长度、模式匹配）

**输出格式**: JSON
```json
{
  "remember": true/false,
  "reason": "原因说明"
}
```

#### `DIGEST_PROMPT` - 日记拆分提示词
**用途**: 将长文本拆分为多条独立记忆

**何时调用**: 
- 调用 `record_diary` 工具时
- 使用 `/memory summarize` 命令时
- 使用 `/memory import_astrbot` 导入历史时

**输出格式**: JSON 数组
```json
[
  {
    "name": "记忆名称",
    "content": "记忆内容",
    "domain": ["主题"],
    "valence": 0.0-1.0,
    "arousal": 0.0-1.0,
    "tags": ["标签"],
    "importance": 1-10
  }
]
```

**⚠️ 可自定义**: 这是唯一可以通过配置文件自定义的提示词！
- 在插件配置中找到 `digest_prompt` 字段
- 需要开启 `advanced_mode: true` 才能看到
- 留空使用默认提示词

---

### 2. 记忆注入提示词
**位置**: `data/plugins/astrbot_plugin_ob_memory/handlers/llm_hooks.py`

#### 记忆块标题
```python
MEMORY_BLOCK_HEADER: str = "=== 长期记忆 ==="
MEMORY_BLOCK_FOOTER: str = "=== 记忆结束 ==="
```

**用途**: 包裹注入到 `system_prompt` 中的记忆内容

**何时注入**: 每次 LLM 请求前（`on_llm_request` 钩子）

**注入格式示例**:
```
=== 长期记忆 ===
【最近浮现】
- 📌[求职] 面试准备: 你下周要去字节面试，准备了算法题 (id:abc123)
- [内心] 焦虑情绪: 你最近对未来感到迷茫 (id:def456)

【相关回忆】
- [语义关联][成长] 学习计划: 你决定每天学习2小时Python (id:ghi789)
=== 记忆结束 ===
```

**自定义方法**: 直接修改 `llm_hooks.py` 中的常量

---

### 3. 工具函数的 Docstring
**位置**: `data/plugins/astrbot_plugin_ob_memory/main.py`

每个 `@filter.llm_tool` 装饰的函数的 docstring 会被转换为工具描述，发送给 LLM。

#### `record_memory` 工具描述
```python
"""记住一件事。AstrBot 会把这条记忆和当前会话绑定，并在以后相关对话里再带出来。

Args:
    content(string): 要记住的内容，越具体越好。包含人物、时间、感受、待办等具体信息。
    importance(number): 1-10 的整数，1 表示水话别记、10 表示核心准则永不忘。默认 5。
    tags(string): 逗号分隔的关键词，方便日后检索；可以留空让系统自动生成。
    pinned(boolean): 是否钉选为永久核心准则；钉选后永不衰减。默认 false。
"""
```

**自定义方法**: 直接修改 `main.py` 中对应函数的 docstring

---

## 🎨 如何自定义提示词

### 方法 1: 通过配置文件（推荐）

**仅适用于 `DIGEST_PROMPT`**

1. 打开插件配置界面
2. 开启 `advanced_mode: true`
3. 找到 `digest_prompt` 字段
4. 输入你的自定义提示词
5. 保存配置（立即生效，无需重启）

**示例自定义**:
```
你是一个记忆整理助手。将用户的长文本拆分为多条独立记忆。

输出 JSON 数组，每个元素包含：
- "name": 简短标题（≤10字）
- "content": 记忆内容（≤200字）
- "domain": 主题标签列表
- "valence": 情感效价 0.0-1.0
- "arousal": 唤醒度 0.0-1.0
- "tags": 关键词列表
- "importance": 重要度 1-10

规则：
- 只输出 JSON，不要其他内容
- 从 AI 的第一人称视角写记忆
- 保留具体细节（人名、时间、地点）
```

### 方法 2: 修改源代码

**适用于所有提示词**

1. 编辑 `core/prompts.py` 文件
2. 修改对应的提示词常量
3. 重启 AstrBot 使更改生效

**示例**:
```python
ANALYZE_PROMPT: str = """\
你是记忆分析专家。分析以下记忆并输出 JSON。

输出格式：
{
  "domain": ["主题"],
  "valence": 0.0-1.0,
  "arousal": 0.0-1.0,
  "tags": ["关键词"],
  "suggested_name": "标题",
  "importance": 1-10
}

[你的自定义规则...]
"""
```

### 方法 3: 修改工具描述

**适用于工具函数的说明**

1. 编辑 `main.py` 文件
2. 找到对应的 `@filter.llm_tool` 函数
3. 修改函数的 docstring
4. 重启 AstrBot

**示例**:
```python
@filter.llm_tool(name="record_memory")
async def record_memory(
    self,
    event: AstrMessageEvent,
    content: str,
    importance: int = 5,
    tags: str = "",
    pinned: bool = False,
) -> str:
    """记录一段重要的记忆。
    
    使用场景：
    - 用户分享了个人信息
    - 用户表达了强烈情感
    - 用户做出了重要决定
    
    Args:
        content(string): 记忆内容，要具体详细
        importance(number): 重要度1-10，默认5
        tags(string): 标签，逗号分隔
        pinned(boolean): 是否永久保留
    """
```

### 方法 4: 修改记忆注入格式

**适用于注入到 system_prompt 的格式**

1. 编辑 `handlers/llm_hooks.py`
2. 修改 `MEMORY_BLOCK_HEADER` 和 `MEMORY_BLOCK_FOOTER`
3. 修改 `_format_hit_for_injection` 和 `_format_surfaced_for_injection` 函数
4. 重启 AstrBot

**示例**:
```python
MEMORY_BLOCK_HEADER: str = "📝 以下是相关记忆："
MEMORY_BLOCK_FOOTER: str = "--- 记忆结束 ---"

def _format_hit_for_injection(hit: SearchHit, *, snippet: str = "") -> str:
    bucket = hit.bucket
    name = bucket.name or bucket.id
    text = snippet or (bucket.content or "").strip()
    return f"• {name}: {text}"
```

---

## 🔧 配置参数说明

### 影响提示词行为的配置

#### `tagging_enabled` (默认: true)
- **作用**: 是否调用 `ANALYZE_PROMPT` 进行自动打标
- **关闭后**: 不调用 LLM 分析，使用默认值
- **节省**: 每次写入省 1 次 LLM 调用

#### `auto_record_enabled` (默认: true)
- **作用**: 是否启用自动记录兜底
- **关闭后**: 完全依赖模型主动调用 `record_memory`

#### `auto_record_use_judge` (默认: true)
- **作用**: 自动记录是否调用 `JUDGE_PROMPT` 复核
- **关闭后**: 跳过 LLM 判断，直接记录
- **节省**: 每次自动记录省 1 次 LLM 调用

#### `analyze_provider_id` (默认: 空)
- **作用**: 指定用于分析、打标、合并的模型
- **留空**: 使用当前会话的默认对话模型
- **推荐**: 使用推理能力强的模型（如 GPT-4、Claude）

#### `digest_prompt` (默认: 空)
- **作用**: 自定义 `DIGEST_PROMPT`
- **留空**: 使用内置默认提示词
- **需要**: `advanced_mode: true`

---

## 📊 提示词调用流程

### 场景 1: 用户主动记录
```
用户消息 → 模型调用 record_memory 工具
         ↓
    [tagging_enabled?]
         ↓ Yes
    调用 ANALYZE_PROMPT 分析
         ↓
    [相似度检查]
         ↓ 相似度 > merge_threshold
    调用 MERGE_PROMPT 合并
         ↓
    保存到数据库
```

### 场景 2: 自动记录
```
用户消息 + 模型回复 → [模型没调用 record_memory?]
                    ↓ Yes
              [auto_record_enabled?]
                    ↓ Yes
              [启发式检查通过?]
                    ↓ Yes
              [auto_record_use_judge?]
                    ↓ Yes
              调用 JUDGE_PROMPT 判断
                    ↓ remember: true
              调用 DIGEST_PROMPT 拆分
                    ↓
              每条记忆调用 ANALYZE_PROMPT
                    ↓
              保存到数据库
```

### 场景 3: 记忆注入
```
每次 LLM 请求前
    ↓
搜索相关记忆（关键词 + 向量）
    ↓
主动浮现未解决记忆
    ↓
格式化为记忆块
    ↓
注入到 system_prompt
    ↓
发送给 LLM
```

---

## 💡 优化建议

### 减少 LLM 调用次数
1. 关闭 `tagging_enabled` - 省分析调用
2. 关闭 `auto_record_use_judge` - 省判断调用
3. 减少 `max_search_results` 和 `max_surface_results` - 减少注入内容

### 提高记忆质量
1. 使用更强的 `analyze_provider_id`
2. 自定义 `digest_prompt` 以适应你的使用场景
3. 调整 `merge_threshold` 控制合并敏感度

### 减少"复读"感
在系统提示中添加：
```
当你调用工具后，不需要重复说明你做了什么，直接继续对话即可。
```

或者修改工具的 docstring，明确说明：
```
调用此工具后，不要再重复说"我已经记住了"，直接继续对话。
```

---

## 🔍 调试提示词

### 查看实际发送的提示词
1. 设置日志级别为 DEBUG
2. 查看日志中的 `[memory]` 标签
3. 可以看到注入的记忆块内容

### 测试提示词效果
1. 使用 `/memory summarize` 测试 `DIGEST_PROMPT`
2. 手动调用 `record_memory` 测试 `ANALYZE_PROMPT`
3. 创建相似记忆测试 `MERGE_PROMPT`

### 常见问题
- **记忆分析不准确**: 调整 `ANALYZE_PROMPT` 或更换 `analyze_provider_id`
- **合并过于激进**: 提高 `merge_threshold`（如 0.90）
- **自动记录太多**: 提高 `auto_record_min_chars` 或添加 `auto_record_skip_patterns`

---

## 📝 总结

| 提示词 | 位置 | 可配置 | 何时调用 |
|--------|------|--------|----------|
| `ANALYZE_PROMPT` | `core/prompts.py` | ❌ 需改代码 | 记录时打标 |
| `MERGE_PROMPT` | `core/prompts.py` | ❌ 需改代码 | 相似记忆合并 |
| `JUDGE_PROMPT` | `core/prompts.py` | ❌ 需改代码 | 自动记录判断 |
| `DIGEST_PROMPT` | `core/prompts.py` | ✅ 配置文件 | 日记拆分 |
| 记忆注入格式 | `handlers/llm_hooks.py` | ❌ 需改代码 | 每次请求前 |
| 工具描述 | `main.py` | ❌ 需改代码 | 工具调用时 |

**快速自定义路径**:
1. 只想改日记拆分 → 配置 `digest_prompt`
2. 想改所有提示词 → 编辑 `core/prompts.py`
3. 想改工具说明 → 编辑 `main.py` 的 docstring
4. 想改注入格式 → 编辑 `handlers/llm_hooks.py`
