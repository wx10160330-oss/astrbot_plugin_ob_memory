# OB Memory 提示词快速参考

## 📂 文件位置

```
data/plugins/astrbot_plugin_ob_memory/
├── core/
│   └── prompts.py              ← 核心提示词（4个）
├── handlers/
│   ├── llm_hooks.py            ← 记忆注入格式
│   └── llm_tools.py            ← 工具实现逻辑
├── main.py                     ← 工具描述（docstring）
└── _conf_schema.json           ← 配置项（digest_prompt）
```

## 🎯 核心提示词（prompts.py）

### 1. ANALYZE_PROMPT
- **用途**: 分析记忆 → 提取主题/情感/标签/重要度
- **触发**: `record_memory` / 自动记录
- **输出**: JSON (domain, valence, arousal, tags, name, importance)
- **可配置**: ❌ 需改代码

### 2. MERGE_PROMPT
- **用途**: 合并两条相似记忆
- **触发**: 相似度 > `merge_threshold` (默认 0.85)
- **输出**: 纯文本（合并后内容）
- **可配置**: ❌ 需改代码

### 3. JUDGE_PROMPT
- **用途**: 判断对话是否值得记录
- **触发**: 自动记录 + `auto_record_use_judge: true`
- **输出**: JSON (remember: bool, reason: string)
- **可配置**: ❌ 需改代码

### 4. DIGEST_PROMPT
- **用途**: 拆分长文本为多条记忆
- **触发**: `record_diary` / `/memory summarize` / 导入历史
- **输出**: JSON 数组
- **可配置**: ✅ 配置文件 `digest_prompt` 字段

## 🔧 快速自定义

### 方法 1: 配置文件（仅 DIGEST_PROMPT）

```json
{
  "advanced_mode": true,
  "digest_prompt": "你的自定义提示词..."
}
```

### 方法 2: 修改源码

```python
# core/prompts.py
ANALYZE_PROMPT: str = """\
你的自定义提示词...
"""
```

### 方法 3: 修改工具描述

```python
# main.py
@filter.llm_tool(name="record_memory")
async def record_memory(...) -> str:
    """你的自定义工具描述..."""
```

### 方法 4: 修改注入格式

```python
# handlers/llm_hooks.py
MEMORY_BLOCK_HEADER: str = "你的自定义标题"
MEMORY_BLOCK_FOOTER: str = "你的自定义结尾"
```

## 📊 LLM 调用次数

| 操作 | 调用次数 | 涉及提示词 |
|------|---------|-----------|
| `record_memory` (tagging开) | 1-2次 | ANALYZE + MERGE? |
| `record_memory` (tagging关) | 0次 | - |
| 自动记录 (judge开) | 2-3次 | JUDGE + DIGEST + ANALYZE |
| 自动记录 (judge关) | 1-2次 | DIGEST + ANALYZE |
| `record_diary` | 1+N次 | DIGEST + N×ANALYZE |
| 记忆注入 | 0次 | - (只读数据库) |

## 🎨 常见自定义场景

### 减少"复读"感

**方法 1**: 修改工具描述
```python
"""记住一件事。调用后直接继续对话，不要说"我已经记住了"。"""
```

**方法 2**: 系统提示
```
当你调用工具后，不需要重复说明，直接继续对话。
```

### 改变记忆风格

**修改 DIGEST_PROMPT**:
```
从 AI 的第一人称视角写记忆，使用"我"指代 AI，"你"指代用户。
例如："你告诉我你拿到了offer，我能感受到你的激动"
```

### 调整分析粒度

**修改 ANALYZE_PROMPT**:
```
- "tags": 提取 3-5 个最核心的关键词（不要太多）
- "importance": 严格评分，只有真正重要的事才给 8 分以上
```

### 更严格的自动记录

**修改 JUDGE_PROMPT**:
```
只有以下情况才记录：
- 用户分享了具体的个人信息（工作、家庭、健康）
- 用户表达了强烈的情感（愤怒、悲伤、喜悦）
- 用户做出了明确的决定或承诺

不记录：
- 闲聊、玩笑、天气查询
- 已经记录过的重复内容
```

## 🔍 调试技巧

### 查看注入的记忆
```python
# 在 llm_hooks.py 的 _inject_memories 末尾添加：
logger.info(f"[memory] injected block:\n{block}")
```

### 查看分析结果
```python
# 在 tagger.py 的 analyze 方法中添加：
logger.info(f"[memory] analyze result: {result}")
```

### 测试提示词
```bash
# 测试 DIGEST_PROMPT
/memory summarize

# 测试 ANALYZE_PROMPT
record_memory(content="测试内容")

# 测试 JUDGE_PROMPT
# 发送一条消息，查看是否自动记录
```

## ⚙️ 相关配置

```json
{
  "tagging_enabled": true,           // 是否调用 ANALYZE_PROMPT
  "auto_record_enabled": true,       // 是否启用自动记录
  "auto_record_use_judge": true,     // 是否调用 JUDGE_PROMPT
  "merge_threshold": 0.85,           // 合并阈值（影响 MERGE_PROMPT）
  "analyze_provider_id": "",         // 分析模型（影响所有提示词）
  "digest_prompt": ""                // 自定义 DIGEST_PROMPT
}
```

## 📝 提示词模板

### ANALYZE_PROMPT 模板
```
你是记忆分析专家。分析以下记忆并输出 JSON。

输出格式：
{
  "domain": ["主题1", "主题2"],
  "valence": 0.0-1.0,
  "arousal": 0.0-1.0,
  "tags": ["关键词"],
  "suggested_name": "标题",
  "importance": 1-10
}

规则：
- 只输出 JSON
- valence: 0=负面, 0.5=中性, 1=正面
- arousal: 0=平静, 1=激动
- importance: 10=核心准则, 1=闲聊
```

### DIGEST_PROMPT 模板
```
将对话拆分为独立记忆。从 AI 视角写，用"我"指代 AI，"你"指代用户。

输出 JSON 数组：
[
  {
    "name": "标题",
    "content": "内容",
    "domain": ["主题"],
    "valence": 0.0-1.0,
    "arousal": 0.0-1.0,
    "tags": ["关键词"],
    "importance": 1-10
  }
]

规则：
- 只输出 JSON 数组
- 每条内容 ≤ 300 字
- 保留具体细节
```

## 🚀 快速开始

1. **只想改日记拆分**: 配置 `digest_prompt`
2. **想改所有提示词**: 编辑 `core/prompts.py`
3. **想改工具说明**: 编辑 `main.py` docstring
4. **想改注入格式**: 编辑 `handlers/llm_hooks.py`

修改后记得：
- 配置文件修改 → 立即生效
- 代码修改 → 重启 AstrBot
