# Dashboard API 文档

本文档描述记忆插件内嵌 HTTP 服务的全部 API 端点，供前端开发者或 LLM 参考。

**Base URL**: `http://{dashboard_host}:{dashboard_port}` (默认 `http://localhost:2140`)

---

## 认证机制

除 `/health`、`/auth/*`、`/dashboard`、`/static/*` 外，所有 `/api/*` 端点需要有效的 session cookie。

- Cookie 名称: `memory_dashboard_session`
- 有效期: 7 天
- 属性: `httpOnly`, `SameSite=Lax`
- 未认证请求返回: `401 {"detail": "auth required"}`

---

## 认证端点

### GET /auth/status

返回当前认证状态。

**响应**:
```json
{
  "setup_needed": false,
  "env_locked": false,
  "authenticated": true
}
```

| 字段 | 说明 |
|------|------|
| `setup_needed` | `true` 表示尚未设置密码，需要先调用 `/auth/setup` |
| `env_locked` | `true` 表示密码由环境变量 `MEMORY_DASHBOARD_PASSWORD` 控制，不可在 UI 修改 |
| `authenticated` | 当前请求是否已认证 |

---

### POST /auth/setup

首次设置密码（仅在 `setup_needed=true` 时可用）。

**请求体**:
```json
{ "password": "your_password" }
```

**成功响应**: `200 {"ok": true}` + Set-Cookie  
**失败响应**: `400 {"error": "密码至少4位"}` 或 `400 {"error": "已配置"}`

---

### POST /auth/login

登录。

**请求体**:
```json
{ "password": "your_password" }
```

**成功响应**: `200 {"ok": true}` + Set-Cookie  
**失败响应**: `401 {"error": "密码错误"}`

---

### POST /auth/logout

登出，清除 session。

**响应**: `200 {"ok": true}` + Delete-Cookie

---

### POST /auth/change-password

修改密码（需已认证，且非 env_locked）。

**请求体**:
```json
{ "current": "old_password", "new": "new_password" }
```

**成功响应**: `200 {"ok": true}` + 新 Set-Cookie  
**失败响应**: `400 {"error": "当前密码错误或新密码太短"}` 或 `400 {"error": "密码由环境变量控制，无法在此修改"}`

---

## 健康检查

### GET /health

无需认证。

**响应**:
```json
{
  "status": "ok",
  "version": "0.1.0"
}
```

---

## 数据 API

以下所有端点需要有效的 session cookie。

---

### GET /api/buckets

列出记忆桶，支持过滤。

**查询参数**:

| 参数 | 类型 | 说明 |
|------|------|------|
| `session` | string | 按 session_id 过滤（留空返回所有 session） |
| `type` | string | 按 bucket_type 过滤：`dynamic` / `permanent` / `feel` / `archived` |
| `q` | string | 全文搜索（匹配 name / content / tags） |
| `limit` | int | 最多返回条数（默认 100） |

**响应**:
```json
{
  "total": 17,
  "buckets": [
    {
      "id": "ab12cd34ef56",
      "session_id": "conv:xxx",
      "name": "实习offer获得",
      "domain": ["成长", "求职"],
      "tags": ["实习", "offer"],
      "valence": 0.8,
      "arousal": 0.7,
      "importance": 7,
      "bucket_type": "dynamic",
      "pinned": false,
      "resolved": false,
      "digested": false,
      "activation_count": 3.0,
      "score": 4.82,
      "created_at": 1747000000.0,
      "last_active_at": 1747200000.0,
      "content_preview": "你告诉我你拿到了实习offer..."
    }
  ]
}
```

桶按 `score`（Activation_Score）降序排列。`content_preview` 截取前 200 字符。

---

### GET /api/bucket/{bucket_id}

获取单个桶的完整详情。

**响应**:
```json
{
  "id": "ab12cd34ef56",
  "session_id": "conv:xxx",
  "name": "实习offer获得",
  "content": "完整内容...",
  "domain": ["成长", "求职"],
  "tags": ["实习", "offer"],
  "valence": 0.8,
  "arousal": 0.7,
  "importance": 7,
  "bucket_type": "dynamic",
  "pinned": false,
  "resolved": false,
  "digested": false,
  "model_valence": null,
  "source_bucket_id": null,
  "activation_count": 3.0,
  "score": 4.82,
  "created_at": 1747000000.0,
  "last_active_at": 1747200000.0
}
```

**404**: `{"error": "未找到"}`

---

### PATCH /api/bucket/{bucket_id}

更新桶的元数据。通过 `MemoryManager.update` 执行，自动应用 clamp 规则。

**请求体**（只传需要修改的字段）:
```json
{
  "name": "新名称",
  "content": "新内容",
  "domain": ["新主题"],
  "tags": ["tag1", "tag2"],
  "valence": 0.7,
  "arousal": 0.5,
  "importance": 8,
  "pinned": true,
  "resolved": false,
  "digested": false
}
```

**可更新字段**: `name`, `content`, `domain`, `tags`, `valence`, `arousal`, `importance`, `pinned`, `resolved`, `digested`

**业务规则**:
- `valence` / `arousal` 自动 clamp 到 [0.0, 1.0]
- `importance` 自动 clamp 到 [1, 10]
- `pinned=true` 会自动设置 `importance=10` 和 `bucket_type="permanent"`
- 修改 `content` 时会自动刷新 embedding 向量

**成功响应**: `200 {"ok": true, "id": "ab12cd34ef56"}`  
**失败响应**: `404 {"error": "未找到"}` / `400 {"error": "无更新字段"}` / `500 {"error": "更新失败"}`

---

### DELETE /api/bucket/{bucket_id}

永久删除一个桶及其 embedding。

**成功响应**: `200 {"ok": true}`  
**失败响应**: `404 {"error": "未找到"}` / `500 {"error": "删除失败"}`

---

### GET /api/stats

全局统计信息。

**响应**:
```json
{
  "sessions": 3,
  "counts": {
    "dynamic": 15,
    "permanent": 2,
    "feel": 5,
    "archived": 8
  },
  "total": 30,
  "today_new": 2,
  "week_new": 9,
  "max_activation": 4.3,
  "decay_engine": "运行中",
  "embedding": "已启用"
}
```

| 字段 | 说明 |
|------|------|
| `sessions` | 不同 session_id 的数量 |
| `counts` | 按 bucket_type 分类的桶数量 |
| `total` | 所有桶总数 |
| `today_new` | 今日新增桶数 |
| `week_new` | 近 7 天新增桶数 |
| `max_activation` | 当前全局最高 activation_count |
| `decay_engine` | `"运行中"` / `"已停止"` |
| `embedding` | `"已启用"` / `"未启用"` |

---

### GET /api/search

双通道搜索（关键词 + 向量）。

**查询参数**:

| 参数 | 类型 | 说明 |
|------|------|------|
| `q` | string | **必填**，搜索关键词 |
| `session` | string | 指定 session_id（留空使用第一个可用 session） |

**响应**:
```json
{
  "results": [
    {
      "id": "ab12cd34ef56",
      "name": "实习offer获得",
      "score": 72.5,
      "via": "both",
      "domain": ["成长", "求职"],
      "content_preview": "你告诉我你拿到了实习offer...",
      "importance": 7,
      "resolved": false,
      "pinned": false
    }
  ]
}
```

| `via` 值 | 含义 |
|-----------|------|
| `"keyword"` | 仅通过关键词匹配命中 |
| `"vector"` | 仅通过向量相似度命中 |
| `"both"` | 两个通道都命中 |

---

## 前端页面

### GET /

重定向到 `/dashboard`。

### GET /dashboard

返回单页 HTML 前端。

### GET /static/{filename}

静态资源（CSS / JS）。当前文件：
- `style.css` — 暗色调样式
- `app.js` — 前端逻辑（vanilla JS）

---

## 数据模型参考

### MemoryBucket 字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 12 位 hex，UUID4 前 12 字符 |
| `session_id` | string | 隔离键，格式取决于 scope_mode |
| `name` | string | 记忆名称（≤80 字符） |
| `content` | string | 记忆正文 |
| `domain` | string[] | 主题域标签（1-2 个） |
| `tags` | string[] | 关键词标签（5-10 个） |
| `valence` | float | Russell 效价 [0.0, 1.0]，0=极负 1=极正 |
| `arousal` | float | Russell 唤醒度 [0.0, 1.0]，0=平静 1=激动 |
| `importance` | int | 重要度 [1, 10] |
| `bucket_type` | string | `dynamic` / `permanent` / `feel` / `archived` |
| `pinned` | bool | 钉选（永不衰减，importance 锁 10） |
| `resolved` | bool | 已解决（权重 ×0.05，不主动浮现） |
| `digested` | bool | 已消化（写过 feel，权重 ×0.02） |
| `model_valence` | float? | 模型对此记忆的主观效价 |
| `source_bucket_id` | string? | feel 桶指向的源事件桶 id |
| `activation_count` | float | 被召回次数（含 time_ripple 的小数增量） |
| `score` | float | 当前 Activation_Score（由 API 计算返回） |
| `created_at` | float | 创建时间（Unix 时间戳） |
| `last_active_at` | float | 最后激活时间（Unix 时间戳） |

### bucket_type 含义

| 类型 | 说明 |
|------|------|
| `dynamic` | 普通记忆，参与衰减 |
| `permanent` | 钉选/核心准则，永不衰减 |
| `feel` | 模型感受，永不衰减，不参与普通浮现 |
| `archived` | 已归档（衰减分低于阈值），不参与搜索和浮现 |
