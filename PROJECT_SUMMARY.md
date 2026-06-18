# AI 知识库问答助手 — 项目总结文档

## 一、项目概述

AI 知识库问答助手是一个基于 RAG（检索增强生成）架构的文档问答系统。用户上传文档后，系统将其分块、向量化并存储到本地数据库，随后用户可以基于文档内容进行自然语言提问，系统检索相关片段并调用大语言模型生成回答。

### 核心特性

- 支持 5 种文档格式：`.txt` / `.pdf` / `.docx` / `.md` / `.csv`
- 扫描 PDF 自动 OCR（本地 RapidOCR 或 Vision API）
- 语义检索：OpenAI `text-embedding-3-small` (1536 维)
- 流式回答（SSE）+ 来源标注
- 对话历史自动持久化
- FTS5 全文搜索
- Docker 一键部署

---

## 二、技术架构

### 技术栈

| 层级 | 技术 | 版本约束 |
|------|------|---------|
| Web 框架 | FastAPI | >=0.110.0, <1.0.0 |
| ASGI 服务器 | Uvicorn | >=0.29.0, <1.0.0 |
| 向量数据库 | SQLite (自定义) | 内置 |
| 嵌入模型 | OpenAI `text-embedding-3-small` | 通过 openai SDK >=1.30.0 |
| LLM 集成 | OpenAI 兼容 API | 同上 |
| PDF 解析 | PyMuPDF | >=1.24.0, <2.0.0 |
| OCR | RapidOCR ONNX | >=1.3.0, <2.0.0 |
| DOCX 解析 | python-docx | >=1.1.0, <2.0.0 |
| 请求验证 | Pydantic v2 | 内置 |
| 速率限制 | slowapi | >=0.1.9, <1.0.0 |
| 前端 | 原生 JS SPA | 单文件 |

### 项目结构

```
AI 知识库问答助手/
├── app.py                  # FastAPI 入口，路由 + 中间件
├── rag_engine.py           # RAG 核心：嵌入、检索、切块、LLM 调用
├── schemas.py              # Pydantic 请求体验证模型
├── requirements.txt        # Python 依赖（版本锁定）
├── Dockerfile              # Docker 构建文件
├── docker-compose.yml      # Docker Compose 编排
├── AGENTS.md               # 开发上下文
├── README.md               # 使用说明
├── static/
│   └── index.html          # 前端 SPA（HTML + CSS + JS 单文件）
├── tests/
│   ├── test_api.py         # API 端点测试 (8)
│   ├── test_chunking.py    # 切块算法测试 (8)
│   ├── test_embed.py       # 嵌入函数测试 (11)
│   ├── test_search.py      # 向量检索测试 (4)
│   ├── test_conversation.py # 对话持久化测试 (5)
│   ├── test_stream.py      # 流式端点测试 (5)
│   ├── test_errors.py      # 错误场景测试 (10)
│   └── test_integration.py # 集成测试 (10)
└── vector_store.db         # SQLite 运行时数据（gitignore）
```

### 数据流

```
用户上传文件
    │
    ▼
POST /api/upload (FastAPI)
    │
    ▼
后台线程处理
    ├── load_document()      → 文本提取 (txt/pdf/docx/md/csv)
    ├── OCR (如需)           → RapidOCR 或 Vision API
    ├── split_text()         → 段落/句子边界智能切块
    ├── create_collection()  → OpenAI Embeddings API 批量嵌入
    └── SQLite INSERT        → 向量存储 + FTS5 索引同步
    │
    ▼
用户提问
    │
    ▼
POST /api/ask/stream (SSE)
    ├── query_collection()   → 余弦相似度检索 Top-K 片段
    ├── _build_messages()    → 构建 System Prompt (含检索上下文)
    └── OpenAI Chat API      → 流式生成回答
    │
    ▼
前端逐 token 渲染 + 来源标注
```

### 数据库设计

**SQLite WAL 模式**，单文件 `vector_store.db`：

```sql
-- 文档向量表
CREATE TABLE documents (
    doc_id TEXT,
    filename TEXT,
    chunk_index INTEGER,
    content TEXT,
    embedding TEXT,           -- JSON 序列化的 1536 维浮点数组
    PRIMARY KEY (doc_id, chunk_index)
);

-- FTS5 全文搜索索引（自动同步）
CREATE VIRTUAL TABLE docs_fts USING fts5(
    content, filename, doc_id UNINDEXED, chunk_index UNINDEXED,
    content='documents', content_rowid='rowid'
);

-- 对话历史表
CREATE TABLE conversations (
    session_id TEXT,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    sources TEXT DEFAULT '[]',
    created_at REAL NOT NULL
);
CREATE INDEX idx_conversations_session ON conversations(session_id, created_at);
```

### 安全设计

| 层级 | 措施 |
|------|------|
| 传输 | POST body 传输（非 URL 查询参数）；HTTPS 反向代理 |
| 输入验证 | Pydantic v2 自动校验 + MIME 魔数检测 + 文件名净化 |
| XSS 防护 | CSP 头 + `escHtml()` 转义 + 文件名 HTML 字符过滤 |
| CSRF | CORS 限制到 localhost + Origin 校验 |
| 速率限制 | slowapi: 问答 30/min, 上传 10/min |
| 信息泄露 | 异常不返回客户端 + API key 不返回前端 |
| Prompt 注入 | System Prompt 硬化 + 分隔符 + 长度限制 |
| 看门狗 | 上传处理 10 分钟超时自动标记失败 |

---

## 三、API 接口列表

### 系统

| 方法 | 路径 | 说明 | 速率限制 |
|------|------|------|---------|
| `GET` | `/` | 前端 SPA 页面 | — |
| `GET` | `/api/health` | 健康检查 | — |

**健康检查响应：**
```json
{"status": "ok", "documents": 5, "sessions": 3, "version": "1.0"}
```

### 模型配置

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/config` | 保存 LLM 配置 |
| `GET` | `/api/config` | 获取当前配置状态 |

**POST /api/config 请求体：**
```json
{
  "api_key": "sk-xxx",          // 必填, 1-4096 字符
  "base_url": "https://api.example.com/v1",  // 必填, 1-4096 字符
  "model": "gpt-4o"             // 必填, 1-256 字符
}
```

**GET /api/config 响应：**
```json
{"configured": true, "base_url": "https://api.example.com/v1", "model": "gpt-4o"}
```

### 文档管理

| 方法 | 路径 | 说明 | 速率限制 |
|------|------|------|---------|
| `POST` | `/api/upload` | 上传文档（multipart） | 10/min |
| `GET` | `/api/upload/status/{task_id}` | 轮询上传进度 | — |
| `GET` | `/api/documents` | 列出所有文档 | — |
| `GET` | `/api/documents/search?q=<query>&limit=<n>` | FTS5 全文搜索 | — |
| `GET` | `/api/documents/{doc_id}/content` | 预览文档内容 | — |
| `DELETE` | `/api/documents/{doc_id}` | 删除单个文档 | — |
| `DELETE` | `/api/data` | 清空所有数据（文档+对话） | — |

**上传进度响应：**
```json
// 处理中
{"status": "processing", "task_id": "a1b2c3d4e5f6", "filename": "doc.pdf", "progress": "ocr: 5/20"}

// 完成
{"status": "done", "filename": "doc.pdf", "doc_id": "x1y2z3", "chunks": 42}

// 失败
{"status": "error", "filename": "doc.pdf", "error": "Processing timed out (10 min limit)"}
```

**全文搜索响应：**
```json
{
  "query": "machine learning",
  "results": [
    {
      "doc_id": "abc123",
      "filename": "ml-intro.pdf",
      "chunk_index": 3,
      "content": "Machine learning is a subset of...",
      "rank": -2.5
    }
  ]
}
```

### 问答

| 方法 | 路径 | 说明 | 速率限制 |
|------|------|------|---------|
| `POST` | `/api/ask` | 非流式问答（返回 JSON） | 30/min |
| `POST` | `/api/ask/stream` | 流式问答（SSE） | 30/min |
| `GET` | `/api/ask/stream` | 返回 405（强制 POST） | — |

**POST /api/ask 请求体：**
```json
{
  "question": "什么是机器学习？",   // 必填, 1-10000 字符
  "chat_history": [                 // 可选, 历史对话
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！"}
  ]
}
```

**POST /api/ask 响应：**
```json
{
  "answer": "机器学习是人工智能的一个分支...",
  "history": [...],
  "sources": [
    {"filename": "ml-intro.pdf", "content": "Machine learning is..."}
  ]
}
```

**POST /api/ask/stream SSE 事件格式：**
```
data: {"token": "机器"}
data: {"token": "学习"}
data: {"token": "是..."}
data: {"meta": {"history": [...], "sources": [...]}}
```

### 对话管理

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/api/sessions` | 列出所有会话 |
| `GET` | `/api/conversation?session_id=<id>` | 加载对话历史 |
| `POST` | `/api/conversation` | 保存对话消息 |
| `DELETE` | `/api/sessions/{session_id}` | 删除会话 |

---

## 四、部署指南

### 方式一：本地开发

```bash
# 1. 克隆项目
cd "AI 知识库问答助手"

# 2. 创建虚拟环境
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate    # Linux/Mac

# 3. 安装依赖
pip install -r requirements.txt

# 4. 启动服务
python -m uvicorn app:app --reload --host 127.0.0.1 --port 8000

# 5. 访问
# 浏览器打开 http://localhost:8000
```

### 方式二：Docker 部署

```bash
# 1. 构建镜像
docker build -t ai-kb .

# 2. 运行容器
docker run -d \
  -p 8000:8000 \
  -v ./data:/app/data \
  --name ai-kb \
  ai-kb

# 3. 或使用 docker-compose
docker-compose up -d
```

**环境变量预配置（可选）：**

```bash
# .env 文件
MIMO_API_KEY=sk-xxx
MIMO_BASE_URL=https://api.example.com/v1
MIMO_MODEL=gpt-4o

# 启动时自动加载配置，无需前端手动输入
docker-compose up -d
```

### 方式三：systemd 服务（Linux 生产环境）

```ini
# /etc/systemd/system/ai-kb.service
[Unit]
Description=AI Knowledge Base Q&A
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/ai-kb
Environment=MIMO_API_KEY=sk-xxx
Environment=MIMO_BASE_URL=https://api.example.com/v1
Environment=MIMO_MODEL=gpt-4o
ExecStart=/opt/ai-kb/venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable ai-kb
sudo systemctl start ai-kb
```

### 反向代理（Nginx）

```nginx
server {
    listen 443 ssl;
    server_name kb.example.com;

    ssl_certificate /etc/letsencrypt/live/kb.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/kb.example.com/privkey.pem;

    client_max_body_size 50M;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # SSE 流式支持
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 300s;
    }
}
```

---

## 五、运行测试

```bash
# 运行全部 59 个测试
python -m pytest tests/ -v

# 运行特定测试文件
python -m pytest tests/test_api.py -v

# 测试覆盖率
pip install pytest-cov
python -m pytest tests/ --cov=app --cov=rag_engine --cov-report=term-missing
```

**测试分布：**

| 文件 | 用例数 | 覆盖范围 |
|------|--------|---------|
| test_api.py | 8 | 配置 CRUD、上传、文档列表 |
| test_chunking.py | 8 | 段落/句子边界切块算法 |
| test_embed.py | 11 | 嵌入维度、确定性、余弦相似度 |
| test_search.py | 4 | 向量存储、检索、删除 |
| test_conversation.py | 5 | 对话持久化 CRUD |
| test_stream.py | 5 | SSE 流式端点 |
| test_errors.py | 10 | 错误场景边界 |
| test_integration.py | 10 | 健康检查、搜索、会话、request_id、embed_model |
| **总计** | **59** | |

---

## 六、配置说明

### LLM API 配置

支持任何 OpenAI 兼容 API：

| 提供商 | base_url | model |
|--------|----------|-------|
| OpenAI | `https://api.openai.com/v1` | `gpt-4o` |
| 小米 MiMo | `https://token-plan-cn.xiaomimimo.com/v1` | `mimo-v2.5-pro` |
| DeepSeek | `https://api.deepseek.com/v1` | `deepseek-chat` |
| 本地 Ollama | `http://localhost:11434/v1` | `llama3` |

### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MAX_FILE_SIZE` | 50MB | 单文件上传上限 |
| `CHUNK_SIZE` | 500 字符 | 默认切块大小（大文档自动增大） |
| `CHUNK_OVERLAP` | 50 字符 | 切块重叠 |
| `EMBED_DIM` | 1536 | 嵌入向量维度 |
| `CONVERSATION_TTL` | 30 天 | 对话历史保留时间 |
| `UPLOAD_TASK_TTL` | 1 小时 | 上传任务状态保留时间 |
| 上传看门狗 | 10 分钟 | 超时自动标记失败 |
| 前端轮询超时 | 10 分钟 | 超时显示警告 |

---

## 七、已知限制

1. **向量检索为暴力扫描** — 适合中小规模（<10K chunks），大规模需引入 ANN 索引
2. **单用户设计** — 无多用户认证，API key 内存存储不持久化
3. **嵌入 API 依赖** — 无 API key 时回退为 hash 嵌入（质量下降）
4. **无 GPU 加速** — OCR 和嵌入均为 CPU 计算
5. **SQLite 并发写入** — WAL 模式下写入串行化，高并发场景可能成为瓶颈
