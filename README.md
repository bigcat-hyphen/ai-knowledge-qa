# AI Knowledge Base Q&A Assistant / AI 知识库问答助手

[English](#english) | [中文](#中文)

---

<a name="english"></a>

## English

RAG-based document Q&A assistant. Upload documents, ask questions, get AI-powered answers with source citations.

## Tech Stack

- **Backend**: FastAPI + SQLite (custom vector store)
- **Embeddings**: OpenAI `text-embedding-3-small` (1536-dim)
- **LLM**: Any OpenAI-compatible API (user-configurable)
- **PDF**: pymupdf + OCR fallback

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## Run

```bash
python -m uvicorn app:app --reload
```

Open http://localhost:8000

## Usage

1. Configure your LLM API (base URL, API key, model name) in the settings panel
2. Upload .txt, .pdf, .docx, .md, .csv documents (supports batch upload, drag & drop)
3. Ask questions — answers are grounded in your documents
4. Chat history is automatically saved and persists across sessions

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Frontend UI |
| POST | `/api/config` | Save model config |
| GET | `/api/config` | Get current config status |
| POST | `/api/upload` | Upload document (multipart) |
| GET | `/api/upload/status/{task_id}` | Poll upload progress |
| GET | `/api/documents` | List documents |
| GET | `/api/documents/search?q=...&limit=...` | FTS5 full-text search |
| GET | `/api/documents/{id}/content` | Get document content preview |
| DELETE | `/api/documents/{id}` | Delete document |
| DELETE | `/api/data` | Clear all data |
| POST | `/api/ask` | Ask question |
| POST | `/api/ask/stream` | SSE streaming answer |
| GET | `/api/sessions` | List conversation sessions |
| GET | `/api/conversation?session_id=...` | Load conversation |
| DELETE | `/api/sessions/{id}` | Delete session |

## Privacy

- All documents and vectors stored locally in `vector_store.db`
- API key stored in memory only, not persisted to disk
- Questions and retrieved context are sent to the configured LLM API
- No telemetry or tracking

## Tests

```bash
python -m pytest tests/ -v
```

---

<a name="中文"></a>

## 中文

基于 RAG 的文档问答助手。上传文档，提出问题，获取带来源引用的 AI 回答。

## 技术栈

- **后端**: FastAPI + SQLite（自定义向量存储）
- **嵌入模型**: OpenAI `text-embedding-3-small`（1536 维）
- **大语言模型**: 任意 OpenAI 兼容 API（用户可配置）
- **PDF 解析**: pymupdf + OCR 兜底

## 环境搭建

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## 启动运行

```bash
python -m uvicorn app:app --reload
```

打开 http://localhost:8000

## 使用说明

1. 在设置面板中配置你的大模型 API（Base URL、API Key、模型名称）
2. 上传文档（支持 .txt、.pdf、.docx、.md、.csv，支持批量上传和拖拽）
3. 提问 — 回答基于你上传的文档内容
4. 对话历史自动保存，跨会话持久化

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 前端页面 |
| POST | `/api/config` | 保存模型配置 |
| GET | `/api/config` | 获取当前配置状态 |
| POST | `/api/upload` | 上传文档（multipart） |
| GET | `/api/upload/status/{task_id}` | 查询上传进度 |
| GET | `/api/documents` | 文档列表 |
| GET | `/api/documents/search?q=...&limit=...` | FTS5 全文搜索 |
| GET | `/api/documents/{id}/content` | 获取文档内容预览 |
| DELETE | `/api/documents/{id}` | 删除文档 |
| DELETE | `/api/data` | 清空所有数据 |
| POST | `/api/ask` | 提问 |
| POST | `/api/ask/stream` | SSE 流式回答 |
| GET | `/api/sessions` | 会话列表 |
| GET | `/api/conversation?session_id=...` | 加载对话记录 |
| DELETE | `/api/sessions/{id}` | 删除会话 |

## 隐私说明

- 所有文档和向量数据本地存储在 `vector_store.db`
- API Key 仅保存在内存中，不写入磁盘
- 提问和检索的上下文会发送到配置的大模型 API
- 无遥测、无追踪

## 运行测试

```bash
python -m pytest tests/ -v
```
