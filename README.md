# AI Knowledge Base Q&A Assistant

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
