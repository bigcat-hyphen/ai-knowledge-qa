# AGENTS.md

## Project

AI Knowledge Base Q&A Assistant (AI 知识库问答助手). RAG-based document Q&A with OpenAI-compatible API.

## Tech Stack

- FastAPI + SQLite (custom vector store) + OpenAI-compatible LLM API
- Embeddings: OpenAI `text-embedding-3-small` (1536-dim), hash fallback if no API key
- No langchain — hand-written RAG logic in `rag_engine.py`
- PDF parsing: pymupdf, with OCR fallback (rapidocr_onnxruntime or vision API)

## Key Files

- `app.py` — FastAPI server, all API routes, security middleware, rate limiting
- `rag_engine.py` — RAG core: embedding, SQLite vector store, smart chunking, LLM calls
- `static/index.html` — SPA frontend (config, docs, chat, session management)
- `tests/` — pytest test suite (59 tests)

## Dev Commands

```bash
pip install -r requirements.txt
python -m uvicorn app:app --reload
python -m pytest tests/ -v
```

## Architecture

- Model config (API key, base_url, model) entered by user in frontend, stored in backend memory only
- Documents stored in local `vector_store.db` (SQLite, WAL mode)
- Conversations persisted in same SQLite DB
- Embeddings: OpenAI API preferred, deterministic hash fallback
- Smart chunking: paragraph/sentence boundaries
- Upload processing: background threads with 10-min watchdog timeout
- Rate limiting: slowapi (30/min ask, 10/min upload)
- Security: CSP headers, CORS restricted to localhost, MIME validation, XSS sanitization

## Status

Production-ready. 59 tests passing. CI via GitHub Actions.
