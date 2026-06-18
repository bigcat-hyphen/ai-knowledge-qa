import json
import os
import time
import uuid
import logging
import threading
from pathlib import Path

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            log_record["exception"] = self.formatException(record.exc_info)
        if hasattr(record, "request_id"):
            log_record["request_id"] = record.request_id
        return json.dumps(log_record, ensure_ascii=False)

_use_json = os.environ.get("LOG_FORMAT", "").lower() == "json"

if _use_json:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter(datefmt='%Y-%m-%dT%H:%M:%S'))
    logging.basicConfig(level=logging.INFO, handlers=[handler])
else:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s.%(msecs)03d %(name)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
logger = logging.getLogger(__name__)


class AccessLogFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        if "/api/health" in msg and "200" in msg:
            return False
        return True


uvicorn_logger = logging.getLogger("uvicorn.access")
uvicorn_logger.addFilter(AccessLogFilter())

from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

import rag_engine
from schemas import ConfigSchema, AskSchema, ConversationSchema

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_EXTENSIONS = {".txt", ".pdf", ".docx", ".md", ".csv"}
UPLOAD_TASK_TTL = 3600  # 1 hour in seconds


def _validate_mime(content: bytes, ext: str) -> bool:
    if ext == ".pdf":
        return content.startswith(b"%PDF")
    elif ext == ".docx":
        return content.startswith(b"PK\x03\x04")
    elif ext in (".txt", ".md", ".csv"):
        try:
            content.decode("utf-8")
            return True
        except (UnicodeDecodeError, UnicodeError):
            try:
                content.decode("gbk")
                return True
            except (UnicodeDecodeError, UnicodeError):
                return False
    return False


_model_config_lock = threading.Lock()
model_config: dict = {}

upload_tasks: dict[str, dict] = {}
_tasks_lock = threading.Lock()


def _get_model_config() -> dict:
    with _model_config_lock:
        return model_config.copy()


def _set_model_config(config: dict):
    with _model_config_lock:
        model_config.clear()
        model_config.update(config)


@asynccontextmanager
async def lifespan(app):
    try:
        rag_engine.cleanup_expired_conversations()
        logger.info("Cleaned up expired conversations on startup")
    except Exception:
        logger.exception("Failed to clean up expired conversations on startup")
    yield
    logger.info("Shutting down, cleaning up resources...")
    with _tasks_lock:
        for tid in list(upload_tasks.keys()):
            if upload_tasks[tid].get("status") == "processing":
                upload_tasks[tid] = {"status": "error", "filename": "", "error": "Server shutting down"}


app = FastAPI(title="AI Knowledge Base Q&A", lifespan=lifespan)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type"],
)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'"
        return response

app.add_middleware(SecurityHeadersMiddleware)


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

app.add_middleware(RequestIDMiddleware)

_env_api_key = os.environ.get("MIMO_API_KEY")
_env_base_url = os.environ.get("MIMO_BASE_URL")
_env_model = os.environ.get("MIMO_MODEL")
if _env_api_key and _env_base_url and _env_model:
    _set_model_config({
        "api_key": _env_api_key,
        "base_url": _env_base_url.rstrip("/"),
        "model": _env_model,
    })
    logger.info(f"Config pre-loaded from environment: {_env_base_url} / {_env_model}")


def _cleanup_old_tasks():
    now = time.time()
    with _tasks_lock:
        stale = [tid for tid, t in upload_tasks.items()
                 if now - t.get("created_at", 0) > UPLOAD_TASK_TTL]
        for tid in stale:
            upload_tasks.pop(tid, None)
    if stale:
        logger.info(f"Cleaned up {len(stale)} stale upload tasks")


@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.get("/api/health")
async def health():
    try:
        docs = rag_engine.list_documents()
        sessions = rag_engine.list_sessions()
        return {"status": "ok", "documents": len(docs), "sessions": len(sessions), "version": "1.0"}
    except Exception:
        return {"status": "error", "version": "1.0"}


MAX_CONFIG_FIELD_LENGTH = 4096


@app.post("/api/config")
async def save_config(config: ConfigSchema):
    cfg = {
        "api_key": config.api_key,
        "base_url": config.base_url.rstrip("/"),
        "model": config.model,
    }
    if config.embed_model:
        cfg["embed_model"] = config.embed_model
    _set_model_config(cfg)
    return {"status": "ok", "message": "Config saved"}


@app.get("/api/config")
async def get_config():
    cfg = _get_model_config()
    if not cfg:
        return {"configured": False}
    result = {
        "configured": True,
        "base_url": cfg["base_url"],
        "model": cfg["model"],
    }
    if cfg.get("embed_model"):
        result["embed_model"] = cfg["embed_model"]
    return result


@app.post("/api/upload")
@limiter.limit("10/minute")
async def upload_file(request: Request, file: UploadFile = File(...)):
    filename = rag_engine._sanitize_filename(file.filename)
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Only .txt, .pdf, .docx, .md, .csv files are allowed, got: {ext}")

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(400, f"File too large (max {MAX_FILE_SIZE // 1024 // 1024}MB)")
    if len(content) == 0:
        raise HTTPException(400, "File is empty")
    if not _validate_mime(content, ext):
        raise HTTPException(400, f"File content does not match extension: {ext}")

    _cleanup_old_tasks()
    task_id = uuid.uuid4().hex[:12]
    with _tasks_lock:
        upload_tasks[task_id] = {"status": "processing", "filename": filename, "progress": "parsing", "created_at": time.time()}

    def set_task(key, value):
        with _tasks_lock:
            if task_id in upload_tasks:
                upload_tasks[task_id][key] = value

    def set_task_done(data):
        with _tasks_lock:
            upload_tasks[task_id] = data

    def watchdog():
        with _tasks_lock:
            task = upload_tasks.get(task_id)
        if task and task.get("status") == "processing":
            logger.error(f"Upload task {task_id} ({filename}) timed out after 10 minutes")
            with _tasks_lock:
                upload_tasks[task_id] = {"status": "error", "filename": filename, "error": "Processing timed out (10 min limit)"}

    timer = threading.Timer(600, watchdog)
    timer.daemon = True
    timer.start()

    def process_in_background():
        try:
            def progress_callback(msg):
                set_task("progress", msg)
            cfg = _get_model_config()
            text = rag_engine.load_document(content, filename, task_id=task_id, model_config=cfg, progress_callback=progress_callback)
            if not text or not text.strip():
                set_task_done({"status": "error", "filename": filename, "error": "File is empty or has no readable text content"})
                return
            set_task("progress", "chunking")
            chunks = rag_engine.split_text(text)
            if not chunks:
                set_task_done({"status": "error", "filename": filename, "error": "File content could not be split into chunks"})
                return
            set_task("progress", "embedding")
            doc_id = uuid.uuid4().hex[:12]
            rag_engine.create_collection(chunks, doc_id, filename, cfg, progress_callback)
            set_task_done({
                "status": "done",
                "filename": filename,
                "doc_id": doc_id,
                "chunks": len(chunks),
            })
        except Exception:
            logger.exception("Upload processing failed")
            set_task_done({"status": "error", "filename": filename, "error": "Processing failed"})
        finally:
            timer.cancel()

    threading.Thread(target=process_in_background, daemon=True).start()
    return {"status": "processing", "task_id": task_id, "filename": filename}


@app.get("/api/upload/status/{task_id}")
async def upload_status(task_id: str):
    task = upload_tasks.get(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    return task


@app.get("/api/documents")
async def get_documents():
    return {"documents": rag_engine.list_documents()}


@app.get("/api/documents/search")
async def search_docs(q: str = "", limit: int = 10):
    if not q.strip():
        raise HTTPException(400, "Query parameter 'q' is required")
    limit = min(limit, 50)
    results = rag_engine.search_documents(q, limit)
    return {"query": q, "results": results}


@app.get("/api/documents/{doc_id}/content")
async def get_doc_content(doc_id: str):
    try:
        content = rag_engine.get_document_content(doc_id)
        if not content:
            raise HTTPException(404, "Document not found")
        return {"doc_id": doc_id, "chunks": content}
    except HTTPException:
        raise
    except Exception:
        logger.exception("Get document content failed")
        raise HTTPException(404, "Document not found")


@app.delete("/api/documents/{doc_id}")
async def delete_doc(doc_id: str):
    try:
        rag_engine.delete_document(doc_id)
        return {"status": "ok"}
    except Exception:
        logger.exception("Delete document failed")
        raise HTTPException(404, "Document not found")


@app.delete("/api/data")
async def clear_all():
    rag_engine.clear_all_data()
    return {"status": "ok", "message": "All data cleared"}


@app.post("/api/conversation")
async def save_conversation(body: ConversationSchema):
    rag_engine.save_messages(body.session_id, body.messages)
    return {"status": "ok"}


@app.get("/api/sessions")
async def get_sessions():
    return {"sessions": rag_engine.list_sessions()}


@app.get("/api/conversation")
async def get_conversation(session_id: str):
    if not session_id:
        raise HTTPException(400, "session_id is required")
    messages = rag_engine.load_conversation(session_id)
    return {"session_id": session_id, "messages": messages}


@app.delete("/api/sessions/{session_id}")
async def del_session(session_id: str):
    rag_engine.delete_session(session_id)
    return {"status": "ok"}


@app.post("/api/ask")
@limiter.limit("30/minute")
async def ask(request: Request, body: AskSchema):
    cfg = _get_model_config()
    if not cfg:
        raise HTTPException(400, "Model not configured. Please set API key, base URL and model first.")

    try:
        result = rag_engine.ask_question(body.question, body.chat_history, cfg)
        return result
    except RuntimeError:
        logger.exception("Ask question failed")
        raise HTTPException(502, "API request failed")


@app.post("/api/ask/stream")
@limiter.limit("30/minute")
async def ask_stream(request: Request, body: AskSchema):
    cfg = _get_model_config()
    if not cfg:
        raise HTTPException(400, "Model not configured.")

    def event_generator():
        try:
            for item in rag_engine.ask_question_stream(body.question, body.chat_history, cfg):
                if isinstance(item, dict) and item.get("_meta"):
                    yield f"data: {json.dumps({'meta': item})}\n\n"
                else:
                    yield f"data: {json.dumps({'token': item})}\n\n"
        except Exception:
            logger.exception("Streaming ask failed")
            yield f"data: {json.dumps({'error': 'Streaming request failed'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/ask/stream")
async def ask_stream_get():
    raise HTTPException(405, "Use POST instead of GET")


app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
