import pytest
from fastapi.testclient import TestClient
from app import app, _model_config_lock, model_config

client = TestClient(app)


@pytest.fixture(autouse=True)
def clean():
    with _model_config_lock:
        model_config.clear()


def test_ask_with_unreachable_api():
    with _model_config_lock:
        model_config.update({
            "api_key": "sk-test",
            "base_url": "http://127.0.0.1:1",
            "model": "test-model"
        })
    resp = client.post("/api/ask", json={"question": "test"})
    assert resp.status_code == 502


def test_upload_empty_file():
    resp = client.post("/api/upload", files={"file": ("empty.txt", b"", "text/plain")})
    assert resp.status_code == 400


def test_upload_wrong_extension():
    resp = client.post("/api/upload", files={"file": ("test.exe", b"some content", "application/octet-stream")})
    assert resp.status_code == 400


def test_upload_fake_pdf():
    resp = client.post("/api/upload", files={"file": ("fake.pdf", b"not a pdf", "application/pdf")})
    assert resp.status_code == 400


def test_upload_fake_docx():
    resp = client.post("/api/upload", files={"file": ("fake.docx", b"not a docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")})
    assert resp.status_code == 400


def test_config_invalid_base_url():
    resp = client.post("/api/config", json={
        "api_key": "sk-test",
        "base_url": "",
        "model": "test-model"
    })
    assert resp.status_code == 422


def test_delete_nonexistent_document():
    resp = client.delete("/api/documents/nonexistent")
    assert resp.status_code == 200


def test_upload_status_nonexistent_task():
    resp = client.get("/api/upload/status/nonexistent")
    assert resp.status_code == 404


def test_conversation_empty_messages():
    resp = client.post("/api/conversation", json={
        "session_id": "test",
        "messages": []
    })
    assert resp.status_code == 422


def test_conversation_missing_session():
    resp = client.post("/api/conversation", json={
        "messages": [{"role": "user", "content": "hi"}]
    })
    assert resp.status_code == 422
