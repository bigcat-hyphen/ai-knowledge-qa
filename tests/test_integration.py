import pytest
from fastapi.testclient import TestClient
from app import app, _model_config_lock, model_config

client = TestClient(app)


@pytest.fixture(autouse=True)
def clean():
    with _model_config_lock:
        model_config.clear()


def test_health_endpoint():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "documents" in data
    assert "sessions" in data
    assert data["version"] == "1.0"


def test_search_empty_query():
    resp = client.get("/api/documents/search?q=")
    assert resp.status_code == 400


def test_search_with_results():
    resp = client.post("/api/config", json={
        "api_key": "sk-test",
        "base_url": "https://api.test.com/v1",
        "model": "test-model"
    })
    assert resp.status_code == 200

    resp = client.get("/api/documents/search?q=test")
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data


def test_sessions_empty():
    resp = client.get("/api/sessions")
    assert resp.status_code == 200
    data = resp.json()
    assert "sessions" in data


def test_conversation_save_load():
    resp = client.post("/api/conversation", json={
        "session_id": "test-session",
        "messages": [{"role": "user", "content": "hello"}]
    })
    assert resp.status_code == 200

    resp = client.get("/api/conversation?session_id=test-session")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["messages"]) == 1
    assert data["messages"][0]["content"] == "hello"


def test_conversation_delete():
    resp = client.post("/api/conversation", json={
        "session_id": "del-test",
        "messages": [{"role": "user", "content": "hi"}]
    })
    assert resp.status_code == 200

    resp = client.delete("/api/sessions/del-test")
    assert resp.status_code == 200

    resp = client.get("/api/conversation?session_id=del-test")
    assert resp.status_code == 200
    assert len(resp.json()["messages"]) == 0


def test_config_validation():
    resp = client.post("/api/config", json={})
    assert resp.status_code == 422

    resp = client.post("/api/config", json={"api_key": "", "base_url": "", "model": ""})
    assert resp.status_code == 422


def test_request_id_generated():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert "X-Request-ID" in resp.headers
    assert len(resp.headers["X-Request-ID"]) == 12


def test_request_id_passthrough():
    custom_id = "my-custom-id"
    resp = client.get("/api/health", headers={"X-Request-ID": custom_id})
    assert resp.status_code == 200
    assert resp.headers["X-Request-ID"] == custom_id


def test_embed_model_in_config():
    resp = client.post("/api/config", json={
        "api_key": "sk-test",
        "base_url": "https://api.test.com/v1",
        "model": "test-model",
        "embed_model": "custom-embed"
    })
    assert resp.status_code == 200

    resp = client.get("/api/config")
    data = resp.json()
    assert data["configured"] is True
    assert data["embed_model"] == "custom-embed"
