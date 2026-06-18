import pytest
from fastapi.testclient import TestClient
from app import app, _model_config_lock, model_config

client = TestClient(app)


@pytest.fixture(autouse=True)
def clear_config():
    with _model_config_lock:
        model_config.clear()


def test_home_page():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


def test_config_not_configured():
    resp = client.get("/api/config")
    assert resp.status_code == 200
    data = resp.json()
    assert data["configured"] is False
    assert "api_key" not in str(data)


def test_config_missing_fields():
    resp = client.post("/api/config", json={"api_key": "test"})
    assert resp.status_code == 422


def test_config_field_too_long():
    resp = client.post("/api/config", json={
        "api_key": "sk-" + "x" * 5000,
        "base_url": "https://api.test.com/v1",
        "model": "test-model"
    })
    assert resp.status_code == 422


def test_save_and_retrieve_config():
    resp = client.post("/api/config", json={
        "api_key": "sk-test123",
        "base_url": "https://api.test.com/v1",
        "model": "test-model"
    })
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

    resp = client.get("/api/config")
    assert resp.status_code == 200
    data = resp.json()
    assert data["configured"] is True
    assert data["base_url"] == "https://api.test.com/v1"
    assert "api_key" not in str(data)


def test_get_ask_stream_returns_405():
    resp = client.get("/api/ask/stream")
    assert resp.status_code == 405


def test_ask_without_config():
    resp = client.post("/api/ask", json={"question": "test"})
    assert resp.status_code == 400


def test_empty_upload_rejected():
    resp = client.post("/api/upload")
    assert resp.status_code == 422
