import pytest
import json
from fastapi.testclient import TestClient
from app import app, _model_config_lock, model_config

client = TestClient(app)


@pytest.fixture(autouse=True)
def clean():
    with _model_config_lock:
        model_config.clear()


def test_stream_without_config():
    resp = client.post("/api/ask/stream", json={"question": "test"})
    assert resp.status_code == 400


def test_stream_get_returns_405():
    resp = client.get("/api/ask/stream")
    assert resp.status_code == 405


def test_stream_with_invalid_config():
    with _model_config_lock:
        model_config.update({
            "api_key": "sk-invalid",
            "base_url": "http://127.0.0.1:99999",
            "model": "test-model"
        })
    resp = client.post("/api/ask/stream", json={"question": "test"})
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    content = resp.text
    assert "data:" in content
    data_lines = [line for line in content.split("\n") if line.startswith("data:")]
    assert len(data_lines) > 0
    last_data = json.loads(data_lines[-1].replace("data: ", ""))
    assert "error" in last_data


def test_stream_question_validation():
    with _model_config_lock:
        model_config.update({
            "api_key": "sk-test",
            "base_url": "https://api.test.com/v1",
            "model": "test-model"
        })
    resp = client.post("/api/ask/stream", json={"question": ""})
    assert resp.status_code == 422


def test_stream_empty_body():
    resp = client.post("/api/ask/stream", json={})
    assert resp.status_code == 422
