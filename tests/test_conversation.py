import pytest
from rag_engine import (
    save_messages, load_conversation, list_sessions,
    delete_session, clear_all_data
)


@pytest.fixture(autouse=True)
def clean_db():
    clear_all_data()


def test_save_and_load_conversation():
    msgs = [
        {"role": "user", "content": "What is AI?", "sources": []},
        {"role": "assistant", "content": "AI is artificial intelligence.", "sources": [{"filename": "test.txt"}]}
    ]
    save_messages("session1", msgs)
    loaded = load_conversation("session1")
    assert len(loaded) == 2
    assert loaded[0]["role"] == "user"
    assert loaded[0]["content"] == "What is AI?"
    assert loaded[1]["role"] == "assistant"


def test_list_sessions():
    save_messages("s1", [{"role": "user", "content": "q1"}])
    save_messages("s2", [{"role": "user", "content": "q2"}])
    sessions = list_sessions()
    assert len(sessions) >= 2
    ids = [s["id"] for s in sessions]
    assert "s1" in ids
    assert "s2" in ids


def test_delete_session():
    save_messages("del_test", [{"role": "user", "content": "q"}])
    delete_session("del_test")
    loaded = load_conversation("del_test")
    assert len(loaded) == 0


def test_empty_conversation():
    loaded = load_conversation("nonexistent")
    assert len(loaded) == 0


def test_conversation_preserves_sources():
    msgs = [{"role": "assistant", "content": "answer", "sources": [{"filename": "doc.pdf"}]}]
    save_messages("src_test", msgs)
    loaded = load_conversation("src_test")
    assert len(loaded) == 1
    assert loaded[0]["sources"][0]["filename"] == "doc.pdf"
