import pytest
from rag_engine import _embed, _cosine_sim


def test_embed_return_list_of_floats():
    vec = _embed("hello world")
    assert isinstance(vec, list)
    assert len(vec) > 0
    assert all(isinstance(v, float) for v in vec)


def test_embed_has_correct_dimension():
    vec = _embed("test")
    assert len(vec) == 1536


def test_embed_deterministic():
    v1 = _embed("same text")
    v2 = _embed("same text")
    assert v1 == v2


def test_embed_different_inputs_different():
    v1 = _embed("apple")
    v2 = _embed("banana")
    assert v1 != v2


def test_embed_empty_text():
    vec = _embed("")
    assert len(vec) == 1536
    assert all(v == 0.0 for v in vec)


def test_embed_normalized():
    vec = _embed("some text to normalize")
    norm = sum(v * v for v in vec) ** 0.5
    assert abs(norm - 1.0) < 1e-6


def test_cosine_sim_identical():
    vec = _embed("test")
    sim = _cosine_sim(vec, vec)
    assert abs(sim - 1.0) < 1e-6


def test_cosine_sim_orthogonal():
    a = [1.0, 0.0, 0.0]
    b = [0.0, 1.0, 0.0]
    sim = _cosine_sim(a, b)
    assert abs(sim) < 1e-6


def test_cosine_sim_similar():
    v1 = _embed("artificial intelligence")
    v2 = _embed("machine learning")
    sim = _cosine_sim(v1, v2)
    assert sim >= 0
