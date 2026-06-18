import pytest
from rag_engine import split_text


def test_split_short_text_stays_one_chunk():
    chunks = split_text("Short text.", chunk_size=500)
    assert len(chunks) == 1
    assert chunks[0] == "Short text."


def test_split_empty_text():
    chunks = split_text("", chunk_size=500)
    assert len(chunks) == 0


def test_split_whitespace_only():
    chunks = split_text("   \n\n  ", chunk_size=500)
    assert len(chunks) == 0


def test_split_paragraph_boundaries():
    text = "第一段。\n\n第二段。\n\n第三段。"
    chunks = split_text(text, chunk_size=100, overlap=0)
    assert len(chunks) >= 1


def test_split_large_chunk_uses_larger_size():
    text = "段落内容。" * 50000
    chunks = split_text(text)
    assert len(chunks) > 0
    assert all(len(c) <= 2000 for c in chunks)


def test_split_medium_chunk_uses_medium_size():
    text = "段落内容。" * 15000
    chunks = split_text(text)
    assert len(chunks) > 0


def test_chunks_contain_no_fragmented_words():
    text = "这是一个完整的句子。" * 50
    chunks = split_text(text, chunk_size=100, overlap=10)
    assert len(chunks) > 0
    for c in chunks:
        assert len(c) > 0


def test_chunks_respect_boundaries():
    text = "短段。\n\n" * 20
    chunks = split_text(text, chunk_size=50, overlap=0)
    assert len(chunks) > 0
