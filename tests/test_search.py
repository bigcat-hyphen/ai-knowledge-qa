import pytest
from rag_engine import query_collection, create_collection, list_documents, delete_document, clear_all_data, _embed, _cosine_sim


def test_store_and_retrieve():
    clear_all_data()
    create_collection(["test content here", "another chunk"], "doc1", "test.txt")
    docs = list_documents()
    assert len(docs) >= 1
    doc = [d for d in docs if d["id"] == "doc1"]
    assert len(doc) == 1
    assert doc[0]["filename"] == "test.txt"
    assert doc[0]["chunks"] == 2
    clear_all_data()


def test_query_returns_results():
    clear_all_data()
    create_collection(["the cat sat on the mat", "dogs are great pets", "quantum physics theory"], "doc2", "animals.txt")
    results = query_collection("cat", n_results=2)
    assert len(results) <= 2
    assert len(results) > 0
    clear_all_data()


def test_query_empty_collection():
    clear_all_data()
    results = query_collection("anything")
    assert len(results) == 0


def test_delete_document():
    clear_all_data()
    create_collection(["delete me"], "delete_doc", "temp.txt")
    docs_before = list_documents()
    delete_document("delete_doc")
    docs_after = list_documents()
    assert len(docs_after) < len(docs_before)
