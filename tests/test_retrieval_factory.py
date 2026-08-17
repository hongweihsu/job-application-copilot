import pytest

from app.retrieval import BM25Retriever, build_evidence_retriever, configured_retriever_name


def test_retriever_configuration_defaults_to_bm25(monkeypatch):
    monkeypatch.delenv("LLM_RETRIEVER", raising=False)

    assert configured_retriever_name() == "bm25"
    assert isinstance(build_evidence_retriever(), BM25Retriever)


def test_retriever_configuration_normalizes_environment_value(monkeypatch):
    monkeypatch.setenv("LLM_RETRIEVER", " EMBEDDING ")

    assert configured_retriever_name() == "embedding"


def test_retriever_configuration_rejects_unpromoted_or_unknown_method(monkeypatch):
    monkeypatch.setenv("LLM_RETRIEVER", "hybrid_rrf")

    with pytest.raises(ValueError, match="bm25, embedding"):
        configured_retriever_name()


def test_explicit_retriever_name_is_validated():
    with pytest.raises(ValueError, match="bm25, embedding"):
        build_evidence_retriever("unknown")
