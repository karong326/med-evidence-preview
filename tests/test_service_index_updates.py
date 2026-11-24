"""Daily update regressions, independent of retrieval benchmark questions."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


def load_bm25_with_store(store, monkeypatch):
    # Avoid initializing a real Chroma client or GPU in the index lifecycle test.
    __import__("sentence_transformers")
    import med_evidence_agentic_rag.rag_retrieval as package

    path = Path(package.__file__).parent / "bm25.py"
    spec = importlib.util.spec_from_file_location("bm25_update_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(
        sys.modules,
        "med_evidence_agentic_rag.models_and_database.chroma",
        SimpleNamespace(vectorstore=store),
    )
    spec.loader.exec_module(module)
    return module


def test_invalidation_during_scoring_keeps_request_documents_aligned(monkeypatch):
    stored = {
        "documents": ["hospital infection", "sun moon", "rain wind"],
        "metadatas": [{"source": name, "source_type": "base"} for name in ["a", "b", "c"]],
    }
    module = load_bm25_with_store(SimpleNamespace(get=lambda **kwargs: stored), monkeypatch)
    module._ensure_bm25_index()
    original = module._bm25.get_scores

    def score_then_update(tokens):
        scores = original(tokens)
        # Deterministic interleaving of a concurrent upload/delete invalidation.
        module.invalidate_bm25_index()
        return scores

    module._bm25.get_scores = score_then_update
    result = module.retrieve_bm25("hospital", top_k=1)
    assert [e.content for e in result.evidences] == ["hospital infection"]


def test_auxiliary_rebuild_encodes_only_changes_and_evicts_deleted_content():
    __import__("sentence_transformers")
    from med_evidence_agentic_rag.rag_retrieval import multilingual as m
    from med_evidence_agentic_rag.utils import config

    class Encoder:
        def __init__(self):
            self.calls = []

        def encode(self, texts, **kwargs):
            self.calls.append(list(texts))
            return np.array([[1.0, 0.0] for _ in texts])

    encoder = Encoder()
    stored = {
        "documents": ["alpha", "beta"],
        "metadatas": [
            {"source": "a", "source_type": "base"},
            {"source": "b", "source_type": "base"},
        ],
    }
    store = SimpleNamespace(get=lambda **kwargs: stored)
    with (
        patch.object(config, "multilingual_embedding_model_path", "local-test"),
        patch.object(m, "_snapshot", None),
        patch.object(m, "_model", encoder),
        patch.object(m, "_vector_cache", {}, create=True),
        patch.dict(
            sys.modules,
            {
                "med_evidence_agentic_rag.models_and_database.chroma": SimpleNamespace(
                    vectorstore=store
                )
            },
        ),
    ):
        m._get_snapshot()
        stored["documents"] = ["beta", "gamma"]
        m.invalidate_multilingual_index()
        snapshot = m._get_snapshot()
        assert [doc.page_content for doc in snapshot.documents] == ["beta", "gamma"]
        assert encoder.calls == [["alpha", "beta"], ["gamma"]]
        stored["documents"] = ["alpha", "gamma"]
        m.invalidate_multilingual_index()
        m._get_snapshot()
        assert encoder.calls[-1] == ["alpha"]
