"""Regression cases: cross-user leakage, stale auxiliary vectors, source ambiguity."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch
from uuid import UUID

import numpy as np
from langchain_core.documents import Document


class MultilingualVisibilityTests(unittest.TestCase):
    def test_private_nearest_neighbor_is_filtered_before_top_k(self):
        from med_evidence_agentic_rag.rag_retrieval.multilingual import MultilingualSnapshot

        user = UUID("11111111-1111-1111-1111-111111111111")
        other = UUID("22222222-2222-2222-2222-222222222222")
        docs = (
            Document(
                page_content="other secret",
                metadata={"source": "private", "source_type": "user", "owner_user_id": str(other)},
            ),
            Document(page_content="public", metadata={"source": "public", "source_type": "base"}),
            Document(
                page_content="my note",
                metadata={"source": "own", "source_type": "user", "owner_user_id": str(user)},
            ),
        )
        index = MultilingualSnapshot(docs, np.array([[1.0, 0.0], [0.8, 0.6], [0.6, 0.8]]))
        public = index.search(np.array([1.0, 0.0]), top_k=1, user_id=None)
        own = index.search(np.array([1.0, 0.0]), top_k=3, user_id=user)
        self.assertEqual([e.content for e in public], ["public"])
        self.assertEqual([e.content for e in own], ["public", "my note"])

    def test_vector_document_alignment_is_validated(self):
        from med_evidence_agentic_rag.rag_retrieval.multilingual import MultilingualSnapshot

        with self.assertRaises(ValueError):
            MultilingualSnapshot((Document(page_content="x"),), np.zeros((2, 3)))

    def test_snapshot_replacement_does_not_reuse_deleted_documents(self):
        from med_evidence_agentic_rag.rag_retrieval.multilingual import MultilingualSnapshot

        a = Document(page_content="old", metadata={"source": "a", "source_type": "base"})
        b = Document(page_content="new", metadata={"source": "b", "source_type": "base"})
        old = MultilingualSnapshot((a,), np.array([[1.0, 0.0]]))
        new = MultilingualSnapshot((b,), np.array([[0.0, 1.0]]))
        self.assertEqual(old.search(np.array([1.0, 0.0]), top_k=1)[0].content, "old")
        self.assertEqual(new.search(np.array([1.0, 0.0]), top_k=1)[0].content, "new")

    def test_invalidation_rebuilds_from_current_store_after_deletion(self):
        import sys

        # Load heavy lazy dependencies before patch.dict restores sys.modules;
        # removing initialized torch modules would leave C-extension state behind.
        __import__("sentence_transformers")

        from med_evidence_agentic_rag.rag_retrieval import multilingual as m
        from med_evidence_agentic_rag.utils import config

        a = Document(page_content="deleted", metadata={"source": "a", "source_type": "base"})
        stale = m.MultilingualSnapshot((a,), np.array([[1.0, 0.0]]))

        class Encoder:
            def encode(self, values, **kwargs):
                return np.array([[0.0, 1.0]]) if isinstance(values, list) else np.array([0.0, 1.0])

        store = SimpleNamespace(
            get=lambda **kwargs: {
                "documents": ["replacement"],
                "metadatas": [{"source": "b", "source_type": "base"}],
            }
        )
        with (
            patch.object(config, "multilingual_embedding_model_path", "local-test"),
            patch.object(m, "_snapshot", stale),
            patch.object(m, "_model", Encoder()),
            patch.object(m, "_vector_cache", {}),
            patch.dict(
                sys.modules,
                {
                    "med_evidence_agentic_rag.models_and_database.chroma": SimpleNamespace(
                        vectorstore=store
                    )
                },
            ),
        ):
            m.invalidate_multilingual_index()
            result = m.retrieve_multilingual("query", top_k=3)
            self.assertEqual([e.content for e in result.evidences], ["replacement"])


class FusionTests(unittest.TestCase):
    def test_auxiliary_route_is_fused_and_duplicate_evidence_is_not_repeated(self):
        from med_evidence_agentic_rag.rag_retrieval.fusion import fuse_results
        from med_evidence_agentic_rag.rag_retrieval.models import Evidence, RetrievalResult

        def evidence(text):
            return Evidence(
                citation_id="old", score=99, source="guide.pdf", title="Guide", content=text
            )

        routes = [
            RetrievalResult(query="q", found=True, evidences=[evidence("a"), evidence("b")]),
            RetrievalResult(query="q", found=True, evidences=[evidence("b")]),
            RetrievalResult(query="q", found=True, evidences=[evidence("c")]),
        ]
        result = fuse_results("q", routes, top_k=3, rrf_k=50)
        self.assertEqual([e.content for e in result.evidences], ["b", "a", "c"])
        self.assertEqual([e.citation_id for e in result.evidences], ["E1", "E2", "E3"])


class SourceContextTests(unittest.TestCase):
    def test_identical_text_from_different_protocols_has_distinct_model_input(self):
        from med_evidence_agentic_rag.rag_retrieval.models import Evidence
        from med_evidence_agentic_rag.rag_retrieval.text_context import evidence_text

        a = Evidence(
            citation_id="E1",
            score=0,
            source="NIDCR_Observational_Protocol.docx",
            title="Observational Protocol",
            content="Describe monitoring visits.",
        )
        b = a.model_copy(
            update={
                "source": "NIDCR_Interventional_Protocol.docx",
                "title": "Interventional Protocol",
            }
        )
        self.assertIn("Observational", evidence_text(a))
        self.assertIn("Interventional", evidence_text(b))
        self.assertNotEqual(evidence_text(a), evidence_text(b))
        self.assertTrue(evidence_text(a).endswith("Describe monitoring visits."))


class ProbabilityPrecisionTests(unittest.TestCase):
    def test_half_precision_logits_do_not_collapse_close_probability_scores(self):
        import torch

        from med_evidence_agentic_rag.models_and_database.inference import Float32Sigmoid

        logits = torch.tensor([5.0, 4.984375], dtype=torch.float16)
        self.assertEqual(float(torch.sigmoid(logits)[0]), float(torch.sigmoid(logits)[1]))
        scores = Float32Sigmoid()(logits)
        self.assertEqual(scores.dtype, torch.float32)
        self.assertGreater(float(scores[0]), float(scores[1]))


if __name__ == "__main__":
    unittest.main()
