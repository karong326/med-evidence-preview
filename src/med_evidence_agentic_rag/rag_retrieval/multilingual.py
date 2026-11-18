"""Optional local multilingual recall over an invalidatable Chroma snapshot.

The original BGE index is never rewritten with a different embedding model.
Visibility is applied BEFORE top-k selection, including anonymous requests.
"""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from threading import RLock
from typing import Any, cast
from uuid import UUID

import numpy as np
from langchain_core.documents import Document

from med_evidence_agentic_rag.rag_retrieval.models import Evidence, RetrievalResult


@dataclass(frozen=True)
class MultilingualSnapshot:
    documents: tuple[Document, ...]
    vectors: np.ndarray

    def __post_init__(self) -> None:
        if self.vectors.ndim != 2 or len(self.documents) != len(self.vectors):
            raise ValueError("辅助向量与文档必须逐行对应")

    def search(
        self, query_vector: np.ndarray, *, top_k: int, user_id: UUID | None = None
    ) -> list[Evidence]:
        if top_k < 1:
            raise ValueError("top_k 必须大于 0")
        allowed = [
            i
            for i, doc in enumerate(self.documents)
            if doc.metadata.get("source_type") == "base"
            or (user_id is not None and doc.metadata.get("owner_user_id") == str(user_id))
        ]
        if not allowed:
            return []
        scores = self.vectors[allowed] @ query_vector
        ranked = np.argsort(-scores, kind="stable")[:top_k]
        result: list[Evidence] = []
        for position in ranked:
            doc = self.documents[allowed[int(position)]]
            source = str(doc.metadata.get("source", "未知来源"))
            result.append(
                Evidence(
                    citation_id=f"E{len(result) + 1}",
                    score=float(scores[position]),
                    source=source,
                    title=str(doc.metadata.get("title") or Path(source).stem),
                    content=doc.page_content,
                )
            )
        return result


_lock = RLock()
_snapshot: MultilingualSnapshot | None = None
_model: Any = None
# Reuse only vectors belonging to the latest corpus; no query/answer cache or user result reuse.
_vector_cache: dict[str, np.ndarray] = {}


def invalidate_multilingual_index() -> None:
    global _snapshot
    with _lock:
        _snapshot = None


def _get_snapshot() -> MultilingualSnapshot | None:
    global _snapshot, _model, _vector_cache
    from med_evidence_agentic_rag.utils.config import multilingual_embedding_model_path

    if not multilingual_embedding_model_path:
        return None
    with _lock:
        if _snapshot is not None:
            return _snapshot
        from sentence_transformers import SentenceTransformer

        from med_evidence_agentic_rag.models_and_database.chroma import vectorstore

        if _model is None:
            _model = SentenceTransformer(
                multilingual_embedding_model_path, device="cuda", local_files_only=True
            )
        stored = vectorstore.get(include=["documents", "metadatas"])
        documents = tuple(
            Document(page_content=text, metadata=metadata or {})
            for text, metadata in zip(
                stored.get("documents") or [], stored.get("metadatas") or [], strict=True
            )
            if isinstance(text, str) and text.strip()
        )
        if documents:
            digests = [sha256(doc.page_content.encode("utf-8")).hexdigest() for doc in documents]
            missing = {
                digest: doc.page_content
                for digest, doc in zip(digests, documents, strict=True)
                if digest not in _vector_cache
            }
            encoded = (
                cast(
                    list[np.ndarray],
                    _model.encode(
                        list(missing.values()),
                        batch_size=32,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                    ),
                )
                if missing
                else []
            )
            additions: dict[str, np.ndarray] = dict(zip(missing, encoded, strict=True))
            # Drop deleted content rather than accumulating an unbounded cache.
            current = {
                digest: _vector_cache[digest] if digest in _vector_cache else additions[digest]
                for digest in digests
            }
            vectors = np.stack([current[digest] for digest in digests])
        else:
            vectors = np.empty((0, _model.get_sentence_embedding_dimension()))
            current = {}
        _snapshot = MultilingualSnapshot(documents, np.asarray(vectors))
        _vector_cache = current
        return _snapshot


def prewarm_multilingual_index() -> int:
    snapshot = _get_snapshot()
    return len(snapshot.documents) if snapshot is not None else 0


def retrieve_multilingual(
    query: str, *, top_k: int, user_id: UUID | None = None
) -> RetrievalResult:
    normalized = query.strip()
    if not normalized or top_k < 1:
        raise ValueError("query 不能为空且 top_k 必须大于 0")
    # Keep index acquisition and query encoding atomic against invalidation/rebuild.
    with _lock:
        snapshot = _get_snapshot()
        if snapshot is None:
            return RetrievalResult(query=normalized, found=False)
        vector = _model.encode(normalized, normalize_embeddings=True, show_progress_bar=False)
        evidences = snapshot.search(np.asarray(vector), top_k=top_k, user_id=user_id)
    return RetrievalResult(query=normalized, found=bool(evidences), evidences=evidences)
