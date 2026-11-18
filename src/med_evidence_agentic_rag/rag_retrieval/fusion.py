"""Rank fusion shared by the original and optional multilingual recall routes."""
from collections import defaultdict

from med_evidence_agentic_rag.rag_retrieval.models import Evidence, RetrievalResult


def fuse_results(
    query: str, results: list[RetrievalResult], *, top_k: int, rrf_k: int
) -> RetrievalResult:
    if top_k < 1 or rrf_k < 1:
        raise ValueError("融合参数必须大于 0")
    scores: defaultdict[tuple[str, str], float] = defaultdict(float)
    originals: dict[tuple[str, str], Evidence] = {}
    for result in results:
        seen: set[tuple[str, str]] = set()
        for rank, evidence in enumerate(result.evidences, 1):
            key = (evidence.source, evidence.content)
            if key in seen:
                continue
            seen.add(key)
            scores[key] += 1 / (rrf_k + rank)
            originals.setdefault(key, evidence)
    ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:top_k]
    evidences = [
        originals[key].model_copy(update={"citation_id": f"E{i}", "score": scores[key]})
        for i, key in enumerate(ranked, 1)
    ]
    return RetrievalResult(query=query, found=bool(evidences), evidences=evidences)
