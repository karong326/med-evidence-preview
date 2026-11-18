"""Preserve the source identity when semantically ranking similar protocol text."""
from pathlib import Path

from med_evidence_agentic_rag.rag_retrieval.models import Evidence


def source_text(source: str, content: str) -> str:
    """Use filename metadata, never an inferred answer or evaluation label."""
    return f"{Path(source).stem.replace('_', ' ')}\n{content}"


def evidence_text(evidence: Evidence) -> str:
    return source_text(evidence.source, evidence.content)
