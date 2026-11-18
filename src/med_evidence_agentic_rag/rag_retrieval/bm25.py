"""维护 BM25 内存索引并执行关键词召回。

模块从 Chroma 当前 chunks 构建语料、分词和 BM25 索引，支持启动预热和资料变化后的失效重建。
检索时只读取基础资料和当前用户自己的资料，再映射为统一 Evidence；其结果将与 Dense 召回
在 hybrid 模块中融合。
"""

from collections.abc import Callable, Sequence
from pathlib import Path
from threading import RLock
from time import perf_counter
from typing import Any, Protocol, cast
from uuid import UUID

import jieba  # pyright: ignore[reportMissingTypeStubs]
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi  # pyright: ignore[reportMissingTypeStubs]

from med_evidence_agentic_rag.models_and_database.chroma import vectorstore
from med_evidence_agentic_rag.rag_retrieval.models import Evidence, RetrievalResult
from med_evidence_agentic_rag.utils.config import hybrid_candidate_k
from med_evidence_agentic_rag.utils.logger import setup_logger

logger = setup_logger(__name__)


class BM25Index(Protocol):
    def get_scores(self, query: list[str]) -> Sequence[float]: ...


_bm25: BM25Index | None = None
_documents: list[Document] = []
_index_lock = RLock()


def invalidate_bm25_index() -> None:
    """让旧的内存索引失效。

    输入：无；调用方已经完成 Chroma 资料的新增、替换或删除。
    输出：无；下一次检索会从最新 chunks 自动重建索引。
    """
    global _bm25, _documents

    with _index_lock:
        was_ready = _bm25 is not None
        _bm25 = None
        _documents = []
    # 所有入库/删除路径已调用此失效入口，同时淘汰辅助向量快照。
    from med_evidence_agentic_rag.rag_retrieval.multilingual import invalidate_multilingual_index

    invalidate_multilingual_index()
    if was_ready:
        logger.info("BM25 内存索引已失效，下次关键词检索将自动重建")


def prewarm_bm25_index() -> int:
    """在启动时准备索引，避免首个用户请求承担构建耗时。

    输入：无。
    输出：当前索引中的文本块数量；没有可索引资料时为 0。
    """
    with _index_lock:
        _ensure_bm25_index()
        count = len(_documents)
    from med_evidence_agentic_rag.rag_retrieval.multilingual import prewarm_multilingual_index

    prewarm_multilingual_index()
    return count


def tokenize(text: str) -> list[str]:
    """使用 jieba 切分中英文文本，并删除空白和纯标点。"""
    return [
        token.lower()
        for token in cast(Callable[[str], list[str]], getattr(jieba, "lcut"))(text)
        if token.strip() and any(character.isalnum() for character in token)
    ]


def _build_index() -> None:
    """从 Chroma 当前 chunks 重建内存 BM25 索引。

    输入：无，数据来源为 Chroma。
    输出：无；更新模块内的 ``_documents`` 与 ``_bm25``。
    """
    global _bm25, _documents

    stored = vectorstore.get(include=["documents", "metadatas"])
    contents = cast(list[object], stored.get("documents") or [])
    metadatas = cast(list[dict[str, Any] | None], stored.get("metadatas") or [])

    documents = [
        Document(page_content=content, metadata=metadata or {})
        for content, metadata in zip(contents, metadatas, strict=False)
        if isinstance(content, str) and content.strip()
    ]
    tokenized_documents = [tokenize(document.page_content) for document in documents]
    factory = cast(Callable[[list[list[str]]], BM25Index], BM25Okapi)
    index = factory(tokenized_documents) if tokenized_documents else None
    # Publication is protected by _ensure_bm25_index's reentrant lock.
    _documents, _bm25 = documents, index
    logger.info("BM25 内存索引已建立：文本块数=%d", len(documents))


def _ensure_bm25_index() -> bool:
    """确保索引可用于检索。

    输入：无。
    输出：索引是否可用；Chroma 没有有效文本块时返回 ``False``。
    """
    with _index_lock:
        if _bm25 is None:
            _build_index()
        return _bm25 is not None


# 主方法：
def retrieve_bm25(
    query: str, *, top_k: int = hybrid_candidate_k, user_id: UUID | None = None
) -> RetrievalResult:
    """按关键词检索基础资料与当前用户资料。

    输入：查询文本 ``query``、最多返回数量 ``top_k``、当前用户编号 ``user_id``。
    输出：统一 ``RetrievalResult``；无匹配或无可索引资料时 ``found=False``。
    """
    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("query 不能为空")
    if top_k < 1:
        raise ValueError("top_k 必须大于 0")
    # 计时：
    started_at = perf_counter()

    with _index_lock:
        if not _ensure_bm25_index():
            return RetrievalResult(query=normalized_query, found=False, evidences=[])
        # Keep this request's index and documents paired even if ingestion invalidates globals.
        index = _bm25
        documents = tuple(_documents)
    assert index is not None

    # bm25里面的一个计算得分算法：get_scores（关键词列表）
    scores = index.get_scores(tokenize(normalized_query))

    # 关键边界：所有人都可检索基础资料；个人资料只允许其所有者本人检索。
    allowed_indexes = [
        index
        for index, document in enumerate(documents)
        if document.metadata.get("source_type") == "base"
        or (user_id is not None and document.metadata.get("owner_user_id") == str(user_id))
    ]
    ranked_indexes = sorted(allowed_indexes, key=scores.__getitem__, reverse=True)[:top_k]

    evidences: list[Evidence] = []
    for document_index in ranked_indexes:
        score = float(scores[document_index])
        if score <= 0:
            continue
        document = documents[document_index]
        source = str(document.metadata.get("source", "未知来源"))
        evidence = Evidence(
            citation_id=f"E{len(evidences) + 1}",
            score=score,
            source=source,
            title=str(document.metadata.get("title") or Path(source).stem),
            content=document.page_content,
        )
        evidences.append(evidence)

    logger.info(
        "[BM25 检索] 问题=%s，候选=%d，耗时=%.2f秒",
        normalized_query,
        len(evidences),
        perf_counter() - started_at,
    )
    return RetrievalResult(query=normalized_query, found=bool(evidences), evidences=evidences)


if __name__ == "__main__":
    result = retrieve_bm25("医院感染管理")
    logger.info("BM25 独立测试完成：是否命中=%s，证据数=%d", result.found, len(result.evidences))
