"""执行第一阶段 Dense 向量 Top-K 检索。

模块以 Chroma 相似度搜索召回文本块，按 source_type 与 owner_user_id 应用共享/私有可见性过滤，
并将原始文档和距离转换为统一 RetrievalResult/Evidence。它不做关键词召回、RRF 或重排序，
这些步骤由相邻模块分别承担。
"""

from pathlib import Path
from time import perf_counter
from uuid import UUID

from med_evidence_agentic_rag.rag_retrieval.models import Evidence, RetrievalResult
from med_evidence_agentic_rag.utils.config import hybrid_candidate_k
from med_evidence_agentic_rag.utils.logger import setup_logger

logger = setup_logger(__name__)


# 主方法：
def retrieve(
    query: str, *, top_k: int = hybrid_candidate_k, user_id: UUID | None = None
) -> RetrievalResult:
    """在本地 Chroma 中进行一次最小 Dense 检索。"""
    from med_evidence_agentic_rag.models_and_database.chroma import vectorstore

    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("query 不能为空")
    if top_k < 1:
        raise ValueError("top_k 必须大于 0")
    started_at = perf_counter()

    # 第一阶段只执行带相关性分数的 Dense Top-K，不提前加入混合检索或重排序。
    matches = vectorstore.similarity_search_with_relevance_scores(
        normalized_query,
        k=top_k,
        filter=(
            {"$or": [{"source_type": "base"}, {"owner_user_id": str(user_id)}]}
            if user_id
            else {"source_type": "base"}
        ),
    )

    # 将 LangChain Document 转成 Agent 可引用的稳定证据结构。
    evidences: list[Evidence] = []

    for index, (document, score) in enumerate(matches, start=1):
        if not document.page_content.strip():
            continue
        source = str(document.metadata.get("source", "未知来源"))

        evidence = Evidence(
            citation_id=f"E{index}",
            score=float(score),
            source=source,
            title=str(document.metadata.get("title") or Path(source).stem),
            content=document.page_content,
        )

        evidences.append(evidence)

    logger.info(
        "Dense retrieval completed: candidates=%d, seconds=%.2f",
        len(evidences),
        perf_counter() - started_at,
    )
    return RetrievalResult(query=normalized_query, found=bool(evidences), evidences=evidences)
