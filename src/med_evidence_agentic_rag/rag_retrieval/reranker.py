"""使用 Cross-Encoder 对混合召回候选进行精确重排序。

``rerank`` 将 query 与每条候选正文配对评分，按分数降序筛选阈值以上的最终 Top-K，并重新编号
引用。``retrieve_with_rerank`` 是给知识库 Tool 使用的完整检索入口：先 RRF 融合候选，再调用
本模块；模型对象可注入以支持测试。
"""

from time import perf_counter
from typing import Any
from uuid import UUID

from med_evidence_agentic_rag.rag_retrieval.hybrid import retrieve_hybrid
from med_evidence_agentic_rag.rag_retrieval.models import Evidence, RetrievalResult
from med_evidence_agentic_rag.rag_retrieval.text_context import evidence_text
from med_evidence_agentic_rag.utils.config import (
    reranker_batch_size,
    reranker_include_source,
    reranker_min_score,
    reranker_top_k,
)
from med_evidence_agentic_rag.utils.logger import setup_logger

logger = setup_logger(__name__)


def rerank(
    query: str,
    candidates: list[Evidence],
    *,
    model: Any | None = None,
    top_k: int = reranker_top_k,
    batch_size: int = reranker_batch_size,
    min_score: float = reranker_min_score,
) -> RetrievalResult:
    """让模型同时阅读问题与候选正文，并按相关性分数重新排序。"""
    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("query 不能为空")
    if top_k < 1 or batch_size < 1:
        raise ValueError("重排序参数必须大于 0")
    if not candidates:
        return RetrievalResult(query=normalized_query, found=False, evidences=[])

    started_at = perf_counter()
    model_load_started_at = perf_counter()

    # 仅在真正需要重排序时加载全局模型，普通聊天和空结果不会加载。
    if model is None:
        from med_evidence_agentic_rag.models_and_database.reranker import (
            reranker as model,
        )

    # 模型加载时间：
    model_load_seconds = perf_counter() - model_load_started_at

    # 每个’输入对‘都是：“对应的用户问题 + 对应的一条候选原文”。
    pairs = [
        [normalized_query, evidence_text(evidence) if reranker_include_source else evidence.content]
        for evidence in candidates
    ]
    inference_started_at = perf_counter()

    # 进入重排序模型，返回新的来源标识
    scores = model.predict(
        pairs,
        batch_size=batch_size,
        show_progress_bar=False,
    )

    # 推理时间：
    inference_seconds = perf_counter() - inference_started_at

    # 分数已降序；一旦低于阈值，后续证据也无需加入最终列表。
    ranked = sorted(
        zip(candidates, scores, strict=True),
        key=lambda item: float(item[1]),
        reverse=True,
    )

    # 统计重排序后的输出证据列表：
    evidences: list[Evidence] = []
    for original, score in ranked:
        if float(score) < min_score or len(evidences) == top_k:
            break
        evidence = original.model_copy(
            update={"citation_id": f"E{len(evidences) + 1}", "score": float(score)}
        )
        evidences.append(evidence)

    # 日志：
    logger.info(
        "Reranking completed: candidates=%d evidence=%d load=%.2fs inference=%.2fs total=%.2fs",
        len(candidates),
        len(evidences),
        model_load_seconds,
        inference_seconds,
        perf_counter() - started_at,
    )
    return RetrievalResult(query=normalized_query, found=bool(evidences), evidences=evidences)


# 这里将’完整检索流程‘封装成一个入口，提供给我们的’检索tool‘：
# （所以调用者不需要分别理解 Dense、BM25 和 RRF，只管调用检索，获取检索即可）
def retrieve_with_rerank(query: str, *, user_id: UUID | None = None) -> RetrievalResult:
    """执行混合召回后，再返回 Cross-Encoder 最终证据。"""
    candidates = retrieve_hybrid(query, user_id=user_id)
    return rerank(query, candidates.evidences)
