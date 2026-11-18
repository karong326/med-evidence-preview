"""调用 Dense、BM25 及可选本地多语言召回，以 RRF 融合候选。

``retrieve_hybrid`` 对同一 query 使用相同 user_id 可见性条件，分别取得候选后按来源和正文去重，
累加名次分数并生成连续 Evidence 引用编号。它只负责候选融合，最终相关性阈值与截断交给
Cross-Encoder 重排序模块。
"""

from uuid import UUID

from med_evidence_agentic_rag.rag_retrieval.bm25 import retrieve_bm25
from med_evidence_agentic_rag.rag_retrieval.fusion import fuse_results
from med_evidence_agentic_rag.rag_retrieval.models import RetrievalResult
from med_evidence_agentic_rag.rag_retrieval.multilingual import retrieve_multilingual
from med_evidence_agentic_rag.rag_retrieval.retrieval import retrieve as retrieve_dense
from med_evidence_agentic_rag.utils.config import hybrid_candidate_k, hybrid_top_k, rrf_k
from med_evidence_agentic_rag.utils.logger import setup_logger

logger = setup_logger(__name__)


def retrieve_hybrid(
    query: str,
    *,
    candidate_k: int = hybrid_candidate_k,
    top_k: int = hybrid_top_k,
    rrf_k: int = rrf_k,
    user_id: UUID | None = None,
) -> RetrievalResult:
    """分别执行 Dense 与 BM25，再按名次融合为统一证据。"""
    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("query 不能为空")
    if candidate_k < 1 or top_k < 1 or rrf_k < 1:
        raise ValueError("检索参数必须大于 0")

    # 1，两路各召回 candidate_k 条候选，原始分数不直接互相比较。
    results = [
        # 这里调用两种已经封装好了的检索方法：
        retrieve_dense(normalized_query, top_k=candidate_k, user_id=user_id),
        retrieve_bm25(normalized_query, top_k=candidate_k, user_id=user_id),
    ]

    # 未配置本地多语言模型时返回空列表，兼容原有双路召回。
    results.append(retrieve_multilingual(normalized_query, top_k=candidate_k, user_id=user_id))
    fused = fuse_results(normalized_query, results, top_k=top_k, rrf_k=rrf_k)

    logger.info(
        "[RRF 融合] 问题=%s，Dense=%d，BM25=%d，多语言=%d，融合候选=%d",
        normalized_query,
        len(results[0].evidences),
        len(results[1].evidences),
        len(results[2].evidences),
        len(fused.evidences),
    )
    return fused


if __name__ == "__main__":
    result = retrieve_hybrid("医院感染暴发 12小时")
    logger.info("混合检索独立测试完成：是否命中=%s，证据数=%d", result.found, len(result.evidences))
