"""加载项目共享的 Cross-Encoder 重排序模型。

该模块只提供模型对象，实际候选构造、分数阈值和 Top-K 截断由 rag_retrieval.reranker 实现。
服务启动预热此对象以避免首个医学问题承担模型加载时间。
"""

from sentence_transformers import CrossEncoder

from med_evidence_agentic_rag.models_and_database.inference import Float32Sigmoid
from med_evidence_agentic_rag.utils.config import reranker_model, reranker_precision
from med_evidence_agentic_rag.utils.logger import setup_logger

logger = setup_logger(__name__)

reranker = CrossEncoder(
    str(reranker_model),
    device="cuda",
    local_files_only=True,
    max_length=512,
    model_kwargs={"torch_dtype": reranker_precision},
    activation_fn=Float32Sigmoid(),
)

logger.info("Cross-Encoder 重排序模型已加载：%s", reranker_model)
