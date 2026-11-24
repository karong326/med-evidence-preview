"""加载项目统一使用的本地文本嵌入模型。

该对象由 Chroma 在写入文档和 Dense 检索时复用。模型路径来自运行配置；导入失败应在服务
启动检查或独立自检时暴露，而不是由每次检索重复加载。
"""

import torch
from langchain_huggingface import HuggingFaceEmbeddings

from med_evidence_agentic_rag.utils.config import embedding_model, embedding_model_path
from med_evidence_agentic_rag.utils.logger import setup_logger

logger = setup_logger(__name__)

# 只加载本地 GPU 模型，并归一化文档向量和查询向量。
if not embedding_model_path or not embedding_model.is_dir():
    raise RuntimeError("Required local embedding model directory is missing")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required by the embedding model")
embeddings = HuggingFaceEmbeddings(
    model_name=str(embedding_model),
    model_kwargs={"device": "cuda", "local_files_only": True},
    encode_kwargs={"normalize_embeddings": True},
    query_encode_kwargs={"normalize_embeddings": True},
    show_progress=False,
)
logger.info("Local embedding model loaded")


if __name__ == "__main__":
    vector = embeddings.embed_query("Embedding 模型测试")
    logger.info("嵌入模型加载成功，向量维度：%d", len(vector))
