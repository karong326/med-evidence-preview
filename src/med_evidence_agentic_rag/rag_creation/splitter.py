"""按统一规则将加载后的 Document 切分为可检索文本块。

切分器保留原始 Document metadata，并为每个块追加连续 ``chunk_index``，供 ingestion 生成稳定
Chroma ID 和引用定位。该模块不读取磁盘、不调用模型，也不决定向量集合或检索 Top-K。
"""

from collections.abc import Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from med_evidence_agentic_rag.utils.logger import setup_logger

logger = setup_logger(__name__)

# 切分器配置：优先按自然段和句子边界递归切分，最后才按字符兜底。
text_splitter = RecursiveCharacterTextSplitter(
    # 优先顺序： 自然段 → 换行 → 中文句号 → 感叹号 → 问号 → 空格 → 最后才按字符硬切
    separators=["\n\n", "\n", "。", "！", "？", " ", ""],
    chunk_size=512,
    chunk_overlap=60,
)


def split_documents(documents: Sequence[Document]) -> list[Document]:
    """切分 Document，并为每个 chunk 标记稳定的顺序号。"""

    chunks = text_splitter.split_documents(list(documents))

    # 顺序号用于生成稳定 chunk_id，并保留原始 Document 元数据。
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index

    logger.info("文档切分完成：输入文档数=%d，切块数=%d", len(documents), len(chunks))
    return chunks
