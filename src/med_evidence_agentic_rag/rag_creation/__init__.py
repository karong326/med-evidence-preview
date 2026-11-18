"""RAG 知识创建阶段：发现、解析、切分、索引与删除。

该包将共享 ``data/raw`` 与用户 uploads 中的文件转换为带来源、可见性和文档身份 metadata 的
Chroma chunks，并在资料变化后失效 BM25 缓存。它不回答用户问题；检索、融合和重排序位于
rag_retrieval 包。
"""
