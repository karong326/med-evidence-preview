"""RAG 检索阶段：召回、融合、重排序与证据数据结构。

该包从 Chroma 和 BM25 获取用户可见候选，以 RRF 合并，再用 Cross-Encoder 输出最终带引用编号的
Evidence。它不负责文件解析或 Chroma 写入；资料创建与缓存失效由 rag_creation 包处理。
"""
