"""外部存储与模型资源的轻量连接层。

本包只创建可复用的 PostgreSQL、Redis、Chroma、Checkpointer、LLM、Embedding 和 Reranker
客户端/模型对象，不包含业务 SQL、检索策略或 API 路由。应用生命周期负责连通性检查和关闭。
"""
