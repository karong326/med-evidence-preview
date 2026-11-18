"""MedEvidence Agentic RAG 的应用源码包。

本包承载从 HTTP 接口、Agent 编排、资料入库和检索，到 PostgreSQL、Redis、
Chroma 与本地模型连接的完整服务实现。运行入口位于 ``main``，业务代码
按 API、持久化、工具、服务和 RAG 两阶段分目录组织；各子包不在此处初始化外部连接。
"""
