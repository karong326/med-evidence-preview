"""MedEvidence Agentic RAG 的统一应用启动入口。

``main`` 解析可选端口（默认 8001），创建 Uvicorn Server 并运行 FastAPI 应用。
FastAPI 同时提供 API 和已构建的 Vue 页面；Windows 下显式使用 Selector 事件循环，
以兼容 psycopg/LangGraph Checkpointer。应用依赖检查、预热与关闭均由 FastAPI lifespan 执行。
"""

import argparse
import asyncio
import sys

import uvicorn

DEFAULT_PORT = 8001


# 命令行
#   ↓
# main.py / main()
#   ↓  解析 --port；未传则使用 8001
# serve(port)
#   ↓
# Uvicorn 加载 api.app:app
#   ↓
# FastAPI 创建 app，并进入 lifespan() ！！！
#   ↓
# 初始化 PostgreSQL、LangGraph Checkpointer、Redis、Chroma
#   ↓
# 校验 LLM 模型、预热 BM25 与 Reranker
#   ↓
# 创建带持久上下文的 Agent，保存到 app.state.chat_agent
#   ↓
# 注册并对外提供 API 路由和 Vue 前端页面

async def serve(port: int) -> None:
    """在指定端口运行完整 FastAPI 应用。"""
    server = uvicorn.Server(
        uvicorn.Config(
            "med_evidence_agentic_rag.api.app:app",
            host="127.0.0.1",
            port=port,
        )
    )
    await server.serve()


def main() -> None:
    """启动统一应用；Windows 为 Psycopg 使用 Selector 事件循环。"""
    parser = argparse.ArgumentParser(description="启动 MedEvidence Agentic RAG 应用")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    try:
        if sys.platform == "win32":
            with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop) as runner:
                runner.run(serve(args.port))
        else:
            asyncio.run(serve(args.port))
    except KeyboardInterrupt:
        # Uvicorn 已完成应用关闭；不把用户主动 Ctrl+C 显示成错误 traceback。
        return


if __name__ == "__main__":
    main()
