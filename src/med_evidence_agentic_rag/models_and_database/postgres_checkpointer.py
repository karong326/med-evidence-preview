"""管理 LangGraph PostgreSQL Checkpointer 的独立连接生命周期。

业务表经 SQLAlchemy/asyncpg 访问，而 LangGraph 使用 psycopg；本模块将同一 PostgreSQL URL
转换为 Checkpointer 可识别的形式，幂等执行其建表并以异步上下文管理器交给 FastAPI 生命周期。
它保存 Agent 的会话状态，不保存可展示的 ChatTurn 业务记录。
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from med_evidence_agentic_rag.utils.config import postgresql_url


@asynccontextmanager
async def open_checkpointer() -> AsyncGenerator[AsyncPostgresSaver]:
    """打开 Checkpointer、幂等建表，并在应用停止时关闭连接。"""
    # 业务表的 SQLAlchemy 使用 asyncpg；LangGraph Checkpointer 使用 psycopg。
    # 两者共用同一 PostgreSQL，只需把连接地址的驱动前缀改为 psycopg 可识别的形式。
    psycopg_url = postgresql_url.replace("postgresql+asyncpg://", "postgresql://", 1)
    async with AsyncPostgresSaver.from_conn_string(psycopg_url) as saver:
        await saver.setup()
        yield saver

# 这里是因为项目中存在的“两类 PostgreSQL 数据”：
# SQLAlchemy 业务表
# ├── users
# ├── conversations
# ├── chat_turns
# └── documents
#
# LangGraph Checkpointer (状态)表 --（就是这里）
# ├── checkpoints
# ├── checkpoint_blobs
# ├── checkpoint_writes
# └── checkpoint_migrations
#
# 两者使用同一个 PostgreSQL，但职责不同：
# - SQLAlchemy 表保存前端可以读取的业务状态；
# - Checkpointer 保存 Agent 内部的消息、节点和运行上下文。
