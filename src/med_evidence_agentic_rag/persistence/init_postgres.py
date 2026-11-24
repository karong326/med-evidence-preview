"""Validate business schema on startup; explicit initialization also sets up checkpoints."""

import asyncio
import sys

from med_evidence_agentic_rag.models_and_database.postgres import engine
from med_evidence_agentic_rag.models_and_database.postgres_checkpointer import open_checkpointer
from med_evidence_agentic_rag.persistence.migrations import migrate
from med_evidence_agentic_rag.utils.logger import setup_logger

logger = setup_logger(__name__)


async def create_application_tables() -> None:
    """验证业务表已迁移到当前版本。"""
    async with engine.begin() as connection:
        await connection.run_sync(migrate, "check")


async def initialize_postgres() -> None:
    """依次创建业务表和 LangGraph Checkpointer 内部表。"""
    async with engine.begin() as connection:
        await connection.run_sync(migrate, "upgrade")
    async with open_checkpointer():
        pass


def main() -> None:
    """在 Windows 使用 psycopg 支持的 SelectorEventLoop 执行初始化。"""
    if sys.platform == "win32":
        with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop) as runner:
            runner.run(initialize_postgres())
    else:
        asyncio.run(initialize_postgres())
    logger.info("PostgreSQL 业务表与 Agent 上下文表初始化完成")


if __name__ == "__main__":
    main()
