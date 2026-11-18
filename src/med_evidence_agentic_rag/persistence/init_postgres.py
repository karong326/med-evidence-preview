"""初始化当前 PostgreSQL 业务表与 LangGraph Checkpointer 表。

业务表完全以 ``postgres_tables.Base`` 的当前 ORM 定义为准；``create_application_tables`` 只对
新数据库幂等创建缺失表，不再保留或迁移任何历史表、列和旧版本数据。LangGraph 自己的内部表
由 ``open_checkpointer`` 创建，向量库与文件系统不在本模块管理。
"""

import asyncio
import sys

from med_evidence_agentic_rag.models_and_database.postgres import engine
from med_evidence_agentic_rag.models_and_database.postgres_checkpointer import open_checkpointer
from med_evidence_agentic_rag.persistence.postgres_tables import Base
from med_evidence_agentic_rag.utils.logger import setup_logger

logger = setup_logger(__name__)


async def create_application_tables() -> None:
    """按当前 ORM 定义幂等创建业务表。"""
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)


async def initialize_postgres() -> None:
    """依次创建业务表和 LangGraph Checkpointer 内部表。"""
    await create_application_tables()
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
