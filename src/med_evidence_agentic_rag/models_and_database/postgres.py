"""创建 PostgreSQL 的 SQLAlchemy 异步 Engine 与 Session 工厂。

业务持久化模块每次操作从 ``AsyncSessionLocal`` 获取独立 Session，避免跨请求共享事务；
Engine 开启失效连接预检。这里只提供连接资源和最小连通性自检，建表迁移在 persistence 层完成。
"""

import asyncio

from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from med_evidence_agentic_rag.utils.config import postgresql_url
from med_evidence_agentic_rag.utils.logger import setup_logger

logger = setup_logger(__name__)

if not postgresql_url:
    raise ValueError("POSTGRESQL_URL 未配置")

# Engine 管理异步连接池，真正执行 SQL 时才获取连接。
engine = create_async_engine(
    postgresql_url,
    pool_pre_ping=True,
    echo=False,
)

# 业务代码每次通过工厂创建独立 Session，不共享全局 Session。
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    expire_on_commit=False,
)


async def test_connection() -> None:
    """执行最小 SQL，验证 PostgreSQL 连通性。"""

    async with engine.connect() as connection:
        await connection.execute(text("SELECT 1"))

    logger.info("关系型数据库连接成功")
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(test_connection())
