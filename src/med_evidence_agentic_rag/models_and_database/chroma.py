"""创建全项目共享的本地 Chroma 向量库实例。

实例使用配置中的持久化目录、集合名和 Embedding 函数。资料创建阶段向其中写入 chunks，
检索与删除阶段按 metadata 查询或清理；本模块不决定文本切分、可见性和排序策略。
"""

from langchain_chroma import Chroma

from med_evidence_agentic_rag.models_and_database.embedding import embeddings
from med_evidence_agentic_rag.utils.config import chroma_collection_name, chroma_persist_directory
from med_evidence_agentic_rag.utils.logger import setup_logger

logger = setup_logger(__name__)

# 全项目复用同一个本地持久化向量库实例。
vectorstore = Chroma(
    collection_name=chroma_collection_name,
    embedding_function=embeddings,
    persist_directory=str(chroma_persist_directory),
)
logger.info("向量数据库已加载：%s", chroma_persist_directory)


if __name__ == "__main__":
    result = vectorstore.get(limit=1)
    logger.info("向量数据库连接成功，当前测试读取数量：%d", len(result.get("ids", [])))
