"""创建 Agent 与查询规划共用的大语言模型客户端。

客户端读取模型名、API Key 与兼容 OpenAI 的服务地址，但不在导入时发送请求。FastAPI 生命周期
通过 ``/models`` 校验配置可用，Agent 工厂和 rewrite 工具随后复用同一模型对象。
"""

from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from med_evidence_agentic_rag.utils.config import llm_api_key, llm_base_url, llm_model_name
from med_evidence_agentic_rag.utils.logger import setup_logger

logger = setup_logger(__name__)

# 全项目复用同一个支持流式输出的聊天模型实例。
llm = ChatOpenAI(
    model=llm_model_name,
    api_key=SecretStr(llm_api_key),
    base_url=llm_base_url,
    temperature=0.5,
    streaming=True,
)


if __name__ == "__main__":
    response = llm.invoke("你是谁？进行简洁自我介绍。")
    logger.info("大语言模型调用成功：%s", response.content)
