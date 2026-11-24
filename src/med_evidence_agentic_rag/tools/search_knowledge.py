"""封装唯一的 ``search_knowledge`` 知识库检索 Tool。

工具从 runtime context 取得当前用户与本轮编号，执行混合召回和 Cross-Encoder 重排序，并把
证据转换为带 ``轮次-子问题.序号`` 引用的 JSON。它只返回当前用户可见的共享/私有证据；是否
允许调用及每轮次数分别由查询计划守卫和次数限制中间件决定。
"""

import json
from typing import Any, cast
from uuid import UUID

from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from med_evidence_agentic_rag.agent.context import UserContext
from med_evidence_agentic_rag.rag_retrieval.models import Evidence, RetrievalResult
from med_evidence_agentic_rag.rag_retrieval.reranker import retrieve_with_rerank as retrieve
from med_evidence_agentic_rag.services.coordination import run_stream_worker
from med_evidence_agentic_rag.utils.logger import setup_logger

logger = setup_logger(__name__)


def _current_query_index(runtime: ToolRuntime[UserContext, dict[str, Any]]) -> int:
    """根据本轮已有的知识库结果确定当前查询字母。"""
    # 它从当前 LangGraph 消息状态倒序检查：
    # 从末尾向前统计本轮检索结果，遇到 HumanMessage 停止。
    # 结果用于生成查询字母：
    #   一个问题的第 1 次检索 → A； 第 2 次检索 → B； 第 3 次检索 → C...
    state = cast(dict[str, Any], getattr(runtime, "state", {}))
    messages = cast(list[object], state.get("messages", []))
    query_index = 0
    for message in reversed(messages):
        if getattr(message, "type", "") == "human":
            break
        if getattr(message, "name", None) == "search_knowledge":
            query_index += 1
    if query_index >= 6:
        # 超过五次时再次拒绝，这里与中间件 ToolCallLimitMiddleware 形成双层保护
        raise ValueError("每轮最多执行六次知识库检索")
    return query_index


# Agent 的唯一 RAG Tool：
@tool
async def search_knowledge(
    query: str,
    runtime: ToolRuntime[UserContext, dict[str, Any]],
) -> str:
    """检索一个明确问题，返回最多三条可引用的原文证据。

    Args:
        query: 单个、可独立检索的医学或资料问题。
        runtime: 系统注入的当前用户与会话上下文，不由模型填写。
    Returns:
        JSON 字符串包含 found、query 与 evidences；每条证据含引用编号、来源、标题、
        以及分数和原文。
    Raises:
        ValueError: query 为空，或同一轮已经检索六次。
    Side effects:
        执行混合检索与重排序，并将命中原文记录到服务端日志；不写入知识库或业务数据库。
    """
    # 从 ToolRuntime 读取：
    raw_user_id = runtime.context.get("user_id", "")
    user_id = UUID(raw_user_id) if raw_user_id else None
    turn_number = int(runtime.context.get("turn_number", 1))
    # 获取问题输入：
    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("query 不能为空")
    query_index = _current_query_index(runtime)

    # 每次只执行一个检索问题，让 Agent 阅读证据后决定是否继续。
    evidences: list[Evidence] = []

    # ！！！进入整个检索流程，最后返回检索结果：
    result = await run_stream_worker(lambda: retrieve(normalized_query, user_id=user_id))
    # 进入retrieve()方法，也就是完整的流程：
    # → Dense Top x
    # → BM25 Top x
    # → RRF Top y
    # → Cross-Encoder
    # → 阈值过滤
    # → 最终 Top z

    for evidence in result.evidences:
        final_evidence = evidence.model_copy(
            update={
                # 配置和生成对应的：{轮次}-{查询字母}.{证据序号}
                #   01   第 1 轮问答或者对话
                #   A    本轮第 1 个计划进行检索的问题
                #   1    该检索查询的第 1 条证据
                "citation_id": (
                    f"{turn_number:02d}-{chr(ord('A') + query_index)}.{len(evidences) + 1}"
                ),
                "query": normalized_query,
            }
        )
        evidences.append(final_evidence)

    # 重新整理后的工具结果：
    result = RetrievalResult(
        query=normalized_query,
        found=bool(evidences),
        evidences=evidences,
    )
    logger.info("Knowledge retrieval completed: evidence=%d", len(evidences))

    # ！！！然后 Tool 最终return返回统一 JSON 的示例如下：
    # {
    #   "query": "医院感染暴发报告时限",
    #   "found": true,
    #   "evidences": [
    #     {
    #       "citation_id": "01-A.1",
    #       "score": 0.87,
    #       "source": "......",
    #       "title": "......",
    #       "content": "具体原文内容",
    #       "query": "医院感染暴发报告时限"
    #     }
    #   ]
    # }
    # JSON 保留 citation_id、来源和正文，供 Agent 生成可引用回答。
    return json.dumps(result.model_dump(), ensure_ascii=False)


# 之后主 Agent：
# 读取 Tool 返回的 query
# 读取 found
# 读取 evidences
# 读取 citation_id、title、source、content
# 结合系统提示词
# 生成带引用的自然语言回答
#
# 回答片段由 streaming.py 捕获，并作为：
# {"type":"answer","content":"......"}
# 发送给前端。
