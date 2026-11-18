"""创建 MedEvidence Agent，并集中装配模型、工具、提示词与中间件。

``create_chat_agent`` 是唯一的 Agent 工厂：读取管理员当前保存的系统与 Todo 提示词，
注册查询规划、知识检索、邮件和用户记忆工具，并加入 Todo、工具次数和查询计划守卫。
传入 PostgreSQL Checkpointer 时，返回可跨会话恢复的实例；模块级 ``agent`` 仅供独立
调用和轻量测试兜底，FastAPI 正式运行使用应用生命周期创建的持久化实例。
"""

from typing import Any

from langchain.agents import create_agent  # type: ignore[reportUnknownVariableType]
from langgraph.checkpoint.base import BaseCheckpointSaver

from med_evidence_agentic_rag.agent.context import UserContext
from med_evidence_agentic_rag.agent.streaming import logger, stream_chat
from med_evidence_agentic_rag.middlewares.query_plan_guard import QueryPlanGuardMiddleware
from med_evidence_agentic_rag.middlewares.todo_list_middleware import build_todo_list_middleware
from med_evidence_agentic_rag.middlewares.tool_call_limit_middleware import (
    build_tool_call_limit_middlewares,
)
from med_evidence_agentic_rag.models_and_database.llm import llm
from med_evidence_agentic_rag.prompts.storage import load_prompts
from med_evidence_agentic_rag.prompts.system_prompt import SYSTEM_PROMPT
from med_evidence_agentic_rag.prompts.todo_planning_prompt import TODO_PLANNING_PROMPT
from med_evidence_agentic_rag.tools.rewrite_queries import rewrite_queries
from med_evidence_agentic_rag.tools.search_knowledge import search_knowledge
from med_evidence_agentic_rag.tools.send_email import send_email_tool
from med_evidence_agentic_rag.tools.user_preferences import (
    manage_user_memory,
    manage_user_preference,
)

__all__ = ["agent", "create_chat_agent", "logger", "stream_chat"]


# create_chat_agent(checkpointer) 创建正式 Agent
def create_chat_agent(checkpointer: BaseCheckpointSaver[Any] | None = None) -> Any:
    """使用相同模型与 Tool 创建可选持久化的唯一 Agent。"""
    prompts = load_prompts()
    return create_agent(
        llm,
        tools=[
            rewrite_queries,
            search_knowledge,
            send_email_tool,
            manage_user_preference,
            manage_user_memory,
        ],
        system_prompt=prompts.get("system_prompt") or SYSTEM_PROMPT,
        context_schema=UserContext,  # 运行时上下文定义的参数
        checkpointer=checkpointer,
        middleware=[
            build_todo_list_middleware(
                todo_prompt=prompts.get("todo_prompt") or TODO_PLANNING_PROMPT
            ),
            *build_tool_call_limit_middlewares(),
            QueryPlanGuardMiddleware(),
        ],

    )


# 保留无持久化实例供独立运行和轻量测试；服务启动后使用 PostgreSQL 实例。
agent: Any = create_chat_agent()
