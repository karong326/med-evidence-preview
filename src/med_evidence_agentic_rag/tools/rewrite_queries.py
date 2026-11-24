"""将一个医学问题转换为受限、可执行的查询计划 Tool。

工具先识别用户明确列出的多项目问题，再调用带结构化输出的独立 LLM，生成 ``QueryPlan`` 的模式、
查询列表和回退标记；最终 JSON 由查询计划守卫和执行记录共同使用。它不检索知识库，也不使用
主 Agent 的隐式推理来拼接查询词。
"""

import json
from typing import Any, Literal, cast

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from med_evidence_agentic_rag.models_and_database.llm import llm
from med_evidence_agentic_rag.prompts.query_rewrite_prompt import QUERY_REWRITE_PROMPT
from med_evidence_agentic_rag.prompts.storage import load_prompts
from med_evidence_agentic_rag.utils.logger import setup_logger

logger = setup_logger(__name__)


class QueryPlan(BaseModel):
    """查询规划的可展示结果，但不包含模型隐式思维。"""
    # 三种模式：
    # single： 单一知识需求，只生成一条查询
    # split： 包含多个真正独立的证据需求，最多五条
    # follow_up： 对之前方案的追问，补全指代后优先形成一条完整查询
    mode: Literal["single", "split", "follow_up"]
    queries: list[str] = Field(min_length=1, max_length=6)  # 限制为 1—6 条
    fallback: bool = False


@tool
def rewrite_queries(question: str) -> str:
    """把医学问题规划为最多六条可独立执行的检索查询。"""
    normalized_question = question.strip()
    if not normalized_question:
        raise ValueError("question 不能为空")

    try:
        # 快速独立调用结构化 LLM，得到实际可展示的查询计划。
        typed_llm = cast(Any, llm)
        planner = typed_llm.model_copy(update={"streaming": False}).with_structured_output(
            QueryPlan,
            method="function_calling",
        )
        saved_prompt = load_prompts()["query_rewrite_prompt"]
        active_prompt = saved_prompt or QUERY_REWRITE_PROMPT

        raw_plan: object = planner.invoke(
            active_prompt.format(question=normalized_question)
        )

        plan = raw_plan if isinstance(raw_plan, QueryPlan) else QueryPlan.model_validate(raw_plan)

        queries = list(dict.fromkeys(query.strip() for query in plan.queries if query.strip()))[:6]

        if not queries:
            raise ValueError("查询规划结果为空")
        plan = plan.model_copy(update={"queries": queries, "fallback": False})

    except Exception:
        logger.warning("Query planning failed; using the original question")
        plan = QueryPlan(mode="single", queries=[normalized_question], fallback=True)

    logger.info("Query planning completed: queries=%d", len(plan.queries))
    return json.dumps(plan.model_dump(), ensure_ascii=False)
