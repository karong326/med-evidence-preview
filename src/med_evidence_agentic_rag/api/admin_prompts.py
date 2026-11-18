"""管理员读取、更新和即时应用三段 Agent 提示词的路由。

接口提供主系统提示词、查询规划提示词和 Todo 提示词，并以 ``updated_at`` 拒绝两个管理员页面
之间的静默覆盖。保存成功后重新创建应用状态中的 Agent，使新系统与 Todo 提示词立即生效；
旧版本归档和并发判断由 prompts.storage 负责。
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from med_evidence_agentic_rag.api.auth import AuthenticatedUser, require_admin
from med_evidence_agentic_rag.prompts.query_rewrite_prompt import QUERY_REWRITE_PROMPT
from med_evidence_agentic_rag.prompts.storage import (
    PromptVersionConflictError,
    load_prompts,
    save_prompts,
)
from med_evidence_agentic_rag.prompts.system_prompt import SYSTEM_PROMPT
from med_evidence_agentic_rag.prompts.todo_planning_prompt import TODO_PLANNING_PROMPT
from med_evidence_agentic_rag.utils.logger import setup_logger

logger = setup_logger(__name__)
admin_prompts_router = APIRouter(prefix="/api/admin", tags=["管理员"])


class PromptRequest(BaseModel):
    """管理员保存完整提示词及页面读取时的版本。"""

    system_prompt: str = Field(min_length=1)
    query_rewrite_prompt: str = Field(min_length=1)
    todo_prompt: str = Field(min_length=1)
    updated_at: str = ""


@admin_prompts_router.get("/prompts")
async def get_prompts(_: AuthenticatedUser = Depends(require_admin)) -> dict[str, str]:
    """读取当前生效或默认的系统、查询与待办提示词。"""
    saved = load_prompts()
    return {
        "system_prompt": saved["system_prompt"] or SYSTEM_PROMPT,
        "query_rewrite_prompt": saved["query_rewrite_prompt"] or QUERY_REWRITE_PROMPT,
        "todo_prompt": saved["todo_prompt"] or TODO_PLANNING_PROMPT,
        "updated_at": saved["updated_at"],
    }


@admin_prompts_router.put("/prompts")
async def update_prompts(
    request_body: PromptRequest,
    request: Request,
    _: AuthenticatedUser = Depends(require_admin),
) -> dict[str, bool | str]:
    """保存提示词；管理员页面过期时拒绝覆盖并要求重新加载。"""
    saved = load_prompts()
    try:
        changed = save_prompts(
            request_body.system_prompt,
            request_body.query_rewrite_prompt,
            request_body.todo_prompt,
            previous_prompts={
                "system_prompt": saved["system_prompt"] or SYSTEM_PROMPT,
                "query_rewrite_prompt": saved["query_rewrite_prompt"] or QUERY_REWRITE_PROMPT,
                "todo_prompt": saved["todo_prompt"] or TODO_PLANNING_PROMPT,
            },
            expected_updated_at=request_body.updated_at,
        )
    except PromptVersionConflictError as exc:
        raise HTTPException(
            status_code=409,
            detail="提示词已被其他管理员更新，请重新加载后再保存",
        ) from exc
    if changed:
        from med_evidence_agentic_rag.agent.agent import create_chat_agent

        request.app.state.chat_agent = create_chat_agent(request.app.state.checkpointer)
        logger.info("管理员已更新 Agent 提示词")
    return {
        "saved": changed,
        "updated_at": load_prompts()["updated_at"] if changed else saved["updated_at"],
    }
