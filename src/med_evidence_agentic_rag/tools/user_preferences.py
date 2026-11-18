"""
5. 用户偏好与长期资料：
让 Agent 在明确规则下维护当前用户的长期信息。

``manage_user_preference`` 保存或清空长期回答偏好；``manage_user_memory`` 分别保存用户称呼
或联系人别名与邮箱。两个 Tool 都从 runtime context 获取用户编号，校验长度和格式，并返回
稳定 JSON 状态供 Agent 与执行记录使用，不允许模型指定其他用户。
"""

import json
from typing import Any, Literal
from uuid import UUID

from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from med_evidence_agentic_rag.agent.context import UserContext
from med_evidence_agentic_rag.persistence.user_preferences import (
    clear_user_preference,
    save_user_contact,
    save_user_name,
    save_user_preference,
)
from med_evidence_agentic_rag.utils.logger import setup_logger

logger = setup_logger(__name__)


# 全部保存在 users 表：
#   preference
#   → 回答风格等稳定偏好
#
#   long_term_memory
#   ├── display_name
#   └── contacts
# get_user_personalization() 一次数据库查询同时返回偏好和长期资料。
def _user_input_error(action: str, message: str) -> str:
    """把用户可自行修正的 Tool 参数错误返回给 Agent，而不是中断整轮。"""
    return json.dumps({"success": False, "action": action, "message": message}, ensure_ascii=False)


@tool
async def manage_user_preference(
    action: Literal["save", "clear"],
    runtime: ToolRuntime[UserContext, dict[str, Any]],
    preference: str | None = None,
) -> str:
    """保存或清空用户稳定、非敏感的长期偏好与资料。

    Args:
        action: save 保存偏好；clear 清空当前用户已有偏好。
        preference: action=save 时必填的稳定、非敏感偏好或资料；clear 时忽略。
        runtime: 系统注入的当前登录用户上下文，不由模型填写。
    Returns:
        JSON 字符串，包含 success、action 与 message。
    Raises:
        ValueError: 当前 Tool 缺少登录用户。
    Side effects:
        仅写入或删除当前登录用户的 PostgreSQL 长期偏好记录。
    """
    raw_user_id = runtime.context.get("user_id", "")
    if not raw_user_id:
        raise ValueError("当前 Tool 缺少登录用户")
    user_id = UUID(raw_user_id)

    # clear    清空偏好：
    if action == "clear":
        await clear_user_preference(user_id)
        logger.info("用户长期回答偏好已清空：用户=%s", user_id)
        return json.dumps(
            {"success": True, "action": "clear", "message": "长期回答偏好已清空"},
            ensure_ascii=False,
        )

    # save     保存完整偏好：
    value = (preference or "").strip()
    if not value:
        return _user_input_error(action, "长期偏好不能为空，请重新提供。")
    if len(value) > 500:
        return _user_input_error(action, "长期偏好不能超过 500 个字符，请缩短后重试。")

    await save_user_preference(user_id, value)
    logger.info("用户长期回答偏好已保存：用户=%s", user_id)
    return json.dumps(
        {"success": True, "action": "save", "message": "长期回答偏好已保存"},
        ensure_ascii=False,
    )


@tool
async def manage_user_memory(
    action: Literal["save_name", "save_contact"],
    runtime: ToolRuntime[UserContext, dict[str, Any]],
    name: str | None = None,
    contact_name: str | None = None,
    email: str | None = None,
) -> str:
    """仅当用户明确要求记住本人称呼或联系人邮箱时调用。

    用户说“我叫卡戎”“我是卡戎”“称呼我卡戎”时使用 save_name；
    用户说“记住 L老师的邮箱是 teacher@example.com”时使用 save_contact。
    不得自动保存病例、患者、医生或对话中偶然出现的人名、邮箱和隐私信息。
    """
    raw_user_id = runtime.context.get("user_id", "")
    if not raw_user_id:
        raise ValueError("当前 Tool 缺少登录用户")
    user_id = UUID(raw_user_id)
    if action == "save_name":
        value = (name or "").strip()
        if not value or len(value) > 50:
            return _user_input_error(action, "用户称呼不能为空且不能超过 50 个字符，请重新提供。")
        await save_user_name(user_id, value)
        logger.info("用户称呼已保存：用户=%s", user_id)
        return json.dumps(
            {"success": True, "action": action, "message": "用户称呼已保存"},
            ensure_ascii=False,
        )

    alias = (contact_name or "").strip()
    address = (email or "").strip()
    if not alias or len(alias) > 50 or "@" not in address or len(address) > 320:
        return _user_input_error(action, "联系人别名或邮箱格式不正确，请重新提供。")
    await save_user_contact(user_id, alias, address)
    logger.info("用户联系人已保存：用户=%s，别名=%s", user_id, alias)
    return json.dumps(
        {"success": True, "action": action, "message": "用户联系人已保存"},
        ensure_ascii=False,
    )
