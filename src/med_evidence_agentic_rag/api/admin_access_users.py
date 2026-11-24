"""
6. 管理员删除访问用户：
管理员管理访问令牌用户及其全部关联数据的路由。

此处只面向 ``auth_type=access_token`` 的匿名访问用户，不影响邮箱用户。列表以一次聚合 SQL
返回会话数量；删除会依次复用资料清理服务、清除 LangGraph checkpoint 表中的 thread 数据、
删除用户及其剩余上传目录。批量清空逐个复用相同删除链，避免两套清理逻辑漂移。
"""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select

from med_evidence_agentic_rag.api.auth import AuthenticatedUser, require_admin
from med_evidence_agentic_rag.api.documents import UPLOAD_DIRECTORY
from med_evidence_agentic_rag.models_and_database.postgres import AsyncSessionLocal
from med_evidence_agentic_rag.models_and_database.postgres_checkpointer import open_checkpointer
from med_evidence_agentic_rag.persistence.operations import journal
from med_evidence_agentic_rag.persistence.postgres_tables import (
    Conversation,
    User,
)
from med_evidence_agentic_rag.rag_creation.ingestion import delete_indexed_document
from med_evidence_agentic_rag.services.coordination import resources, run_protected
from med_evidence_agentic_rag.services.recovery import remove_account
from med_evidence_agentic_rag.utils.logger import setup_logger


def delete_vectors(selector: tuple[str, str]) -> None:
    delete_indexed_document(selector[1])


logger = setup_logger(__name__)
admin_access_users_router = APIRouter(prefix="/api/admin", tags=["管理员"])


# _delete_access_user() 清理顺序：
#   找到访问令牌用户
#   → 获取其会话和资料
#   → 复用统一资料删除服务
#   → 删除 LangGraph Checkpointer 数据
#   → 删除 User
#   → 数据库级联删除 Conversation 和 ChatTurn
#   → 清理残留上传目录
async def _delete_access_user(user_id: UUID) -> bool:
    """删除一位访问令牌用户的会话、检查点、上传资料和账号。"""
    lease = await resources.obtain(("owner-delete", str(user_id)))

    async def erase() -> bool:
        async with AsyncSessionLocal() as session:
            exists = await session.scalar(
                select(User.id).where(User.id == user_id, User.auth_type == "access_token")
            )
        if not exists:
            return False
        op = await journal.begin("owner-delete", user_id, None, "")
        async with open_checkpointer() as saver:
            await remove_account(
                user_id, UPLOAD_DIRECTORY, journal, delete_vectors, saver.adelete_thread
            )
        await journal.finish(op.id)
        return True

    return await run_protected(lease, erase)


@admin_access_users_router.get("/access-users")
async def list_access_users(
    _: AuthenticatedUser = Depends(require_admin),
) -> list[dict[str, object]]:
    """列出仅通过访问令牌创建的普通用户及会话数量。"""
    async with AsyncSessionLocal() as session:
        rows = (
            await session.execute(
                select(User, func.count(Conversation.id))
                .outerjoin(Conversation, Conversation.user_id == User.id)
                .where(User.auth_type == "access_token")
                .group_by(User.id)
                .order_by(User.last_login_at.desc())
            )
        ).all()
        return [
            {
                "id": str(user.id),
                "created_at": user.created_at,
                "last_login_at": user.last_login_at,
                "conversation_count": count,
            }
            for user, count in rows
        ]


@admin_access_users_router.delete("/access-users/{user_id}")
async def delete_access_user(
    user_id: UUID, _: AuthenticatedUser = Depends(require_admin)
) -> dict[str, bool]:
    """干净删除指定访问令牌用户的全部项目数据。"""
    if not await _delete_access_user(user_id):
        raise HTTPException(status_code=404, detail="访问令牌用户不存在")
    logger.info("Administrator deleted an access-code user")
    return {"deleted": True}


@admin_access_users_router.delete("/access-users")
async def clear_access_users(_: AuthenticatedUser = Depends(require_admin)) -> dict[str, int]:
    """逐个清理全部访问令牌用户，保留邮箱用户。"""
    async with AsyncSessionLocal() as session:
        user_ids = list(
            await session.scalars(select(User.id).where(User.auth_type == "access_token"))
        )
    for user_id in user_ids:
        await _delete_access_user(user_id)
    logger.info("管理员已清空访问令牌用户：数量=%d", len(user_ids))
    return {"deleted": len(user_ids)}
