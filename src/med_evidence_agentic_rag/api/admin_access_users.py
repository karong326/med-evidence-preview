"""
6. 管理员删除访问用户：
管理员管理访问令牌用户及其全部关联数据的路由。

此处只面向 ``auth_type=access_token`` 的匿名访问用户，不影响邮箱用户。列表以一次聚合 SQL
返回会话数量；删除会依次复用资料清理服务、清除 LangGraph checkpoint 表中的 thread 数据、
删除用户及其剩余上传目录。批量清空逐个复用相同删除链，避免两套清理逻辑漂移。
"""

import shutil
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select, text

from med_evidence_agentic_rag.api.auth import AuthenticatedUser, require_admin
from med_evidence_agentic_rag.api.documents import UPLOAD_DIRECTORY
from med_evidence_agentic_rag.models_and_database.postgres import AsyncSessionLocal
from med_evidence_agentic_rag.persistence.postgres_tables import (
    Conversation,
    DocumentRecord,
    User,
)
from med_evidence_agentic_rag.services.document_deletion import delete_document_assets
from med_evidence_agentic_rag.utils.logger import setup_logger

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
    async with AsyncSessionLocal() as session:
        user = await session.scalar(
            select(User).where(User.id == user_id, User.auth_type == "access_token")
        )
        if user is None:
            return False
        conversation_ids = [
            str(value)
            for value in await session.scalars(
                select(Conversation.id).where(Conversation.user_id == user_id)
            )
        ]
        documents = list(
            await session.scalars(
                select(DocumentRecord).where(DocumentRecord.owner_user_id == user_id)
            )
        )
        for record in documents:
            await delete_document_assets(
                record,
                upload_root=UPLOAD_DIRECTORY / str(user_id),
            )
        for conversation_id in conversation_ids:
            for table in ("checkpoint_writes", "checkpoint_blobs", "checkpoints"):
                await session.execute(
                    text(f"DELETE FROM {table} WHERE thread_id = :thread_id"),
                    {"thread_id": conversation_id},
                )
        await session.delete(user)
        await session.commit()
    upload_directory = (UPLOAD_DIRECTORY / str(user_id)).resolve()
    if upload_directory.is_dir():
        shutil.rmtree(upload_directory)
    return True


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
    logger.info("管理员已删除访问令牌用户：%s", user_id)
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
