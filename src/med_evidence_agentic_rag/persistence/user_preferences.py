"""读写用户长期偏好、称呼和联系人 JSON 记忆。

偏好存储在 User 的单一文本字段；称呼与联系人别名存储在 ``long_term_memory`` JSON，并在读取
时规范化 contacts 字典。邮件工具通过本模块在服务端解析“我的邮箱”或别名，模型永不直接
读取联系人真实邮箱。
"""

from typing import cast
from uuid import UUID

from med_evidence_agentic_rag.models_and_database.postgres import AsyncSessionLocal
from med_evidence_agentic_rag.persistence.postgres_tables import User


async def save_user_preference(user_id: UUID, preference: str) -> User:
    """新增或覆盖当前用户唯一的一份长期回答偏好。"""
    async with AsyncSessionLocal() as session:
        record = await session.get(User, user_id)
        if record is None:
            raise LookupError("用户不存在")
        record.preference = preference
        await session.commit()
        await session.refresh(record)
        return record


async def clear_user_preference(user_id: UUID) -> None:
    """删除当前用户已经保存的长期回答偏好。"""
    async with AsyncSessionLocal() as session:
        record = await session.get(User, user_id)
        if record is not None:
            record.preference = None
            await session.commit()


def _memory_from_user(user: User) -> dict[str, object]:
    """规范化读取 users 中唯一的长期资料 JSON。"""
    memory = dict(user.long_term_memory or {})
    contacts = memory.get("contacts")
    memory["contacts"] = dict(cast(dict[str, str], contacts)) if isinstance(contacts, dict) else {}
    return memory


async def get_user_personalization(user_id: UUID) -> tuple[str | None, dict[str, object]]:
    """一次读取当前用户的长期偏好及仅供模型展示的长期记忆。"""
    async with AsyncSessionLocal() as session:
        user = await session.get(User, user_id)
        if user is None:
            return None, {"contacts": {}}
        return user.preference, _memory_from_user(user)


async def save_user_name(user_id: UUID, name: str) -> User:
    """保存用户明确提供的称呼。"""
    async with AsyncSessionLocal() as session:
        user = await session.get(User, user_id)
        if user is None:
            raise LookupError("用户不存在")
        memory = _memory_from_user(user)
        memory["display_name"] = name
        user.long_term_memory = memory
        await session.commit()
        await session.refresh(user)
        return user


async def save_user_contact(user_id: UUID, contact_name: str, email: str) -> User:
    """保存用户明确指定的联系人别名与邮箱。"""
    async with AsyncSessionLocal() as session:
        user = await session.get(User, user_id)
        if user is None:
            raise LookupError("用户不存在")
        memory = _memory_from_user(user)
        contacts = cast(dict[str, str], memory["contacts"])
        contacts[contact_name] = email
        user.long_term_memory = memory
        await session.commit()
        await session.refresh(user)
        return user


async def resolve_email_recipient(user_id: UUID, recipient: str) -> str | None:
    """在服务端解析“我的邮箱”或已保存别名，原始邮箱不作转换。"""
    value = recipient.strip()
    if "@" in value:
        return value
    async with AsyncSessionLocal() as session:
        user = await session.get(User, user_id)
        if user is None:
            return None
        if value in {"我的邮箱", "我自己的邮箱", "本人邮箱"}:
            return user.email
        contacts = cast(dict[str, str], _memory_from_user(user)["contacts"])
        return contacts.get(value)
