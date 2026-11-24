"""提供邮箱验证码、访问令牌登录与 JWT 身份依赖。

流程为：Redis 生成并一次性消费图片人机验证凭据；用户通过后请求邮件验证码，验证码与
一分钟冷却均保存在 Redis；验证成功后创建或恢复 PostgreSQL 用户并签发 24 小时 JWT。
``require_user`` 和 ``require_admin`` 是其余路由的统一权限边界，避免业务接口自行解析令牌。
"""

import asyncio
import hmac
import secrets
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import cast
from uuid import UUID

import jwt
from email_validator import EmailNotValidError, validate_email
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, field_validator

from med_evidence_agentic_rag.models_and_database.redis import (
    clear_email_code_cooldown,
    consume_human_verification,
    create_human_challenge,
    delete_verification_code,
    save_verification_code,
    start_email_code_cooldown,
    verify_email_code,
    verify_human_challenge,
)
from med_evidence_agentic_rag.persistence.conversations import (
    get_or_create_access_user,
    get_or_create_user,
)
from med_evidence_agentic_rag.tools.send_email import (
    DeliveryUncertain,
    EmailRecipientError,
    send_email,
)
from med_evidence_agentic_rag.utils.config import (
    access_token_code,
    admin_access_code,
    jwt_algorithm,
    jwt_secret_key,
)
from med_evidence_agentic_rag.utils.logger import setup_logger

logger = setup_logger(__name__)
auth_router = APIRouter(prefix="/api/auth", tags=["认证"])
bearer_scheme = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class AuthenticatedUser:
    """JWT 中恢复出的当前登录用户。"""

    id: UUID
    email: str
    role: str = "user"


class EmailRequest(BaseModel):
    """发送验证码所需的原始邮箱请求体。"""

    email: str

    @field_validator("email")
    @classmethod
    def validate_original_email(cls, value: str) -> str:
        """只验证邮箱格式，不使用验证库返回的规范化结果。"""
        try:
            validate_email(value, check_deliverability=False)
        except EmailNotValidError as exc:
            raise ValueError("邮箱格式不正确") from exc
        return value


class EmailLoginRequest(EmailRequest):
    """邮箱登录请求体，在邮箱基础上增加用户输入的验证码。"""

    code: str


class HumanTokenRequest(BaseModel):
    """需要先通过人机校验的请求体基础字段。"""

    human_token: str = Field(min_length=1, max_length=200)


class EmailCodeRequest(EmailRequest, HumanTokenRequest):
    """发送邮箱验证码时携带的一次性人机验证凭据。"""


class HumanCheckRequest(BaseModel):
    """提交人机题编号与用户答案。"""

    challenge_id: str = Field(min_length=1, max_length=200)
    answer: str = Field(min_length=1, max_length=20)


class AccessTokenLoginRequest(HumanTokenRequest):
    """访问令牌登录时携带浏览器标识，保证同一浏览器复用同一用户。"""

    access_code: str = Field(min_length=1, max_length=200)
    visitor_id: str = Field(min_length=16, max_length=200)


def generate_verification_code() -> str:
    """生成安全的六位数字验证码。"""
    return f"{secrets.randbelow(1_000_000):06d}"


def create_access_token(user_id: UUID, email: str, role: str = "user") -> str:
    """为已验证身份签发包含用户编号、邮箱和角色的 JWT。"""
    if not jwt_secret_key:
        raise RuntimeError("JWT_SECRET_KEY 未配置")
    payload = {
        "sub": str(user_id),
        "email": email,
        "role": role,
        "exp": datetime.now(UTC) + timedelta(hours=24),
    }
    return jwt.encode(payload, jwt_secret_key, algorithm=jwt_algorithm)  # type: ignore[reportUnknownMemberType]


def require_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> AuthenticatedUser:
    """校验 Bearer JWT，并返回用户编号与原始邮箱。"""
    unauthorized = HTTPException(
        status_code=401,
        detail="您目前的状态：未登录或登录已过期",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if credentials is None or credentials.scheme != "Bearer":
        raise unauthorized
    if not jwt_secret_key:
        raise RuntimeError("JWT_SECRET_KEY 未配置")

    try:
        # PyJWT's optional cryptography key types are incomplete in its annotations.
        # Keep its decoded JSON at an object boundary and validate claims below.
        decode = cast(Callable[..., dict[str, object]], getattr(jwt, "decode"))
        payload = decode(
            credentials.credentials,
            jwt_secret_key,
            algorithms=[jwt_algorithm],
        )
        user_id = UUID(str(payload.get("sub", "")))
    except (jwt.InvalidTokenError, ValueError) as exc:
        raise unauthorized from exc

    email = payload.get("email")
    if not isinstance(email, str) or not email:
        raise unauthorized
    role = payload.get("role", "user")
    if not isinstance(role, str) or role not in {"user", "admin"}:
        raise unauthorized
    return AuthenticatedUser(id=user_id, email=email, role=role)


def require_admin(current_user: AuthenticatedUser = Depends(require_user)) -> AuthenticatedUser:
    """只允许管理员 JWT 调用后台管理接口。"""
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="需要管理员权限")
    return current_user


@auth_router.get("/captcha")
async def get_human_challenge() -> dict[str, str]:
    """返回一张图片验证码；正确答案只保存在 Redis。"""
    return await create_human_challenge()


@auth_router.post("/human-check")
async def check_human(request: HumanCheckRequest) -> dict[str, str]:
    """答对人机题后返回短期、一次性可消费的验证凭据。"""
    human_token = await verify_human_challenge(request.challenge_id, request.answer)
    if human_token is None:
        raise HTTPException(status_code=400, detail="人机验证答案不正确或已过期")
    return {"human_token": human_token}


@auth_router.post("/email-code")
async def send_email_code(request: EmailCodeRequest) -> dict[str, str]:
    """生成验证码、写入 Redis，并在线程中发送邮件。"""
    email = request.email
    if not await consume_human_verification(request.human_token):
        raise HTTPException(status_code=400, detail="请先完成人机验证")
    if not await start_email_code_cooldown(email):
        raise HTTPException(status_code=429, detail="验证码已发送，请 1 分钟后再试")
    code = generate_verification_code()

    # 先写入 Redis；邮件失败时立即删除，避免留下无法使用的验证码。
    await save_verification_code(email, code)
    content = (
        f"您好！您的 MedEvidence-Agent app 登录验证码是：\n\n{code}\n\n"
        "验证码两分钟内有效，请勿泄露给他人。"
    )
    # SMTP 是同步阻塞调用，放入工作线程避免阻塞 FastAPI 事件循环。
    try:
        sent = await asyncio.to_thread(send_email, email, "MedEvidence 登录验证码", content)
    except EmailRecipientError as exc:
        await delete_verification_code(email)
        await clear_email_code_cooldown(email)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except DeliveryUncertain:
        sent = False
    if not sent:
        await delete_verification_code(email)
        await clear_email_code_cooldown(email)
        raise HTTPException(status_code=502, detail="验证码邮件发送失败")
    logger.info("登录验证码邮件已发送")
    return {"message": "验证码已发送"}


@auth_router.post("/login/email")
async def login_with_email(request: EmailLoginRequest) -> dict[str, str]:
    """校验验证码，创建或获取用户并签发 JWT。"""
    email = request.email
    result = await verify_email_code(email, request.code)
    if result == "exhausted":
        raise HTTPException(status_code=429, detail="验证码尝试次数过多，请重新获取验证码")
    if result == "missing":
        raise HTTPException(status_code=400, detail="验证码已过期或不存在")
    if result != "success":
        raise HTTPException(status_code=400, detail="验证码不正确")

    user = await get_or_create_user(email)
    token = create_access_token(user.id, user.email)
    logger.info("Email sign-in succeeded")
    return {"access_token": token, "token_type": "bearer"}


@auth_router.post("/login/access-token")
async def login_with_access_token(request: AccessTokenLoginRequest) -> dict[str, str]:
    """通过人机校验后，用普通或管理员访问令牌进入对应页面。"""
    if not access_token_code or not admin_access_code:
        raise HTTPException(status_code=503, detail="访问令牌未配置")
    if not await consume_human_verification(request.human_token):
        raise HTTPException(status_code=400, detail="请先完成人机验证")

    if hmac.compare_digest(request.access_code, admin_access_code):
        admin_id = UUID("00000000-0000-0000-0000-000000005145")
        token = create_access_token(admin_id, "admin@local", "admin")
        logger.info("管理员访问令牌登录成功")
        return {"access_token": token, "token_type": "bearer", "role": "admin"}
    if not hmac.compare_digest(request.access_code, access_token_code):
        raise HTTPException(status_code=401, detail="访问令牌不正确")

    user = await get_or_create_access_user(request.visitor_id)
    token = create_access_token(user.id, user.email)
    logger.info("Access-code sign-in succeeded")
    return {"access_token": token, "token_type": "bearer", "role": "user"}
