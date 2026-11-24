"""提供 Redis 连接及短期认证状态操作。

Redis 保存图片验证码答案、一次性人机验证 token、邮箱验证码和发送冷却时间，均使用 TTL 自动
过期。函数将“读取并删除”的一次性消费语义集中在此处，认证路由只关心流程和 HTTP 响应。
"""

import asyncio
import base64
import secrets
from collections.abc import Awaitable
from typing import cast

import redis.asyncio as redis
from captcha.image import ImageCaptcha

from med_evidence_agentic_rag.utils.config import redis_url
from med_evidence_agentic_rag.utils.logger import setup_logger

logger = setup_logger(__name__)

REDIS_KEY_PREFIX = "email_code:"
EMAIL_CODE_COOLDOWN_PREFIX = "email_code_cooldown:"
HUMAN_CHALLENGE_PREFIX = "human_challenge:"
HUMAN_VERIFICATION_PREFIX = "human_verification:"
# 图片验证码答案的保存时限
HUMAN_CHALLENGE_TTL_SECONDS = 120
# 人机验证通过凭据的时限
HUMAN_VERIFICATION_TTL_SECONDS = 120
# 邮箱登录验证码的保存时间
VERIFICATION_CODE_TTL_SECONDS = 120
# 邮件发送间隔冷却的时间
EMAIL_CODE_COOLDOWN_SECONDS = 60
EMAIL_CODE_MAX_FAILURES = 5
EMAIL_CODE_FAILURE_PREFIX = "email_code_failures:"

# One Redis execution checks the budget, consumes success, or records a failure.
# The failure budget expires with the code, and cannot be reset by a successful guess.
VERIFY_EMAIL_CODE_SCRIPT = """
local expected = redis.call('GET', KEYS[1])
if not expected then return 'missing' end
local failures = tonumber(redis.call('GET', KEYS[2]) or '0')
if failures >= tonumber(ARGV[2]) then return 'exhausted' end
if expected == ARGV[1] then
    redis.call('DEL', KEYS[1], KEYS[2])
    return 'success'
end
local ttl = redis.call('PTTL', KEYS[1])
failures = redis.call('INCR', KEYS[2])
redis.call('PEXPIRE', KEYS[2], math.max(ttl, 1))
if failures >= tonumber(ARGV[2]) then return 'exhausted' end
return 'invalid'
"""

# 全项目复用连接池客户端，具体操作仍按异步请求执行。
redis_client = redis.from_url(
    redis_url,
    decode_responses=True,
)


def _code_key(email: str) -> str:
    """使用原始邮箱生成 Redis Key。"""
    return f"{REDIS_KEY_PREFIX}{email}"


async def create_human_challenge() -> dict[str, str]:
    """生成带字符旋转、错位和扭曲的短期 PNG 图片验证码。"""
    captcha_code = "".join(secrets.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(5))
    captcha = ImageCaptcha(width=180, height=70)
    captcha.character_rotate = (-50, 50)
    captcha.character_offset_dx = (0, 6)
    captcha.character_offset_dy = (0, 10)
    captcha.character_warp_dx = (0.15, 0.35)
    captcha.character_warp_dy = (0.25, 0.4)
    captcha.word_space_probability = 0.2
    image_bytes = captcha.generate(captcha_code)
    challenge_id = secrets.token_urlsafe(16)
    await redis_client.setex(
        f"{HUMAN_CHALLENGE_PREFIX}{challenge_id}",
        HUMAN_CHALLENGE_TTL_SECONDS,
        captcha_code,
    )
    image_data = base64.b64encode(image_bytes.getvalue()).decode("ascii")
    return {
        "challenge_id": challenge_id,
        "image_data_url": f"data:image/png;base64,{image_data}",
    }


async def verify_human_challenge(challenge_id: str, answer: str) -> str | None:
    """核对并作废人机题；答对后签发一次短期 Redis 验证凭据。"""
    challenge_key = f"{HUMAN_CHALLENGE_PREFIX}{challenge_id}"
    expected_answer = await redis_client.getdel(challenge_key)
    if expected_answer != answer.strip():
        return None
    human_token = secrets.token_urlsafe(16)
    await redis_client.setex(
        f"{HUMAN_VERIFICATION_PREFIX}{human_token}",
        HUMAN_VERIFICATION_TTL_SECONDS,
        "verified",
    )
    return human_token


async def consume_human_verification(human_token: str) -> bool:
    """读取并作废人机验证凭据，避免它被脚本重复使用。"""
    verification_key = f"{HUMAN_VERIFICATION_PREFIX}{human_token}"
    verified = await redis_client.getdel(verification_key)
    return verified == "verified"


async def start_email_code_cooldown(email: str) -> bool:
    """为邮箱原子设置一分钟发送冷却；已有冷却时返回 False。"""
    result = await redis_client.set(
        f"{EMAIL_CODE_COOLDOWN_PREFIX}{email}",
        "cooling",
        ex=EMAIL_CODE_COOLDOWN_SECONDS,
        nx=True,
    )
    return bool(result)


async def clear_email_code_cooldown(email: str) -> None:
    """邮件未能投递时清除冷却，允许用户修正邮箱后立即重试。"""
    await redis_client.delete(f"{EMAIL_CODE_COOLDOWN_PREFIX}{email}")


async def save_verification_code(email: str, code: str) -> None:
    """保存邮箱验证码，并由 Redis TTL 控制有效期。"""
    async with redis_client.pipeline(transaction=True) as pipeline:
        pipeline.setex(_code_key(email), VERIFICATION_CODE_TTL_SECONDS, code)
        pipeline.delete(f"{EMAIL_CODE_FAILURE_PREFIX}{email}")
        await pipeline.execute()
    logger.info("Verification code stored")


async def get_verification_code(email: str) -> str | None:
    """读取尚未过期的邮箱验证码。"""
    code = await redis_client.get(_code_key(email))
    logger.info("Verification code looked up")
    return code


async def verify_email_code(email: str, code: str) -> str:
    """Atomically verify, limit guesses to five failures, and consume success."""
    return await cast(
        Awaitable[str],
        redis_client.eval(
            VERIFY_EMAIL_CODE_SCRIPT,
            2,
            _code_key(email),
            f"{EMAIL_CODE_FAILURE_PREFIX}{email}",
            code,
            EMAIL_CODE_MAX_FAILURES,
        ),
    )


async def delete_verification_code(email: str) -> None:
    """删除邮箱验证码，使其不能再次使用。"""
    await redis_client.delete(_code_key(email))
    logger.info("Verification code deleted")


async def test_connection() -> None:
    """验证 Redis 可访问。"""
    await redis_client.ping()  # type: ignore[reportUnknownMemberType, reportGeneralTypeIssues]
    logger.info("Redis 数据库连接成功")
    await redis_client.aclose()


if __name__ == "__main__":
    asyncio.run(test_connection())
