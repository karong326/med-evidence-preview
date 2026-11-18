"""解析项目路径、.env 覆盖项和运行期配置常量。

模块优先从源码包所在项目定位 .env，再检查工作目录及其父目录；进程环境变量优先，
.env 中非空值仅补充尚未设置的变量，空值不覆盖已有凭证。
随后集中定义模型、存储、认证和检索配置，供其他模块只读导入。
"""

import os
from pathlib import Path

from dotenv import dotenv_values

# 运行目录可能属于另一个服务；优先以本包所在的项目目录为准，再允许安装场景
# 通过当前目录或上级目录提供配置。这样从 Harness 目录启动也不会读错 .env。
working_directory = Path.cwd().resolve()
source_project = Path(__file__).resolve().parents[3]
project_candidates = [
    source_project,
    working_directory,
    working_directory / "med-evidence-agentic-rag",
    *working_directory.parents,
]
PROJECT_ROOT = next(
    (path for path in project_candidates if (path / ".env").is_file()), working_directory
)
ENV_FILE = PROJECT_ROOT / ".env"


def load_project_environment(env_file: Path) -> None:
    """加载项目非空配置；部署进程显式提供的变量优先。"""
    for name, value in dotenv_values(env_file).items():
        if value and name not in os.environ:
            os.environ[name] = value


# 进程环境变量优先；项目非空值补充缺项，空占位不覆盖已有凭证。
load_project_environment(ENV_FILE)


# ===============================================================
# llm 大语言 模型连接配置。
llm_model_name = os.getenv("LLM_MODEL_NAME", "")
llm_api_key = os.getenv("LLM_API_KEY", "")
llm_base_url = os.getenv("LLM_BASE_URL", "")

# 本地 Embedding 向量化 模型路径。
embedding_model_path = os.getenv("EMBEDDING_MODEL_PATH", "")
embedding_model = Path(embedding_model_path)

# 可选辅助召回：独立本地多语言模型，不改变原 Chroma 向量空间。
multilingual_embedding_model_path = os.getenv("MULTILINGUAL_EMBEDDING_MODEL_PATH", "")

# 本地 Cross-Encoder 重排序 模型路径。
reranker_model_path = os.getenv("RERANKER_MODEL_PATH", "")
reranker_model = Path(reranker_model_path)

# Chroma 向量数据库 持久化目录。
chroma_persist_directory = Path(os.getenv("CHROMA_PERSIST_DIRECTORY", ""))
if not chroma_persist_directory.is_absolute():
    chroma_persist_directory = (PROJECT_ROOT / chroma_persist_directory).resolve()

chroma_collection_name = os.getenv("CHROMA_COLLECTION_NAME", "")


# 完整检索流程和参数：
# 用户问题
# ├─ Dense 向量检索：Top 80 候选
# ├─ BM25 关键词检索：Top 80 候选
# └─ 可选本地多语言向量检索：Top 80 候选
#         ↓ ↓
# RRF 融合：去重后取 Top 80，rrf_k=50 为平滑参数
#          ↓
# Cross-Encoder：结合来源文件名重排候选
#          ↓
# 分数 ≥ 0.50 的最终 Top 3 证据
hybrid_candidate_k = int(os.getenv("HYBRID_CANDIDATE_K", "80"))
hybrid_top_k = int(os.getenv("HYBRID_TOP_K", "80"))
rrf_k = 50
reranker_top_k = 3
reranker_batch_size = 10
reranker_min_score = float(os.getenv("RERANKER_MIN_SCORE", "0.50"))
reranker_include_source = os.getenv("RERANKER_INCLUDE_SOURCE", "true").lower() == "true"
reranker_precision = os.getenv("RERANKER_PRECISION", "float16")
if reranker_precision not in {"float16", "float32", "bfloat16"}:
    raise ValueError("RERANKER_PRECISION 必须是 float16、float32 或 bfloat16")


# PostgreSQL、Redis 和登录令牌配置。
postgresql_url = os.getenv("POSTGRESQL_URL", "")

redis_url = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")

jwt_secret_key = os.getenv("JWT_SECRET_KEY", "")
jwt_algorithm = os.getenv("JWT_ALGORITHM", "HS256")


# 访问令牌：普通用户进入聊天，管理员进入后台。
access_token_code = os.getenv("ACCESS_TOKEN_CODE", "")
admin_access_code = os.getenv("ADMIN_ACCESS_CODE", "")


# MinerU 精准解析凭证。
mineru_token = os.getenv("MINERU_TOKEN", "")
