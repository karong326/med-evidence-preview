"""Validate local setup without loading models, opening databases or revealing secrets."""

import argparse
import json
import os
import sys
from pathlib import Path


def validate_configuration(values, root):
    issues = []
    required = (
        "LLM_MODEL_NAME",
        "LLM_API_KEY",
        "LLM_BASE_URL",
        "POSTGRESQL_URL",
        "CHROMA_COLLECTION_NAME",
        "JWT_SECRET_KEY",
        "ACCESS_TOKEN_CODE",
        "ADMIN_ACCESS_CODE",
    )
    for key in required:
        value = values.get(key, "")
        if not value or "replace-with" in value or "your-password" in value:
            issues.append(f"{key}: required configuration is missing or a placeholder")
    if values.get("JWT_SECRET_KEY") and len(values["JWT_SECRET_KEY"].encode()) < 32:
        issues.append("JWT_SECRET_KEY: use at least 32 bytes")
    for key in ("EMBEDDING_MODEL_PATH", "RERANKER_MODEL_PATH", "MULTILINGUAL_EMBEDDING_MODEL_PATH"):
        value = values.get(key, "")
        if key.startswith("MULTILINGUAL") and not value:
            continue
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = root / path
        if not value or not path.is_dir():
            issues.append(f"{key}: local model directory is missing")
    data_root = Path(values.get("DATA_ROOT") or root / "data")
    if not data_root.is_absolute():
        issues.append("DATA_ROOT: use an absolute deployment directory")
    chroma = Path(values.get("CHROMA_PERSIST_DIRECTORY") or data_root / "chroma")
    if (root / chroma).resolve() == root.resolve():
        issues.append("CHROMA_PERSIST_DIRECTORY: use a dedicated data directory")
    if values.get("RERANKER_PRECISION", "float16") not in {"float16", "float32", "bfloat16"}:
        issues.append("RERANKER_PRECISION: invalid precision")
    return issues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--cuda", action="store_true", help="Also check the installed CUDA runtime")
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        from dotenv import dotenv_values
    except ImportError:
        print("Install project dependencies before running this check.")
        return 1
    values = {k: v for k, v in dotenv_values(root / ".env").items() if v}
    values.update(os.environ)  # Same precedence as utils/config.py.
    issues = validate_configuration(values, root)
    if sys.version_info[:2] != (3, 12):
        issues.append("Python: this project requires Python 3.12")
    if not (root / "frontend" / "dist" / "index.html").is_file():
        issues.append("frontend: run npm ci and npm run build in frontend/")
    if args.cuda:
        try:
            import torch

            if not torch.cuda.is_available():
                issues.append("CUDA: required by the current embedding and reranker implementation")
        except ImportError:
            issues.append("PyTorch: not installed")
    print(json.dumps({"ok": not issues, "issues": issues}, ensure_ascii=False, indent=2))
    return int(bool(issues))


if __name__ == "__main__":
    raise SystemExit(main())
