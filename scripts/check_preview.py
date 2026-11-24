"""Check exactly the files a Git checkout exposes to a code-preview reviewer."""

import argparse
import re
import subprocess
from pathlib import Path, PurePosixPath

ROOT_FILES = {
    "README.md",
    ".env.example",
    ".gitignore",
    ".gitattributes",
    ".python-version",
    "pyproject.toml",
    "alembic.ini",
    ".dockerignore",
    "Dockerfile",
    "compose.yaml",
}
DEPLOYMENT_FILES = {"deploy/prometheus.yml", "deploy/alerts.yml"}
SECRET_KEYS = (
    "LLM_API_KEY|MINERU_TOKEN|JWT_SECRET_KEY|MY_SENDER_PASSWORD|DIAGNOSTICS_TOKEN"
    "|ACCESS_TOKEN_CODE|ADMIN_ACCESS_CODE|POSTGRES_PASSWORD|EVAL_JUDGE_API_KEY"
)
ASSIGNMENT = re.compile(
    rf'(?im)^[ \t]*(?P<key>{SECRET_KEYS})[ \t]*[:=][ \t]*'
    r'(?:"(?P<double>[^"\r\n]*)"|\x27(?P<single>[^\x27\r\n]*)\x27|'
    r'(?P<bare>[^\s#]+))[ \t]*(?:#.*)?$'
)
PROVIDER_SECRET = re.compile(
    r"(?:\bsk-(?:proj-)?[A-Za-z0-9_-]{26,}|\b(?:gh[pousr]_|github_pat_)[A-Za-z0-9_]{30,}"
    r"|\bAKIA[A-Z0-9]{16}\b|-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----)"
)
PERSONAL_PATH = re.compile(r'(?i)(?:[a-z]:[\\/]+Users[\\/]+|/(?:home|Users)/)[^\s"\x27/$\[{]+')
DB_URI = re.compile(r"(?i)\b(?:postgres(?:ql)?(?:\+\w+)?|rediss?)://[^\s\x27\x22]+")


def inspect_paths(paths: list[str]) -> list[str]:
    issues: list[str] = []
    for name in paths:
        path = PurePosixPath(name)
        if name in ROOT_FILES or name in DEPLOYMENT_FILES or name == "data/raw/.gitkeep":
            continue
        parts = path.parts
        allowed = (
            parts[0] in {"src", "migrations"}
            and path.suffix == ".py"
            or parts[0] == "tests"
            and path.suffix == ".py"
            or parts[0] == "scripts"
            and path.suffix in {".py", ".ps1"}
            or parts[:2] == (".github", "workflows")
            and path.suffix in {".yaml", ".yml"}
            or parts[0] == "frontend"
            and (
                name
                in {
                    "frontend/package.json",
                    "frontend/package-lock.json",
                    "frontend/index.html",
                    "frontend/vite.config.js",
                }
                or len(parts) > 2
                and parts[1] in {"src", "public", "test"}
                and path.suffix in {".vue", ".js", ".mjs", ".css", ".svg"}
            )
        )
        if not allowed:
            issues.append(f"Not part of public code preview: {name}")
    if "README.md" not in paths:
        issues.append("The root README.md is required")
    return issues


def inspect_content(name: str, data: bytes) -> list[str]:
    if name == "data/raw/.gitkeep":
        return [f"Data marker must be empty: {name}"] if data else []
    try:
        content = data.decode("utf-8")
    except UnicodeDecodeError:
        return [f"Public preview files must be UTF-8 text: {name}"]
    issues = []
    if "\0" in content:
        issues.append(f"Binary content: {name}")
    if PROVIDER_SECRET.search(content):
        issues.append(f"Possible provider key or private key: {name}")
    if PERSONAL_PATH.search(content):
        issues.append(f"Possible personal filesystem path: {name}")
    for match in ASSIGNMENT.finditer(content):
        value = match["double"] or match["single"] or match["bare"] or ""
        if value and not (
            value.startswith(("replace-with-", "${"))
            or value in {"None", "''", '""'}
        ):
            issues.append(f"Possible hardcoded {match['key'].upper()}: {name}")
    for match in DB_URI.finditer(content):
        credentials, _, _ = match[0].split("://", 1)[1].rpartition("@")
        if ":" in credentials:
            password = credentials.split(":", 1)[1]
            if password not in {"your-password", "test", "pass", "p"} and not password.startswith(
                ("${", "replace-with-")
            ):
                issues.append(f"Possible database credentials: {name}")
    return issues


def inspect_git(root: Path, *, head: bool = False) -> list[str]:
    git = ["git", "-C", str(root)]
    inventory = ["ls-tree", "-rz", "HEAD"] if head else ["ls-files", "--stage", "-z"]
    entries = subprocess.check_output([*git, *inventory]).decode("utf-8").split("\0")
    paths: list[str] = []
    blobs: dict[str, str] = {}
    unsupported: list[str] = []
    for entry in filter(None, entries):
        metadata, name = entry.split("\t", 1)
        mode, middle, tail = metadata.split()
        paths.append(name)
        if mode not in {"100644", "100755"} or (not head and tail != "0"):
            unsupported.append(f"Unsupported Git entry mode or stage: {name}")
        else:
            blobs[name] = tail if head else middle

    issues = inspect_paths(paths)
    issues.extend(unsupported)
    with subprocess.Popen(
        [*git, "cat-file", "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE
    ) as process:
        assert process.stdin is not None and process.stdout is not None
        for name, object_id in blobs.items():
            process.stdin.write(object_id.encode("ascii") + b"\n")
            process.stdin.flush()
            header = process.stdout.readline().split()
            if len(header) != 3 or header[1] != b"blob":
                issues.append(f"Could not read Git blob: {name}")
                continue
            content = process.stdout.read(int(header[2]))
            process.stdout.read(1)
            issues.extend(inspect_content(name, content))
            if name == "README.md" and (
                "这是本人离职后重新开发的基础预览版项目".encode() not in content
            ):
                issues.append(
                    "README must identify this as the author's post-employment basic preview"
                )
        process.stdin.close()
        if process.wait():
            issues.append("Git content inspection failed")
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--head", action="store_true", help="Inspect committed HEAD instead of index"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    issues = inspect_git(root, head=args.head)
    for issue in issues:
        print(issue)
    if not issues:
        print("Public preview contains code, configuration and one root README.")
    return int(bool(issues))


if __name__ == "__main__":
    raise SystemExit(main())
