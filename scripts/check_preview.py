"""Check exactly the files a Git checkout exposes to a code-preview reviewer."""

import subprocess
from pathlib import Path, PurePosixPath

ROOT_FILES = {
    "README.md",
    ".env.example",
    ".gitignore",
    ".gitattributes",
    ".python-version",
    "pyproject.toml",
}


def inspect_paths(paths: list[str]) -> list[str]:
    issues: list[str] = []
    for name in paths:
        path = PurePosixPath(name)
        if name in ROOT_FILES or name == "data/raw/.gitkeep":
            continue
        parts = path.parts
        allowed = (
            parts[0] == "src"
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


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    tracked = (
        subprocess.check_output(["git", "-C", str(root), "ls-files", "-z"]).decode().split("\0")
    )
    paths = [name for name in tracked if name and (root / name).is_file()]
    issues = inspect_paths(paths)
    readme = root.joinpath("README.md").read_text(encoding="utf-8")
    if "本人离职后" not in readme or "基础预览版" not in readme:
        issues.append("README must identify this as the author's post-employment basic preview")
    for issue in issues:
        print(issue)
    if not issues:
        print("Public preview contains code, configuration and one root README.")
    return int(bool(issues))


if __name__ == "__main__":
    raise SystemExit(main())
