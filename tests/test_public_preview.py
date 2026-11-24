import importlib.util
import subprocess
from pathlib import Path


def checker():
    path = Path(__file__).parents[1] / "scripts/check_preview.py"
    spec = importlib.util.spec_from_file_location("preview_check", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_only_root_readme_is_public_documentation():
    inspect = checker().inspect_paths
    assert inspect(["README.md", "src/a.py", ".github/workflows/ci.yml", ".env.example"]) == []
    issues = inspect(
        ["README.md", "docs/REPORT.md", "frontend/README.md", "results/summary.json", ".env"]
    )
    assert len(issues) == 4


def test_dataset_results_and_runtime_artifacts_are_rejected():
    issues = checker().inspect_paths(
        [
            "README.md",
            "evaluations/dev.jsonl",
            "data/raw/book.pdf",
            "data/chroma/index.sqlite3",
            "models/model.bin",
            "frontend/dist/index.html",
            "tmp/capture.png",
        ]
    )
    assert len(issues) == 6


def test_git_scan_rejects_index_secrets_and_links_even_if_worktree_is_clean(tmp_path):
    def git(*args, payload=None):
        return subprocess.check_output(["git", "-C", str(tmp_path), *args], input=payload)

    git("init", "-q")
    (tmp_path / "README.md").write_text(
        "这是本人离职后重新开发的基础预览版项目", encoding="utf-8"
    )
    (tmp_path / "src").mkdir()
    staged = tmp_path / "src/private.py"
    secret = "sk-" + "H6jFs9Q2Lm" * 4
    staged.write_text(f'LLM_API_KEY = "{secret}"', encoding="utf-8")
    git("add", "README.md", "src/private.py")
    staged.unlink()
    blob = git("hash-object", "-w", "--stdin", payload=b"../outside").decode().strip()
    git("update-index", "--add", "--cacheinfo", f"120000,{blob},src/link.py")

    issues = checker().inspect_git(tmp_path)
    assert any("src/private.py" in issue for issue in issues)
    assert any("src/link.py" in issue for issue in issues)
    assert secret not in str(issues)
