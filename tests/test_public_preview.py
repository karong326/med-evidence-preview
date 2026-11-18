import importlib.util
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
