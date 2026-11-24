"""Preflight must reject incomplete setup before loading GPU/database clients."""

import importlib.util
from pathlib import Path


def checker():
    path = Path(__file__).parents[1] / "scripts" / "check_environment.py"
    spec = importlib.util.spec_from_file_location("delivery_preflight", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_empty_config_reports_required_settings_without_loading_models(tmp_path):
    issues = checker().validate_configuration({}, tmp_path)
    assert any("LLM_MODEL_NAME" in issue for issue in issues)
    assert any("JWT_SECRET_KEY" in issue for issue in issues)
    assert any(
        "DATA_ROOT" in issue
        for issue in checker().validate_configuration({"DATA_ROOT": "relative"}, tmp_path)
    )


def test_valid_config_and_missing_optional_model(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    values = dict.fromkeys(
        [
            "LLM_MODEL_NAME",
            "LLM_API_KEY",
            "LLM_BASE_URL",
            "POSTGRESQL_URL",
            "CHROMA_COLLECTION_NAME",
            "ACCESS_TOKEN_CODE",
            "ADMIN_ACCESS_CODE",
        ],
        "configured",
    )
    values.update(
        EMBEDDING_MODEL_PATH=str(model),
        RERANKER_MODEL_PATH=str(model),
        CHROMA_PERSIST_DIRECTORY="data/chroma",
        JWT_SECRET_KEY="x" * 40,
    )
    assert checker().validate_configuration(values, tmp_path) == []
    values["MULTILINGUAL_EMBEDDING_MODEL_PATH"] = str(tmp_path / "missing")
    assert any(
        "MULTILINGUAL" in issue for issue in checker().validate_configuration(values, tmp_path)
    )


def test_configuration_errors_do_not_echo_secret(tmp_path):
    secret = "private-short-value"
    assert secret not in str(checker().validate_configuration({"JWT_SECRET_KEY": secret}, tmp_path))
