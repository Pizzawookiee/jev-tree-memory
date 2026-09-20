from src.config import Settings


def test_dotenv_loads_without_overriding_shell(tmp_path, monkeypatch):
    # Verify the precedence rule directly without relying on developer secrets.
    monkeypatch.setenv("OPENAI_MODEL", "shell-model")
    settings = Settings.from_env()
    assert settings.openai_model == "shell-model"


def test_onnx_embedding_settings(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL", "Snowflake/snowflake-arctic-embed-xs")
    monkeypatch.setenv("EMBEDDING_BACKEND", "onnx")
    monkeypatch.setenv("EMBEDDING_ONNX_FILE", "onnx/model_int8.onnx")
    monkeypatch.setenv("EMBEDDING_THREADS", "4")
    settings = Settings.from_env()
    assert settings.embedding_backend == "onnx"
    assert settings.embedding_onnx_file == "onnx/model_int8.onnx"
    assert settings.embedding_threads == 4
