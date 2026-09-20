import numpy as np

from src import embeddings


class _FakeModel:
    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        return np.ones((len(texts), 384), dtype=np.float32)


def test_snowflake_onnx_uses_int8_backend_and_query_prompt(monkeypatch):
    fake = _FakeModel()
    loaded = []

    def load(*args):
        loaded.append(args)
        return fake

    monkeypatch.setattr(embeddings, "_load_sentence_transformer", load)
    model = embeddings.LocalEmbedder(
        "Snowflake/snowflake-arctic-embed-xs",
        batch_size=128,
        inference_backend="onnx",
        onnx_file="onnx/model_int8.onnx",
        threads=4,
    )
    model.encode(["document"])
    model.encode_query(["question"])

    assert loaded == [(
        "Snowflake/snowflake-arctic-embed-xs",
        "onnx",
        "onnx/model_int8.onnx",
        4,
    )]
    assert fake.calls[0][1].get("prompt_name") is None
    assert fake.calls[1][0] == [
        "Represent this sentence for searching relevant passages: question"
    ]
    assert fake.calls[1][1].get("prompt_name") is None
    assert "onnx/model_int8.onnx" in model.backend


def test_hash_fallback_supports_query_encoding():
    model = embeddings.HashEmbedder()
    assert np.array_equal(model.encode(["same"]), model.encode_query(["same"]))
