from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from pathlib import Path
from typing import Protocol

import numpy as np


SNOWFLAKE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class _OnnxSentenceEncoder:
    """Minimal ONNX Runtime encoder for BERT-style sentence models."""

    def __init__(self, model_name: str, onnx_file: str, threads: int | None):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        model_path = Path(model_name)
        if model_path.exists():
            graph_path = model_path / onnx_file
            tokenizer_source = model_path
        else:
            from huggingface_hub import hf_hub_download

            graph_path = Path(hf_hub_download(repo_id=model_name, filename=onnx_file))
            tokenizer_source = model_name
        if not graph_path.exists():
            raise FileNotFoundError(f"ONNX embedding graph not found: {graph_path}")
        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(graph_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
        self.input_names = {item.name for item in self.session.get_inputs()}

    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int = 256,
        normalize_embeddings: bool = True,
        **_kwargs,
    ) -> np.ndarray:
        chunks: list[np.ndarray] = []
        for offset in range(0, len(texts), batch_size):
            encoded = self.tokenizer(
                texts[offset:offset + batch_size],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="np",
            )
            inputs = {
                name: np.asarray(encoded[name], dtype=np.int64)
                for name in self.input_names
                if name in encoded
            }
            # Arctic Embed is trained to use the first-token (CLS) vector.
            token_embeddings = np.asarray(self.session.run(None, inputs)[0], dtype=np.float32)
            vectors = token_embeddings[:, 0, :]
            if normalize_embeddings:
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                vectors = vectors / np.maximum(norms, 1e-12)
            chunks.append(vectors)
        return np.concatenate(chunks, axis=0) if chunks else np.empty((0, 0), dtype=np.float32)


@lru_cache(maxsize=8)
def _load_sentence_transformer(
    model_name: str,
    backend: str = "torch",
    onnx_file: str | None = None,
    threads: int | None = None,
):
    from sentence_transformers import SentenceTransformer

    if backend == "torch":
        return SentenceTransformer(model_name)
    if backend != "onnx":
        raise ValueError(f"Unsupported embedding backend: {backend}")
    try:
        import onnxruntime  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "ONNX embeddings require `python -m pip install -e .` to install onnxruntime"
        ) from exc
    return _OnnxSentenceEncoder(model_name, onnx_file or "onnx/model.onnx", threads)


class Embedder(Protocol):
    backend: str
    def encode(self, texts: list[str]) -> np.ndarray: ...
    def encode_query(self, texts: list[str]) -> np.ndarray: ...


class HashEmbedder:
    """Deterministic offline test double; never a benchmark backend."""
    backend = "deterministic-hash-fallback"

    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions

    def encode(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
                index = int.from_bytes(digest, "little") % self.dimensions
                matrix[row, index] += 1.0
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.maximum(norms, 1e-12)

    def encode_query(self, texts: list[str]) -> np.ndarray:
        return self.encode(texts)


class LocalEmbedder:
    def __init__(
        self,
        model_name: str,
        allow_fallback: bool = False,
        batch_size: int = 256,
        inference_backend: str = "torch",
        onnx_file: str | None = None,
        threads: int | None = None,
        query_prompt_name: str | None = None,
    ):
        self.batch_size = batch_size
        self.inference_backend = inference_backend
        self.query_prompt_name = query_prompt_name
        if self.query_prompt_name is None and model_name.rstrip("/").lower().endswith(
            "snowflake-arctic-embed-xs"
        ):
            self.query_prompt_name = "query"
        if allow_fallback:
            self.model = None
            self._fallback = HashEmbedder()
            self.backend = self._fallback.backend
            return
        try:
            self.model = _load_sentence_transformer(
                model_name, inference_backend, onnx_file, threads
            )
            suffix = f":{onnx_file or 'model.onnx'}" if inference_backend == "onnx" else ""
            self.backend = f"{model_name}:{inference_backend}{suffix}"
            self._fallback = None
        except Exception:
            raise

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._fallback:
            return self._fallback.encode(texts)
        return np.asarray(
            self.model.encode(
                texts, batch_size=self.batch_size, normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )

    def encode_query(self, texts: list[str]) -> np.ndarray:
        if self._fallback:
            return self._fallback.encode_query(texts)
        if (
            self.inference_backend == "onnx"
            and self.query_prompt_name == "query"
        ):
            texts = [SNOWFLAKE_QUERY_PREFIX + text for text in texts]
            kwargs = {}
        else:
            kwargs = {"prompt_name": self.query_prompt_name} if self.query_prompt_name else {}
        return np.asarray(
            self.model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
                **kwargs,
            ),
            dtype=np.float32,
        )
