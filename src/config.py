from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    # Non-Windows installations can continue using Python's default CA store.
    pass

USE_DIRECT_CONTEXT = False
assert USE_DIRECT_CONTEXT is False


@dataclass(frozen=True)
class Settings:
    typesafe_api_key: str | None = None
    openai_api_key: str | None = None
    longmemeval_path: Path | None = None
    longmemeval_oracle_path: Path | None = None
    openai_model: str = "gpt-4o"
    jev_model: str = "jev-latest"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_backend: str = "torch"
    embedding_onnx_file: str | None = None
    embedding_threads: int | None = None
    embedding_query_prompt_name: str | None = None
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    embedding_batch_size: int = 256
    reranker_batch_size: int = 64
    results_dir: Path = Path("results")
    max_depth: int = 3
    max_assignments: int = 2
    max_new_topics: int = 1
    max_search_steps: int = 6
    jev_routing_batch_size: int = 25

    @classmethod
    def from_env(cls) -> "Settings":
        # Load the project-root file while preserving values explicitly supplied
        # by the calling shell or process environment.
        load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
        dataset = os.getenv("LONGMEMEVAL_PATH")
        oracle = os.getenv("LONGMEMEVAL_ORACLE_PATH")
        return cls(
            typesafe_api_key=os.getenv("TYPESAFE_API_KEY"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            longmemeval_path=Path(dataset) if dataset else None,
            longmemeval_oracle_path=Path(oracle) if oracle else None,
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            jev_model=os.getenv("JEV_MODEL", "jev-latest"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            embedding_backend=os.getenv("EMBEDDING_BACKEND", "torch").lower(),
            embedding_onnx_file=os.getenv("EMBEDDING_ONNX_FILE") or None,
            embedding_threads=(
                int(os.environ["EMBEDDING_THREADS"]) if os.getenv("EMBEDDING_THREADS") else None
            ),
            embedding_query_prompt_name=os.getenv("EMBEDDING_QUERY_PROMPT_NAME") or None,
            reranker_model=os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            embedding_batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "256")),
            reranker_batch_size=int(os.getenv("RERANKER_BATCH_SIZE", "64")),
            results_dir=Path(os.getenv("RESULTS_DIR", "results")),
            jev_routing_batch_size=int(os.getenv("JEV_ROUTING_BATCH_SIZE", "25")),
        )
