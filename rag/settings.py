from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class RagSettings:
    chroma_persist_dir: str
    chroma_collection: str
    embedding_model: str
    generation_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    top_k_retrieve: int
    top_k_generate: int
    max_distance: float
    query_expansion: bool
    use_reranker: bool
    reranker_model: str
    corpus_dir: str


def load_settings() -> RagSettings:
    base_dir = Path(__file__).resolve().parents[1]
    load_dotenv(base_dir / ".env")
    repo_root = base_dir.parent
    top_k_val = int(os.getenv("TOP_K", "5"))
    return RagSettings(
        chroma_persist_dir=os.getenv("CHROMA_PERSIST_DIR", str(base_dir / "data" / "chroma")),
        chroma_collection=os.getenv("CHROMA_COLLECTION", "newsletter_chunks"),
        embedding_model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
        generation_model=os.getenv("OLLAMA_GENERATE_MODEL", "llama3.2:3b"),
        chunk_size=int(os.getenv("CHUNK_SIZE", "800")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "120")),
        top_k=top_k_val,
        top_k_retrieve=int(os.getenv("TOP_K_RETRIEVE", "30")),
        top_k_generate=int(os.getenv("TOP_K_GENERATE", str(top_k_val))),
        max_distance=float(os.getenv("MAX_DISTANCE", "2.0")),
        query_expansion=os.getenv("QUERY_EXPANSION", "1").strip().lower() not in {"0", "false", "no"},
        use_reranker=os.getenv("USE_RERANKER", "1").strip().lower() not in {"0", "false", "no"},
        reranker_model=os.getenv(
            "RERANKER_MODEL",
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
        ),
        corpus_dir=os.getenv("NEWSLETTER_CORPUS_DIR", str(repo_root / "turing_newsletter")),
    )
