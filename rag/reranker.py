from __future__ import annotations

from dataclasses import replace
from typing import Any

from newsletter_rag.vectorstore.chroma_store import RetrievedChunk


class CrossEncoderReranker:
    """Rerank retrieved chunks with a cross-encoder (same idea as newsletter_expert_reflection-assets)."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model: Any = None

    @property
    def model(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if not chunks:
            return []
        pairs = [(query, c.chunk_text) for c in chunks]
        scores = self.model.predict(pairs, show_progress_bar=False, batch_size=16)
        scored = sorted(zip(chunks, scores), key=lambda x: float(x[1]), reverse=True)
        out: list[RetrievedChunk] = []
        for chunk, score in scored[:top_k]:
            out.append(replace(chunk, rerank_score=float(score)))
        return out
