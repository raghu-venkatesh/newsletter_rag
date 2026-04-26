from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from newsletter_rag.rag.generator import generate_grounded_answer
from newsletter_rag.rag.retriever import NewsletterRetriever
from newsletter_rag.rag.settings import RagSettings


@dataclass(frozen=True)
class RagAnswer:
    question: str
    answer: str
    sources: list[dict[str, Any]]


class NewsletterRagPipeline:
    def __init__(self, settings: RagSettings) -> None:
        self.settings = settings
        self.retriever = NewsletterRetriever(settings)

    def ask(self, question: str, top_k: int | None = None) -> RagAnswer:
        chunks = self.retriever.retrieve(question=question, top_k=top_k)
        answer = generate_grounded_answer(
            model=self.settings.generation_model,
            question=question,
            chunks=chunks,
        )
        return RagAnswer(
            question=question,
            answer=answer,
            sources=[
                {
                    "rank": i + 1,
                    "chunk_id": c.chunk_id,
                    "source_file": c.source_file,
                    "title": c.title,
                    "distance": c.distance,
                    **({"rerank_score": c.rerank_score} if c.rerank_score is not None else {}),
                }
                for i, c in enumerate(chunks)
            ],
        )

    @staticmethod
    def to_dict(result: RagAnswer) -> dict[str, Any]:
        return asdict(result)
