from __future__ import annotations

from newsletter_rag.rag.retriever import NewsletterRetriever
from newsletter_rag.rag.settings import load_settings


EVAL_QUESTIONS = [
    "What is the difference between AI in the business and AI on the business?",
    "Why are output tokens usually more expensive than input tokens?",
    "What is a token and why does tokenization matter?",
]


def main() -> None:
    settings = load_settings()
    retriever = NewsletterRetriever(settings)
    for question in EVAL_QUESTIONS:
        print(f"\nQ: {question}")
        chunks = retriever.retrieve(question, top_k=2)
        for idx, chunk in enumerate(chunks, start=1):
            rs = f" | rerank={chunk.rerank_score:.4f}" if chunk.rerank_score is not None else ""
            print(f"  [{idx}] {chunk.source_file} | {chunk.title} | distance={chunk.distance:.4f}{rs}")


if __name__ == "__main__":
    main()
