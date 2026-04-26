from __future__ import annotations

import re
from ollama import embed

from newsletter_rag.rag.reranker import CrossEncoderReranker
from newsletter_rag.rag.settings import RagSettings
from newsletter_rag.vectorstore.chroma_store import ChromaVectorStore, RetrievedChunk


class NewsletterRetriever:
    def __init__(self, settings: RagSettings) -> None:
        self.settings = settings
        self.store = ChromaVectorStore(
            persist_directory=settings.chroma_persist_dir,
            collection_name=settings.chroma_collection,
        )
        self._reranker: CrossEncoderReranker | None = None
        if settings.use_reranker:
            self._reranker = CrossEncoderReranker(settings.reranker_model)

    def embed_query(self, question: str) -> list[float]:
        response = embed(model=self.settings.embedding_model, input=question)
        return response["embeddings"][0]

    @staticmethod
    def _expand_query(question: str) -> list[str]:
        base = question.strip()
        variants = [base]
        if base.endswith("?"):
            variants.append(base[:-1].strip())
        return list(dict.fromkeys(v for v in variants if v))

    @staticmethod
    def _keyword_bonus(question: str, text: str) -> float:
        q_terms = set(re.findall(r"[a-z0-9]+", question.lower()))
        t_terms = set(re.findall(r"[a-z0-9]+", text.lower()))
        stop = {
            "the", "a", "an", "and", "or", "to", "of", "in", "on", "for",
            "is", "are", "was", "were", "be", "with", "what", "which", "how",
            "when", "where", "why", "who", "that", "this", "these", "those",
        }
        q_terms = {t for t in q_terms if t not in stop and len(t) > 2}
        if not q_terms:
            return 0.0
        overlap = len(q_terms.intersection(t_terms))
        ratio = overlap / len(q_terms)
        return min(0.12 * ratio, 0.12)

    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievedChunk]:
        final_k = top_k or self.settings.top_k_generate
        retrieve_k = max(final_k, self.settings.top_k_retrieve)
        queries = self._expand_query(question) if self.settings.query_expansion else [question]

        pool: dict[str, RetrievedChunk] = {}
        for q in queries:
            query_embedding = self.embed_query(q)
            for chunk in self.store.similarity_search(query_embedding, top_k=retrieve_k):
                existing = pool.get(chunk.chunk_id)
                if existing is None or chunk.distance < existing.distance:
                    pool[chunk.chunk_id] = chunk

        candidates = sorted(pool.values(), key=lambda c: c.distance)[:retrieve_k]

        if self._reranker is not None and candidates:
            return self._reranker.rerank(question, candidates, top_k=final_k)

        filtered = [c for c in candidates if c.distance <= self.settings.max_distance]
        if not filtered:
            filtered = sorted(candidates, key=lambda c: c.distance)[: max(1, final_k)]

        reranked = sorted(
            filtered,
            key=lambda c: c.distance - self._keyword_bonus(question, c.chunk_text),
        )
        return reranked[:final_k]
