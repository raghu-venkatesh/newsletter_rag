from __future__ import annotations

import argparse
from typing import Any

from ollama import embed

from newsletter_rag.ingestion.chunking import build_chunks
from newsletter_rag.ingestion.load_newsletters import load_newsletters
from newsletter_rag.rag.settings import load_settings
from newsletter_rag.vectorstore.chroma_store import ChromaVectorStore


def _embed_texts(model: str, texts: list[str]) -> list[list[float]]:
    response = embed(model=model, input=texts)
    return response["embeddings"]


def index_newsletters(top_level_corpus: str | None = None, prune_stale: bool = True) -> dict[str, Any]:
    settings = load_settings()
    corpus_dir = top_level_corpus or settings.corpus_dir
    docs = load_newsletters(corpus_dir=corpus_dir)

    store = ChromaVectorStore(settings.chroma_persist_dir, settings.chroma_collection)

    all_rows: list[dict[str, Any]] = []
    for doc in docs:
        for chunk in build_chunks(
            doc_id=doc.doc_id,
            source_file=doc.source_file,
            title=doc.title,
            text=doc.body,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        ):
            all_rows.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "source_file": chunk.source_file,
                    "title": chunk.title,
                    "chunk_index": chunk.chunk_index,
                    "chunk_text": chunk.text,
                    "metadata": {
                        "source_file": chunk.source_file,
                        "title": chunk.title,
                    },
                }
            )

    if not all_rows:
        if prune_stale:
            stale_count = store.delete_chunk_ids(list(store.list_chunk_ids()))
            stats = store.collection_stats()
            return {"documents": 0, "chunks_indexed": 0, "chunks_deleted": stale_count, "stats": stats}
        return {"documents": 0, "chunks_indexed": 0, "chunks_deleted": 0, "stats": {"chunk_count": 0, "document_count": 0}}

    embeddings = _embed_texts(settings.embedding_model, [r["chunk_text"] for r in all_rows])
    for row, embedding in zip(all_rows, embeddings):
        row["embedding"] = embedding

    store.ensure_schema(embedding_dimensions=len(embeddings[0]))
    deleted = 0
    if prune_stale:
        indexed_ids = {row["chunk_id"] for row in all_rows}
        existing_ids = store.list_chunk_ids()
        stale_ids = sorted(existing_ids - indexed_ids)
        deleted = store.delete_chunk_ids(stale_ids)
    upserted = store.upsert_chunks(all_rows)
    stats = store.collection_stats()
    return {"documents": len(docs), "chunks_indexed": upserted, "chunks_deleted": deleted, "stats": stats}


def main() -> None:
    parser = argparse.ArgumentParser(description="Index newsletter corpus into ChromaDB")
    parser.add_argument("--corpus-dir", default=None, help="Path to directory with .txt newsletters")
    parser.add_argument(
        "--no-prune-stale",
        action="store_true",
        help="Disable stale-vector cleanup (by default, stale vectors are deleted).",
    )
    args = parser.parse_args()
    result = index_newsletters(top_level_corpus=args.corpus_dir, prune_stale=not args.no_prune_stale)
    print(f"Documents processed: {result['documents']}")
    print(f"Chunks indexed: {result['chunks_indexed']}")
    print(f"Chunks deleted (stale cleanup): {result['chunks_deleted']}")
    print(f"Collection stats: {result['stats']}")


if __name__ == "__main__":
    main()
