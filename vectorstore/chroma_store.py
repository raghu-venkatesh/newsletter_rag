from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    source_file: str
    title: str
    chunk_text: str
    distance: float
    metadata: dict[str, Any]
    rerank_score: float | None = None


class ChromaVectorStore:
    def __init__(self, persist_directory: str, collection_name: str = "newsletter_chunks") -> None:
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection: Collection = self.client.get_or_create_collection(name=collection_name)

    def ensure_schema(self, embedding_dimensions: int) -> None:
        # Chroma is schemaless for this use-case; this validates embeddings are configured by caller.
        if embedding_dimensions <= 0:
            raise ValueError("embedding_dimensions must be > 0")

    def upsert_chunks(self, rows: list[dict[str, Any]]) -> int:
        if not rows:
            return 0
        self.collection.upsert(
            ids=[row["chunk_id"] for row in rows],
            documents=[row["chunk_text"] for row in rows],
            metadatas=[
                {
                    **row.get("metadata", {}),
                    "doc_id": row["doc_id"],
                    "source_file": row["source_file"],
                    "title": row["title"],
                    "chunk_index": row["chunk_index"],
                }
                for row in rows
            ],
            embeddings=[row["embedding"] for row in rows],
        )
        return len(rows)

    def list_chunk_ids(self) -> set[str]:
        rows = self.collection.get(include=[])
        ids = rows.get("ids", [])
        return {chunk_id for chunk_id in ids if isinstance(chunk_id, str)}

    def delete_chunk_ids(self, chunk_ids: list[str]) -> int:
        if not chunk_ids:
            return 0
        self.collection.delete(ids=chunk_ids)
        return len(chunk_ids)

    def similarity_search(self, query_embedding: list[float], top_k: int = 4) -> list[RetrievedChunk]:
        results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        return [
            RetrievedChunk(
                chunk_id=chunk_id,
                source_file=(metadata or {}).get("source_file", "unknown"),
                title=(metadata or {}).get("title", "unknown"),
                chunk_text=chunk_text,
                distance=float(distance),
                metadata=metadata if isinstance(metadata, dict) else {},
            )
            for chunk_id, chunk_text, metadata, distance in zip(ids, documents, metadatas, distances)
        ]

    def collection_stats(self) -> dict[str, int]:
        count = self.collection.count()
        all_rows = self.collection.get(include=["metadatas"])
        doc_ids = {
            metadata.get("doc_id")
            for metadata in all_rows.get("metadatas", [])
            if isinstance(metadata, dict) and metadata.get("doc_id")
        }
        return {
            "chunk_count": int(count),
            "document_count": len(doc_ids),
        }
