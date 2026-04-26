from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    doc_id: str
    source_file: str
    title: str
    chunk_index: int
    text: str


def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 120) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    cleaned = " ".join(text.split())
    if not cleaned:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + chunk_size)
        window = cleaned[start:end]
        if end < len(cleaned):
            last_space = window.rfind(" ")
            if last_space > int(chunk_size * 0.6):
                window = window[:last_space]
                end = start + last_space
        chunks.append(window.strip())
        if end >= len(cleaned):
            break
        start = end - chunk_overlap
    return chunks


def build_chunks(
    *,
    doc_id: str,
    source_file: str,
    title: str,
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> Iterable[TextChunk]:
    for idx, chunk in enumerate(chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)):
        yield TextChunk(
            chunk_id=f"{doc_id}:chunk:{idx}",
            doc_id=doc_id,
            source_file=source_file,
            title=title,
            chunk_index=idx,
            text=chunk,
        )
