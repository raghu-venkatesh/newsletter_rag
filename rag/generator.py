from __future__ import annotations

from ollama import chat

from newsletter_rag.vectorstore.chroma_store import RetrievedChunk


def build_grounded_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    context_lines: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        context_lines.append(
            f"[{idx}] source={chunk.source_file} title={chunk.title}\n{chunk.chunk_text}"
        )
    context_block = "\n\n".join(context_lines) if context_lines else "[NO_CONTEXT]"
    return (
        "You are answering questions over a private newsletter corpus.\n"
        "Use only the provided context. If the context is insufficient, say you do not have enough evidence.\n"
        "Always cite sources with bracket indices like [1], [2].\n"
        "For fact extraction questions, only report values explicitly present in context.\n"
        "If a value is not explicit, say it is not stated.\n"
        "Do not include irrelevant sources.\n\n"
        "If evidence is insufficient, stop after that statement and source citations; do not add extra advice.\n\n"
        f"Question:\n{question}\n\n"
        f"Context:\n{context_block}\n"
    )


def generate_grounded_answer(model: str, question: str, chunks: list[RetrievedChunk]) -> str:
    prompt = build_grounded_prompt(question, chunks)
    response = chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.1},
    )
    return response["message"]["content"].strip()
