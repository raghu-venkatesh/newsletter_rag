# Newsletter RAG (uv-first)

This subproject implements newsletter ingestion, ChromaDB indexing, and grounded Q&A over local newsletter text files.

## Folder layout

- `newsletter_rag/ingestion`: document loading and chunking
- `newsletter_rag/vectorstore`: ChromaDB storage/retrieval
- `newsletter_rag/rag`: retrieval/generation pipeline
- `newsletter_rag/scripts`: indexing, querying, and retrieval evaluation

## Prerequisites

- Ollama installed and running
- Models pulled in Ollama:
  - `ollama pull nomic-embed-text`
  - `ollama pull llama3.2:3b`
- First query run downloads the cross-encoder reranker weights (`cross-encoder/ms-marco-MiniLM-L-6-v2`, via `sentence-transformers`). Requires network once unless cached.

## Setup with uv

From repo root:

1. Sync dependencies:
   - `uv sync`
2. Configure environment:
   - `cp newsletter_rag/.env.example newsletter_rag/.env`
   - Edit `newsletter_rag/.env` if you want non-default model names or Chroma path.

## Run commands with uv

From repo root:

- Index newsletters:
  - `uv run python -m newsletter_rag.scripts.index_newsletters --corpus-dir ./turing_newsletter`
- Ask one question:
  - `uv run python -m newsletter_rag.scripts.ask_newsletters --question "What is a token?"`
- Interactive Q&A:
  - `uv run python -m newsletter_rag.scripts.ask_newsletters`
- Retrieval regression check:
  - `uv run python -m newsletter_rag.scripts.eval_retrieval`

## Notes

- Indexing uses deterministic chunk IDs so re-running indexing updates rows instead of duplicating.
- Responses include source references from retrieved chunks.

## Retrieval

- **Wide recall:** Chroma returns up to `TOP_K_RETRIEVE` chunks (default 30) from vector similarity.
- **Rerank:** A cross-encoder scores `(question, chunk)` pairs and keeps the top `TOP_K` / `TOP_K_GENERATE` (default 5) for the LLM. Set `USE_RERANKER=0` to fall back to vector order plus a small lexical tie-break.
- Source lines in the CLI may show both vector `distance` and `rerank` score when reranking is enabled.
