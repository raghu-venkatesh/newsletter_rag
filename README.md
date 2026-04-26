# Newsletter RAG (uv-first)

This subproject implements newsletter ingestion, ChromaDB indexing, and grounded Q&A over local newsletter text files.

This project originated from a **[deepatlas.ai](https://deepatlas.ai)** workshop exercise (RAG over a private newsletter corpus).

## Folder layout

- `newsletter_rag/ingestion`: document loading and chunking
- `newsletter_rag/vectorstore`: ChromaDB storage/retrieval
- `newsletter_rag/rag`: retrieval/generation pipeline
- `newsletter_rag/scripts`: indexing, querying, and retrieval evaluation

## Prerequisites

- Ollama installed and running
- Models pulled in Ollama:
  - `ollama pull nomic-embed-text`
  - `ollama pull gemma4:e4b-it-q4_K_M`
- First query run downloads the cross-encoder reranker weights (`cross-encoder/ms-marco-MiniLM-L-6-v2`, via `sentence-transformers`). Requires network once unless cached.

## Setup with uv

1. Sync dependencies (run this from the directory that contains the workspace `pyproject.toml`, e.g. the parent `deep_atlas_mli` folder if this package lives there):

   - `uv sync`

2. Copy the environment template and edit local values:

   - `cp newsletter_rag/.env.example newsletter_rag/.env`

   The app loads **`newsletter_rag/.env`** automatically (via `python-dotenv`). That file is **gitignored**; keep machine-specific paths and any secrets there, not in git.

### Getting the app working (`.env` context)

**Working directory matters:** paths like `CHROMA_PERSIST_DIR` and `NEWSLETTER_CORPUS_DIR` are **relative to your shell’s current working directory** when you run `uv run`, not relative to the `.env` file.

- If you run commands from the **parent repo root** (e.g. `deep_atlas_mli/`), the defaults in `.env.example` match: `CHROMA_PERSIST_DIR=./newsletter_rag/data/chroma`, `NEWSLETTER_CORPUS_DIR=./turing_newsletter` (or change the latter to `./email_txts`, `./corpus`, etc.).
- If you run only from **`newsletter_rag/`** as cwd, set e.g. `CHROMA_PERSIST_DIR=./data/chroma` and `NEWSLETTER_CORPUS_DIR=./corpus` (or `../email_txts`) so Chroma and your `.txt` corpus resolve correctly.

**Minimal checklist:**

1. **Ollama** running locally; pull the models named in `.env` (`OLLAMA_EMBED_MODEL`, `OLLAMA_GENERATE_MODEL`).
2. **Corpus:** a folder of `.txt` files; point `NEWSLETTER_CORPUS_DIR` at it (or pass `--corpus-dir` on the index command, which overrides the env default for that run).
3. **Index once** (builds embeddings + Chroma under `CHROMA_PERSIST_DIR`); stale vectors are pruned by default on reindex.
4. **Ask** — queries use the same `.env` for models, retrieval width (`TOP_K_RETRIEVE`), reranker (`USE_RERANKER`, `RERANKER_MODEL`), and chunk sizes.

After editing `.env`, rerun `uv run …` from the same cwd you intend to use so paths stay consistent.

## Run commands with uv

From the same directory you used for `uv sync` (typically the parent repo root):

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
