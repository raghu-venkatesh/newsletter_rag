from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path


@dataclass(frozen=True)
class NewsletterDocument:
    doc_id: str
    source_file: str
    title: str
    body: str
    ingested_at: str


def _title_from_text(path: Path, text: str) -> str:
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    return first_line[:120] if first_line else path.stem


def _doc_id(path: Path, text: str) -> str:
    digest = sha256(f"{path.name}:{text}".encode("utf-8")).hexdigest()
    return f"newsletter_{digest[:16]}"


def load_newsletters(corpus_dir: str | Path) -> list[NewsletterDocument]:
    base = Path(corpus_dir)
    if not base.exists():
        raise FileNotFoundError(f"Corpus directory does not exist: {base}")

    docs: list[NewsletterDocument] = []
    for path in sorted(base.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        docs.append(
            NewsletterDocument(
                doc_id=_doc_id(path, text),
                source_file=path.name,
                title=_title_from_text(path, text),
                body=text,
                ingested_at=datetime.now(timezone.utc).isoformat(),
            )
        )
    return docs
