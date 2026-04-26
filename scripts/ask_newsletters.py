from __future__ import annotations

import argparse
import json

from newsletter_rag.rag.pipeline import NewsletterRagPipeline
from newsletter_rag.rag.settings import load_settings


def ask_once(question: str, top_k: int | None = None) -> dict:
    settings = load_settings()
    pipeline = NewsletterRagPipeline(settings)
    answer = pipeline.ask(question=question, top_k=top_k)
    return pipeline.to_dict(answer)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask questions over the newsletter corpus")
    parser.add_argument("--question", default=None, help="Single question to ask")
    parser.add_argument("--top-k", type=int, default=None, help="Override retrieval top-k")
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    args = parser.parse_args()

    if args.question:
        result = ask_once(args.question, top_k=args.top_k)
        if args.json:
            print(json.dumps(result, indent=2))
            return
        print(f"\nQ: {result['question']}\n")
        print(result["answer"])
        print("\nSources:")
        for source in result["sources"]:
            rs = source.get("rerank_score")
            rs_part = f", rerank={rs:.4f}" if isinstance(rs, (int, float)) else ""
            print(
                f"- [{source['rank']}] {source['source_file']} "
                f"({source['chunk_id']}, distance={source['distance']:.4f}{rs_part})"
            )
        return

    print("Interactive mode (Ctrl+C to exit)")
    while True:
        q = input("\nQuestion> ").strip()
        if not q:
            continue
        result = ask_once(q, top_k=args.top_k)
        print("\nAnswer:")
        print(result["answer"])
        print("\nSources:")
        for source in result["sources"]:
            rs = source.get("rerank_score")
            rs_part = f", rerank={rs:.4f}" if isinstance(rs, (int, float)) else ""
            print(
                f"- [{source['rank']}] {source['source_file']} "
                f"({source['chunk_id']}, distance={source['distance']:.4f}{rs_part})"
            )


if __name__ == "__main__":
    main()
