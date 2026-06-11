"""
Command-line search interface for the multi-modal vector store.

Usage:
    python -m scripts.search "your natural language query"
    python -m scripts.search "your query" --k 10
    python -m scripts.search "your query" --modality text

Currently only text is ingested; audio/video modality filters will become
useful in Milestone 2.
"""

import argparse
import sys

from src.embeddings import embed_text
from src.vectorstore import query, count


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Search the multi-modal vector store.",
    )
    parser.add_argument("query", help="Natural language query string.")
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of top results to return (default: 5).",
    )
    parser.add_argument(
        "--modality",
        choices=["text", "audio", "video"],
        default=None,
        help="Restrict results to a single modality.",
    )
    args = parser.parse_args()

    total = count()
    if total == 0:
        print("[search] ChromaDB collection is empty. Ingest some files first.")
        return 1

    print(f"[search] Collection size: {total} docs")
    print(f"[search] Query: {args.query!r}")
    if args.modality:
        print(f"[search] Modality filter: {args.modality}")
    print()

    q_emb = embed_text(args.query).tolist()
    where = {"modality": args.modality} if args.modality else None
    results = query(q_emb, n_results=args.k, where=where)

    ids = results["ids"][0]
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    if not ids:
        print("[search] No matches.")
        return 0

    for i, (doc_id, doc, meta, dist) in enumerate(zip(ids, docs, metas, dists), start=1):
        source = meta.get("source", "?")
        modality = meta.get("modality", "?")
        snippet = doc.strip().replace("\n", " ")
        if len(snippet) > 200:
            snippet = snippet[:200] + "..."
        print(f"#{i}  [{modality}] {source}  (distance: {dist:.4f})")
        print(f"    {snippet}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
