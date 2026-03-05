"""
search.py - Query the embedding database built by index.py.

Usage:
    python search.py "your search query" [--db DB_PATH] [--top N] [--show-chunks]

Defaults:
    --db          : embeddings.db
    --top         : 5
    --show-chunks : False
"""

import os
import sys
import argparse
import pickle
import textwrap
import numpy as np
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_db(db_path: str) -> dict:
    """Load the database written by index.py."""
    if not os.path.exists(db_path):
        print(f"[ERROR] Database not found: '{db_path}'", file=sys.stderr)
        print("        Run index.py first to build the database.", file=sys.stderr)
        sys.exit(1)

    with open(db_path, "rb") as fh:
        return pickle.load(fh)


def search(query: str, db: dict, model: SentenceTransformer, top_n: int) -> list[dict]:
    """
    Embed *query*, compute cosine similarity against every stored chunk,
    then aggregate per document by averaging chunk scores.

    Returns a list of result dicts sorted by descending average similarity.
    """
    query_emb = model.encode([query], convert_to_numpy=True)          # (1, dim)
    chunk_sims = cosine_similarity(query_emb, db["embeddings"])[0]    # (N_chunks,)

    # Group chunk similarities by file path
    file_chunk_sims: dict[str, list[float]] = defaultdict(list)
    file_chunk_details: dict[str, list[dict]] = defaultdict(list)

    for idx, (sim, record) in enumerate(zip(chunk_sims, db["chunks"])):
        fp = record["file_path"]
        file_chunk_sims[fp].append(float(sim))
        file_chunk_details[fp].append({
            "chunk_index":   record["chunk_index"],
            "total_chunks":  record["total_chunks"],
            "similarity":    float(sim),
            "text_preview":  record["text_preview"],
        })

    # Compute per-document average similarity
    results = []
    for fp, sims in file_chunk_sims.items():
        results.append({
            "file_path":       fp,
            "rel_path":        os.path.relpath(fp, db["root"]),
            "avg_similarity":  float(np.mean(sims)),
            "max_similarity":  float(np.max(sims)),
            "num_chunks":      len(sims),
            "chunk_details":   sorted(file_chunk_details[fp],
                                      key=lambda x: x["similarity"], reverse=True),
        })

    results.sort(key=lambda x: x["avg_similarity"], reverse=True)
    return results[:top_n]


def read_file_safe(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except Exception:
        return "[Could not read file]"


def print_results(results: list[dict], show_chunks: bool, query: str) -> None:
    sep = "─" * 72

    print(f"\n{'═' * 72}")
    print(f"  Query : {query!r}")
    print(f"  Hits  : {len(results)}")
    print(f"{'═' * 72}\n")

    for rank, res in enumerate(results, start=1):
        print(f"  #{rank}  {res['rel_path']}")
        print(f"       Avg similarity : {res['avg_similarity']:.4f}")
        print(f"       Max similarity : {res['max_similarity']:.4f}")
        print(f"       Chunks matched : {res['num_chunks']}")

        if show_chunks:
            print(f"\n       Top-3 matching chunks:")
            for chunk in res["chunk_details"][:3]:
                preview = textwrap.shorten(chunk["text_preview"], width=80, placeholder=" …")
                print(f"         [chunk {chunk['chunk_index']+1}/{chunk['total_chunks']}]"
                      f"  sim={chunk['similarity']:.4f}  {preview}")

        print(f"\n  {sep}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search the local embedding database.")
    parser.add_argument("query", help="Natural-language search query")
    parser.add_argument("--db", default="embeddings.db",
                        help="Path to the database built by index.py (default: embeddings.db)")
    parser.add_argument("--top", type=int, default=5,
                        help="Number of top documents to return (default: 5)")
    parser.add_argument("--show-chunks", action="store_true",
                        help="Show the top-3 matching chunk previews for each result")
    args = parser.parse_args()

    # Load DB ----------------------------------------------------------------
    print(f"Loading database '{args.db}' …")
    db = load_db(args.db)
    print(f"  {db['num_chunks']} chunks from {len({r['file_path'] for r in db['chunks']})} file(s)"
          f"  |  model: {db['model']}  |  chunk size: {db['chunk_size']} tokens")

    # Load model (must match what was used for indexing) ---------------------
    model_name = db.get("model", "all-MiniLM-L6-v2")
    print(f"Loading model '{model_name}' …\n")
    model = SentenceTransformer(model_name)

    # Search -----------------------------------------------------------------
    results = search(args.query, db, model, top_n=args.top)

    if not results:
        print("No results found.")
        sys.exit(0)

    print_results(results, show_chunks=args.show_chunks, query=args.query)

    # Offer to print the best matching file ----------------------------------
    best = results[0]
    print(f"  Best match: {best['rel_path']}  (avg sim = {best['avg_similarity']:.4f})\n")