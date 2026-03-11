"""
search.py - Query the SQLite embedding database built by index.py.

Fixes over v1:
  - Reads from SQLite instead of pickle (safe, no arbitrary code execution)
  - max_similarity is now the PRIMARY ranking key (avg_similarity is secondary)
    → a document with one very strong chunk ranks above a diluted long document
  - Model is loaded once and reused (fast for repeated queries in the same session)
  - Displays breadcrumb and notebook in results so you know exactly WHERE a
    match lives inside a note

Usage:
    python search.py "your query" [--db DB_PATH] [--top N] [--show-chunks]

Defaults:
    --db          : embeddings.db
    --top         : 5
    --show-chunks : False
"""

import os
import sys
import argparse
import sqlite3
import textwrap
import numpy as np
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------

def open_db(db_path: str) -> sqlite3.Connection:
    if not os.path.exists(db_path):
        print(f"[ERROR] Database not found: '{db_path}'", file=sys.stderr)
        print("        Run index.py first to build the database.", file=sys.stderr)
        sys.exit(1)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def load_meta(conn: sqlite3.Connection) -> dict:
    rows = conn.execute("SELECT key, value FROM meta").fetchall()
    return {r["key"]: r["value"] for r in rows}


def load_chunks(conn: sqlite3.Connection):
    """
    Load all chunks joined with their file info.
    Returns (embeddings_matrix: np.ndarray, records: list[dict])
    """
    rows = conn.execute("""
        SELECT
            c.id, c.chunk_index, c.total_chunks,
            c.note_title, c.breadcrumb, c.notebook,
            c.text_preview, c.embedding,
            f.file_path, f.rel_path
        FROM chunks c
        JOIN files  f ON f.id = c.file_id
        ORDER BY f.id, c.chunk_index
    """).fetchall()

    if not rows:
        return np.empty((0, 384), dtype=np.float32), []

    records    = []
    embeddings = []

    for row in rows:
        emb = np.frombuffer(row["embedding"], dtype=np.float32).copy()
        embeddings.append(emb)
        records.append({
            "file_path":    row["file_path"],
            "rel_path":     row["rel_path"],
            "chunk_index":  row["chunk_index"],
            "total_chunks": row["total_chunks"],
            "note_title":   row["note_title"] or "",
            "breadcrumb":   row["breadcrumb"] or "",
            "notebook":     row["notebook"] or "",
            "text_preview": row["text_preview"] or "",
        })

    return np.vstack(embeddings), records


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search(query: str, embeddings: np.ndarray, records: list[dict],
           model: SentenceTransformer, top_n: int) -> list[dict]:
    """
    Embed *query*, compute cosine similarity against all chunks, then
    aggregate per document.

    Ranking keys (primary → secondary):
        1. max_similarity  — best single-chunk match in the document
        2. avg_similarity  — average across all chunks (tiebreaker)

    This prevents long, broadly-relevant documents from burying short,
    highly-specific notes.
    """
    if embeddings.shape[0] == 0:
        return []

    query_emb  = model.encode([query], convert_to_numpy=True)        # (1, dim)
    chunk_sims = cosine_similarity(query_emb, embeddings)[0]         # (N,)

    # Group by file
    file_sims:    dict[str, list[float]] = defaultdict(list)
    file_details: dict[str, list[dict]]  = defaultdict(list)
    file_meta:    dict[str, dict]        = {}

    for sim, rec in zip(chunk_sims, records):
        fp = rec["file_path"]
        file_sims[fp].append(float(sim))
        file_details[fp].append({
            "chunk_index":  rec["chunk_index"],
            "total_chunks": rec["total_chunks"],
            "breadcrumb":   rec["breadcrumb"],
            "similarity":   float(sim),
            "text_preview": rec["text_preview"],
        })
        if fp not in file_meta:
            file_meta[fp] = {
                "rel_path":   rec["rel_path"],
                "note_title": rec["note_title"],
                "notebook":   rec["notebook"],
            }

    results = []
    for fp, sims in file_sims.items():
        meta = file_meta[fp]
        results.append({
            "file_path":      fp,
            "rel_path":       meta["rel_path"],
            "note_title":     meta["note_title"],
            "notebook":       meta["notebook"],
            "max_similarity": float(np.max(sims)),
            "avg_similarity": float(np.mean(sims)),
            "num_chunks":     len(sims),
            "chunk_details":  sorted(
                file_details[fp], key=lambda x: x["similarity"], reverse=True
            ),
        })

    # Primary sort: max_similarity  |  Secondary: avg_similarity
    results.sort(key=lambda x: (x["max_similarity"], x["avg_similarity"]), reverse=True)
    return results[:top_n]


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

SCORE_LABELS = {
    0.75: "strong match   ██████",
    0.50: "related        ████░░",
    0.00: "weak           ██░░░░",
}

def score_label(sim: float) -> str:
    for threshold, label in sorted(SCORE_LABELS.items(), reverse=True):
        if sim >= threshold:
            return label
    return "weak           ██░░░░"


def print_results(results: list[dict], show_chunks: bool, query: str) -> None:
    bar = "─" * 72

    print(f"\n{'═' * 72}")
    print(f"  Query : {query!r}")
    print(f"  Hits  : {len(results)}")
    print(f"{'═' * 72}\n")

    for rank, res in enumerate(results, start=1):
        label = score_label(res["max_similarity"])
        print(f"  #{rank}  {res['note_title'] or res['rel_path']}")
        print(f"       Path       : {res['rel_path']}")
        if res["notebook"]:
            print(f"       Notebook   : {res['notebook']}")
        print(f"       Max sim    : {res['max_similarity']:.4f}  ({label})")
        print(f"       Avg sim    : {res['avg_similarity']:.4f}")
        print(f"       Chunks     : {res['num_chunks']}")

        if show_chunks:
            print(f"\n       Top-3 matching chunks:")
            for chunk in res["chunk_details"][:3]:
                preview = textwrap.shorten(chunk["text_preview"], width=70, placeholder=" …")
                bc      = f"  [{chunk['breadcrumb']}]" if chunk["breadcrumb"] else ""
                print(f"         chunk {chunk['chunk_index']+1}/{chunk['total_chunks']}"
                      f"  sim={chunk['similarity']:.4f}{bc}")
                print(f"           {preview}")

        print(f"\n  {bar}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

# Module-level model cache so repeated calls in a long session don't reload
_model_cache: dict[str, SentenceTransformer] = {}

def get_model(name: str) -> SentenceTransformer:
    if name not in _model_cache:
        print(f"Loading model '{name}' …")
        _model_cache[name] = SentenceTransformer(name)
    return _model_cache[name]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search the local embedding database.")
    parser.add_argument("query",
                        help="Natural-language search query")
    parser.add_argument("--db", default="embeddings.db",
                        help="Path to the database built by index.py (default: embeddings.db)")
    parser.add_argument("--top", type=int, default=5,
                        help="Number of top results to return (default: 5)")
    parser.add_argument("--show-chunks", action="store_true",
                        help="Show top-3 matching chunk previews for each result")
    args = parser.parse_args()

    # Load DB ----------------------------------------------------------------
    print(f"Loading database '{args.db}' …")
    conn = open_db(args.db)
    meta = load_meta(conn)

    model_name = meta.get("model", "all-MiniLM-L6-v2")
    root       = meta.get("root", ".")

    embeddings, records = load_chunks(conn)
    conn.close()

    num_files = len({r["file_path"] for r in records})
    print(f"  {len(records)} chunks from {num_files} file(s)"
          f"  |  model: {model_name}"
          f"  |  chunk size: {meta.get('chunk_size', '?')} tokens\n")

    # Load model (cached) ----------------------------------------------------
    model = get_model(model_name)

    # Search -----------------------------------------------------------------
    results = search(args.query, embeddings, records, model, top_n=args.top)

    if not results:
        print("No results found.")
        sys.exit(0)

    print_results(results, show_chunks=args.show_chunks, query=args.query)

    best = results[0]
    print(f"  Best match: {best['note_title'] or best['rel_path']}"
          f"  (max sim = {best['max_similarity']:.4f})\n")
