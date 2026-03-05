"""
index.py - Recursively indexes documents from the current directory into an embedding database.

Usage:
    python index.py [directory] [--db DB_PATH] [--extensions ext1 ext2 ...]

Defaults:
    directory   : current working directory
    --db        : embeddings.db
    --extensions: .txt .md .py .rst .csv .json .html .xml
"""

import os
import sys
import json
import argparse
import pickle
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_DB_PATH = "embeddings.db"
DEFAULT_EXTENSIONS = {".txt", ".md", ".py", ".rst", ".csv", ".json", ".html", ".xml"}
CHUNK_TOKEN_SIZE = 256          # max tokens per chunk (model context = 256)
CHUNK_OVERLAP_TOKENS = 32       # overlap between consecutive chunks
MODEL_NAME = "all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def collect_files(root: str, extensions: set) -> list[str]:
    """Walk *root* and return every file whose suffix is in *extensions*."""
    matches = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if Path(name).suffix.lower() in extensions:
                matches.append(os.path.join(dirpath, name))
    return sorted(matches)


def read_file(path: str) -> str | None:
    """Read a file as UTF-8 text; return None on failure."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except Exception as exc:
        print(f"  [WARN] Cannot read {path}: {exc}", file=sys.stderr)
        return None


def tokenise(text: str, model) -> list[int]:
    """Tokenise *text* using the model's built-in tokeniser."""
    return model.tokenizer.encode(text, add_special_tokens=False)


def chunk_tokens(token_ids: list[int], chunk_size: int, overlap: int) -> list[list[int]]:
    """
    Split *token_ids* into overlapping windows of length *chunk_size*.
    The last chunk is padded to be non-empty.
    """
    chunks = []
    step = max(1, chunk_size - overlap)
    start = 0
    while start < len(token_ids):
        chunks.append(token_ids[start : start + chunk_size])
        start += step
    return chunks


def decode_tokens(token_ids: list[int], model) -> str:
    """Convert token ids back to a string."""
    return model.tokenizer.decode(token_ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Main indexing routine
# ---------------------------------------------------------------------------

def build_index(root: str, db_path: str, extensions: set) -> None:
    print(f"Loading model '{MODEL_NAME}' …")
    model = SentenceTransformer(MODEL_NAME)

    files = collect_files(root, extensions)
    if not files:
        print("No matching files found. Exiting.")
        return

    print(f"Found {len(files)} file(s) to index.\n")

    # Each entry in these lists corresponds to one chunk
    all_embeddings: list[np.ndarray] = []
    chunk_records: list[dict] = []   # {file_path, chunk_index, char_start, char_end, text_preview}

    for file_path in files:
        rel_path = os.path.relpath(file_path, root)
        text = read_file(file_path)
        if text is None or not text.strip():
            print(f"  [SKIP] {rel_path}  (empty or unreadable)")
            continue

        token_ids = tokenise(text, model)
        if not token_ids:
            print(f"  [SKIP] {rel_path}  (no tokens)")
            continue

        chunks = chunk_tokens(token_ids, CHUNK_TOKEN_SIZE, CHUNK_OVERLAP_TOKENS)
        chunk_texts = [decode_tokens(c, model) for c in chunks]

        # Encode all chunks for this file in one batch
        embeddings = model.encode(chunk_texts, show_progress_bar=False, convert_to_numpy=True)

        for idx, (chunk_text, emb) in enumerate(zip(chunk_texts, embeddings)):
            all_embeddings.append(emb)
            chunk_records.append({
                "file_path": file_path,        # absolute path
                "rel_path":  rel_path,          # relative to root
                "chunk_index": idx,
                "total_chunks": len(chunks),
                "token_count": len(chunks[idx]),
                "text_preview": chunk_text[:120].replace("\n", " "),
            })

        print(f"  [OK]   {rel_path}  →  {len(chunks)} chunk(s)")

    if not all_embeddings:
        print("\nNothing to save.")
        return

    embedding_matrix = np.vstack(all_embeddings)  # shape (N_chunks, embedding_dim)

    db = {
        "root":           root,
        "model":          MODEL_NAME,
        "chunk_size":     CHUNK_TOKEN_SIZE,
        "overlap":        CHUNK_OVERLAP_TOKENS,
        "embedding_dim":  embedding_matrix.shape[1],
        "num_chunks":     embedding_matrix.shape[0],
        "embeddings":     embedding_matrix,   # numpy array
        "chunks":         chunk_records,      # parallel list of metadata
    }

    with open(db_path, "wb") as fh:
        pickle.dump(db, fh, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = os.path.getsize(db_path) / 1_048_576
    print(f"\nSaved {embedding_matrix.shape[0]} chunk embeddings → '{db_path}' ({size_mb:.2f} MB)")
    print(f"Unique files indexed: {len({r['file_path'] for r in chunk_records})}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a local embedding database from documents.")
    parser.add_argument("directory", nargs="?", default=".",
                        help="Root directory to index (default: current directory)")
    parser.add_argument("--db", default=DEFAULT_DB_PATH,
                        help=f"Output database path (default: {DEFAULT_DB_PATH})")
    parser.add_argument("--extensions", nargs="+", default=None,
                        help="File extensions to include (default: txt md py rst csv json html xml)")
    args = parser.parse_args()

    root_dir  = os.path.abspath(args.directory)
    exts      = {e if e.startswith(".") else f".{e}" for e in args.extensions} \
                if args.extensions else DEFAULT_EXTENSIONS

    print(f"Indexing : {root_dir}")
    print(f"Database : {args.db}")
    print(f"Extensions: {', '.join(sorted(exts))}\n")

    build_index(root_dir, args.db, exts)