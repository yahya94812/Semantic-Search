"""
index.py - Recursively indexes documents into a SQLite embedding database.

Fixes over v1:
  - SQLite storage instead of pickle (safe, portable, inspectable)
  - Markdown-aware chunking: splits by headings first, falls back to token
    chunking only when a section is too long
  - Rich metadata per chunk: note_title, breadcrumb, notebook (folder name)
  - Incremental re-indexing: skips files that haven't changed (via mtime + size)

Usage:
    python index.py [directory] [--db DB_PATH] [--extensions ext1 ext2 ...]

Defaults:
    directory    : current working directory
    --db         : embeddings.db
    --extensions : txt md py rst csv json html xml
"""

import os
import re
import sys
import json
import sqlite3
import argparse
import hashlib
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_DB_PATH   = "embeddings.db"
DEFAULT_EXTENSIONS = {".txt", ".md", ".py", ".rst", ".csv", ".json", ".html", ".xml"}
CHUNK_TOKEN_SIZE  = 256
CHUNK_OVERLAP_TOKENS = 32
MODEL_NAME        = "all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------

def open_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS meta (
            key   TEXT PRIMARY KEY,
            value TEXT
        );

        CREATE TABLE IF NOT EXISTS files (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT UNIQUE NOT NULL,
            rel_path  TEXT NOT NULL,
            mtime     REAL NOT NULL,
            fsize     INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS chunks (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id       INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
            chunk_index   INTEGER NOT NULL,
            total_chunks  INTEGER NOT NULL,
            note_title    TEXT,
            breadcrumb    TEXT,
            notebook      TEXT,
            text_preview  TEXT,
            embedding     BLOB NOT NULL
        );
    """)
    conn.commit()
    return conn


def save_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)", (key, value))


def file_needs_reindex(conn: sqlite3.Connection, file_path: str,
                       mtime: float, fsize: int) -> bool:
    row = conn.execute(
        "SELECT mtime, fsize FROM files WHERE file_path=?", (file_path,)
    ).fetchone()
    if row is None:
        return True
    return row["mtime"] != mtime or row["fsize"] != fsize


def delete_file_chunks(conn: sqlite3.Connection, file_path: str) -> None:
    conn.execute("DELETE FROM files WHERE file_path=?", (file_path,))


def insert_file(conn: sqlite3.Connection, file_path: str, rel_path: str,
                mtime: float, fsize: int) -> int:
    cur = conn.execute(
        "INSERT OR REPLACE INTO files(file_path, rel_path, mtime, fsize) VALUES(?,?,?,?)",
        (file_path, rel_path, mtime, fsize)
    )
    return cur.lastrowid


def insert_chunk(conn: sqlite3.Connection, file_id: int, chunk_index: int,
                 total_chunks: int, note_title: str, breadcrumb: str,
                 notebook: str, text_preview: str, embedding: np.ndarray) -> None:
    conn.execute(
        """INSERT INTO chunks
           (file_id, chunk_index, total_chunks, note_title, breadcrumb,
            notebook, text_preview, embedding)
           VALUES(?,?,?,?,?,?,?,?)""",
        (file_id, chunk_index, total_chunks, note_title, breadcrumb,
         notebook, text_preview, embedding.astype(np.float32).tobytes())
    )


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def collect_files(root: str, extensions: set) -> list[str]:
    matches = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if Path(name).suffix.lower() in extensions:
                matches.append(os.path.join(dirpath, name))
    return sorted(matches)


def read_file(path: str) -> str | None:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except Exception as exc:
        print(f"  [WARN] Cannot read {path}: {exc}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Markdown-aware chunking
# ---------------------------------------------------------------------------

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)", re.MULTILINE)


def parse_markdown_sections(text: str) -> list[dict]:
    """
    Split *text* into sections based on Markdown headings.
    Returns a list of dicts:
        {
            "breadcrumb": "Title > H2 > H3",
            "content":    "section body text",
            "level":      3,
        }
    Falls back to a single section for non-Markdown files.
    """
    matches = list(HEADING_RE.finditer(text))

    if not matches:
        # No headings — treat the whole text as one section
        return [{"breadcrumb": "", "content": text.strip(), "level": 0}]

    sections = []
    heading_stack: list[tuple[int, str]] = []  # (level, title)

    for i, match in enumerate(matches):
        level = len(match.group(1))   # number of # symbols
        title = match.group(2).strip()

        # Build heading stack (breadcrumb)
        # Pop anything at same or deeper level
        while heading_stack and heading_stack[-1][0] >= level:
            heading_stack.pop()
        heading_stack.append((level, title))

        breadcrumb = " > ".join(t for _, t in heading_stack)

        # Section body = text between this heading and the next
        body_start = match.end()
        body_end   = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body       = text[body_start:body_end].strip()

        if body:
            sections.append({
                "breadcrumb": breadcrumb,
                "content":    body,
                "level":      level,
            })

    return sections if sections else [{"breadcrumb": "", "content": text.strip(), "level": 0}]


def tokenise(text: str, model) -> list[int]:
    return model.tokenizer.encode(text, add_special_tokens=False)


def decode_tokens(token_ids: list[int], model) -> str:
    return model.tokenizer.decode(token_ids, skip_special_tokens=True)


def token_chunks(text: str, model,
                 chunk_size: int = CHUNK_TOKEN_SIZE,
                 overlap: int = CHUNK_OVERLAP_TOKENS) -> list[str]:
    """Split *text* into overlapping token-based chunks."""
    token_ids = tokenise(text, model)
    if not token_ids:
        return []
    step   = max(1, chunk_size - overlap)
    chunks = []
    start  = 0
    while start < len(token_ids):
        chunk_ids = token_ids[start: start + chunk_size]
        chunks.append(decode_tokens(chunk_ids, model))
        start += step
    return chunks


def make_chunks(text: str, model, note_title: str,
                notebook: str, is_markdown: bool) -> list[dict]:
    """
    Return a list of chunk dicts ready for embedding:
        {
            "text":       full text to embed (breadcrumb + content),
            "breadcrumb": heading path,
            "note_title": note title,
            "notebook":   parent folder,
            "preview":    first 120 chars of content,
        }
    Strategy:
        1. For .md files: split by headings first.
        2. If a section exceeds CHUNK_TOKEN_SIZE tokens, further split it
           using overlapping token windows.
        3. For non-Markdown files: go straight to token chunking.
    """
    chunks = []

    if is_markdown:
        sections = parse_markdown_sections(text)
    else:
        sections = [{"breadcrumb": "", "content": text.strip(), "level": 0}]

    for section in sections:
        bc      = section["breadcrumb"]
        content = section["content"]
        if not content:
            continue

        # Check token length of this section
        tok_count = len(tokenise(content, model))

        if tok_count <= CHUNK_TOKEN_SIZE:
            # Section fits in one chunk — embed breadcrumb + content together
            embed_text = f"{bc}\n\n{content}" if bc else content
            chunks.append({
                "text":       embed_text,
                "breadcrumb": bc,
                "note_title": note_title,
                "notebook":   notebook,
                "preview":    content[:120].replace("\n", " "),
            })
        else:
            # Section too long — split into token windows
            sub_chunks = token_chunks(content, model)
            for sub in sub_chunks:
                embed_text = f"{bc}\n\n{sub}" if bc else sub
                chunks.append({
                    "text":       embed_text,
                    "breadcrumb": bc,
                    "note_title": note_title,
                    "notebook":   notebook,
                    "preview":    sub[:120].replace("\n", " "),
                })

    return chunks


# ---------------------------------------------------------------------------
# Main indexing routine
# ---------------------------------------------------------------------------

def build_index(root: str, db_path: str, extensions: set) -> None:
    print(f"Loading model '{MODEL_NAME}' …")
    model = SentenceTransformer(MODEL_NAME)

    conn = open_db(db_path)
    save_meta(conn, "model",      MODEL_NAME)
    save_meta(conn, "chunk_size", str(CHUNK_TOKEN_SIZE))
    save_meta(conn, "overlap",    str(CHUNK_OVERLAP_TOKENS))
    save_meta(conn, "root",       root)

    files = collect_files(root, extensions)
    if not files:
        print("No matching files found. Exiting.")
        conn.close()
        return

    print(f"Found {len(files)} file(s) to index.\n")

    indexed = skipped = 0

    for file_path in files:
        rel_path = os.path.relpath(file_path, root)
        stat     = os.stat(file_path)
        mtime    = stat.st_mtime
        fsize    = stat.st_size

        if not file_needs_reindex(conn, file_path, mtime, fsize):
            print(f"  [--]   {rel_path}  (unchanged, skipped)")
            skipped += 1
            continue

        text = read_file(file_path)
        if not text or not text.strip():
            print(f"  [SKIP] {rel_path}  (empty or unreadable)")
            continue

        # Derive metadata
        suffix      = Path(file_path).suffix.lower()
        is_markdown = suffix == ".md"
        note_title  = Path(file_path).stem          # filename without extension
        notebook    = Path(rel_path).parent.name or Path(root).name  # parent folder

        chunk_list = make_chunks(text, model, note_title, notebook, is_markdown)
        if not chunk_list:
            print(f"  [SKIP] {rel_path}  (no chunks produced)")
            continue

        # Embed all chunks for this file in one batch
        texts      = [c["text"] for c in chunk_list]
        embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

        # Persist — delete old entry first (CASCADE removes old chunks)
        delete_file_chunks(conn, file_path)
        file_id = insert_file(conn, file_path, rel_path, mtime, fsize)

        for idx, (chunk, emb) in enumerate(zip(chunk_list, embeddings)):
            insert_chunk(
                conn, file_id,
                chunk_index   = idx,
                total_chunks  = len(chunk_list),
                note_title    = chunk["note_title"],
                breadcrumb    = chunk["breadcrumb"],
                notebook      = chunk["notebook"],
                text_preview  = chunk["preview"],
                embedding     = emb,
            )

        conn.commit()
        print(f"  [OK]   {rel_path}  →  {len(chunk_list)} chunk(s)")
        indexed += 1

    # Summary stats
    total_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    total_files  = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
    conn.close()

    size_mb = os.path.getsize(db_path) / 1_048_576
    print(f"\nDone. {indexed} indexed, {skipped} skipped.")
    print(f"Database: '{db_path}' ({size_mb:.2f} MB) — "
          f"{total_chunks} chunks across {total_files} file(s)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a local SQLite embedding database from documents."
    )
    parser.add_argument("directory", nargs="?", default=".",
                        help="Root directory to index (default: current directory)")
    parser.add_argument("--db", default=DEFAULT_DB_PATH,
                        help=f"Output database path (default: {DEFAULT_DB_PATH})")
    parser.add_argument("--extensions", nargs="+", default=None,
                        help="File extensions to include (default: txt md py rst csv json html xml)")
    args = parser.parse_args()

    root_dir = os.path.abspath(args.directory)
    exts     = (
        {e if e.startswith(".") else f".{e}" for e in args.extensions}
        if args.extensions else DEFAULT_EXTENSIONS
    )

    print(f"Indexing  : {root_dir}")
    print(f"Database  : {args.db}")
    print(f"Extensions: {', '.join(sorted(exts))}\n")

    build_index(root_dir, args.db, exts)
