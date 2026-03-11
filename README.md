# Semantic Search — Documentation

Local semantic search for documents and notes. Finds results by **meaning**, not keywords. Runs entirely on your machine with no API keys required.

---

## Installation

```bash
pip install sentence-transformers scikit-learn numpy
```

---

## Quick Start

```bash
# 1. Index your notes folder
python index.py ./my-notes --db notes.db

# 2. Search
python search.py "database outage last month" --db notes.db
```

---

## index.py

Scans a folder recursively, chunks each document, generates embeddings, and saves everything to a SQLite database.

### Usage
```bash
python index.py [directory] [--db DB_PATH] [--extensions ext1 ext2 ...]
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `directory` | `.` (current dir) | Root folder to index |
| `--db` | `embeddings.db` | Output SQLite database path |
| `--extensions` | `txt md py rst csv json html xml` | File types to include |

### Example
```bash
python index.py ./notes --db notes.db --extensions md txt
```

### How it chunks
- **Markdown files** — splits by headings (`#`, `##`, `###` …) first. Each section becomes one chunk with its breadcrumb path attached (e.g. `Setup > Installation`). Sections longer than 256 tokens are further split with overlapping token windows.
- **All other files** — split directly into overlapping 256-token windows (32-token overlap).

### Incremental indexing
Re-running `index.py` skips files that haven't changed (checked via modification time + file size). Only new or edited files are re-embedded.

---

## search.py

Embeds your query and retrieves the most semantically similar documents from the database.

### Usage
```bash
python search.py "your query" [--db DB_PATH] [--top N] [--show-chunks]
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `query` | *(required)* | Natural-language search query |
| `--db` | `embeddings.db` | Database built by `index.py` |
| `--top` | `5` | Number of results to return |
| `--show-chunks` | off | Show top-3 matching chunk previews i.e the sub headings per result |

### Examples
```bash
python search.py "startup meeting in Berlin"
python search.py "database outage" --db notes.db --top 10 --show-chunks
```

### Ranking
Results are ranked by **max similarity** (the best single chunk match in the document) as the primary key, with **avg similarity** as tiebreaker. This ensures a short, highly specific note ranks above a long document that only partially matches.

### Similarity score guide

| Score | Meaning |
|---|---|
| `0.75+` | Strong match |
| `0.50 – 0.74` | Related, worth checking |
| `below 0.50` | Likely irrelevant |

---

## Database Schema

SQLite database with 3 tables:

- **`meta`** — stores model name, chunk size, root directory
- **`files`** — one row per indexed file (path, mtime, size)
- **`chunks`** — one row per chunk (embedding, breadcrumb, note_title, notebook, preview)

---

## Model

Uses `all-MiniLM-L6-v2` — a lightweight, fast sentence embedding model optimised for semantic similarity. Runs fully offline after the first download (~80 MB).

---

## Limitations

- Not a keyword search — use `grep` for exact phrases or proper nouns
- English-first (multilingual quality may vary)
- Large folders take a few minutes to index; search is always instant
- LLM-based reranking is not included — results are purely embedding-based