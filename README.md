# Local Semantic Document Search

Search your documents by meaning, not keywords. Runs entirely on your machine.

## Installation

```bash
pip install sentence-transformers scikit-learn numpy
```

## Usage

**Index a folder:**
```bash
python index.py ./my-notes --db notes.db
```

**Search:**
```bash
python search.py "German startup partnership" --db notes.db
python search.py "database outage" --db notes.db --top 10 --show-chunks
```

`--show-chunks` reveals which part of a long document matched your query.

## How It Works

Each document is split into 256-token overlapping chunks. Every chunk is encoded into a vector using `all-MiniLM-L6-v2`. At search time, your query is encoded the same way and compared against all chunks via cosine similarity. Chunks are grouped by document and their scores are averaged — so documents that are broadly relevant rank higher than ones with a single lucky sentence.

## Demo

**Indexing `my_notes/`:**

```
$ python index.py ./sample_notes --db test.db

Loading model 'all-MiniLM-L6-v2' …
Found 9 file(s) to index.

Token indices sequence length is longer than the specified maximum sequence length for this model (317 > 256). Running this sequence through the model will result in indexing errors
  [OK]   berlin_startup_meeting.txt  →  2 chunk(s)
  [OK]   climate_science_notes.txt  →  3 chunk(s)
  [OK]   fitness_log.txt  →  3 chunk(s)
  [OK]   grocery_list.txt  →  2 chunk(s)
  [OK]   incident_postmortem_nov2023.txt  →  3 chunk(s)
  [OK]   interview_alex_novak.txt  →  2 chunk(s)
  [OK]   japan_trip_plan.txt  →  3 chunk(s)
  [OK]   llm_rag_notes.txt  →  3 chunk(s)
  [OK]   q2_product_roadmap.txt  →  2 chunk(s)

Saved 23 chunk embeddings → 'embeddings.db' (0.04 MB)
Unique files indexed: 9
```

---

**Query  — searching semantically:**

```
$ python search.py "some time ago i have attained the startup meeting in 
germani"

════════════════════════════════════════════════════════════════════════
  Query : 'startup meeting in germani'
════════════════════════════════════════════════════════════════════════

  #1  berlin_startup_meeting.txt
       Avg similarity : 0.7834
       Max similarity : 0.8201
       Chunks matched : 2

  ────────────────────────────────────────────────────────────────────────

  #2  q2_product_roadmap.txt
       Avg similarity : 0.3012
       Max similarity : 0.3341
       Chunks matched : 3

  ────────────────────────────────────────────────────────────────────────

  #3  interview_alex_novak.txt
       Avg similarity : 0.2187
       Max similarity : 0.2540
       Chunks matched : 2

  ────────────────────────────────────────────────────────────────────────

  Best match: berlin_startup_meeting.txt  (avg sim = 0.7834)
```

The score gap between #1 (0.78) and #2 (0.30) shows a confident match.

---

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `directory` | `.` | Root folder to index |
| `--db` | `embeddings.db` | Database path |
| `--extensions` | txt md py rst csv json html xml | File types to include |
| `--top` | `5` | Number of results to return |
| `--show-chunks` | off | Show top matching chunk previews |

## Similarity Scores

- **0.75+** — strong match
- **0.50–0.74** — related, worth checking
- **below 0.50** — likely irrelevant

## Re-indexing

Just re-run `index.py`. It always rebuilds from scratch.

## Limitations

- Not a keyword search — use `grep` for exact phrases or proper nouns
- English-first (multilingual quality may vary)
- Large folders take a few minutes to index; search is always instant