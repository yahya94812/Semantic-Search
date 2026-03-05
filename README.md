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

**Indexing `sample_notes/`:**

```
$ python index.py ./sample_notes --db test.db

Loading model 'all-MiniLM-L6-v2' …
Found 9 file(s) to index.

  [OK]   personal/fitness_log.txt            →  3 chunk(s)
  [OK]   personal/grocery_list.txt           →  1 chunk(s)
  [OK]   personal/japan_trip_plan.txt        →  2 chunk(s)
  [OK]   research/climate_science_notes.txt  →  3 chunk(s)
  [OK]   research/llm_rag_notes.txt          →  3 chunk(s)
  [OK]   work/berlin_startup_meeting.txt     →  2 chunk(s)
  [OK]   work/incident_postmortem_nov2023.txt →  4 chunk(s)
  [OK]   work/interview_alex_novak.txt       →  2 chunk(s)
  [OK]   work/q2_product_roadmap.txt         →  3 chunk(s)

Saved 23 chunk embeddings → 'test.db' (0.87 MB)
Unique files indexed: 9
```

---

**Query 1 — topic the document covers verbatim:**

```
$ python search.py "German startup partnership deal" --db test.db --top 3

════════════════════════════════════════════════════════════════════════
  Query : 'German startup partnership deal'
  Hits  : 3
════════════════════════════════════════════════════════════════════════

  #1  work/berlin_startup_meeting.txt
       Avg similarity : 0.7834
       Max similarity : 0.8201
       Chunks matched : 2

  ────────────────────────────────────────────────────────────────────────

  #2  work/q2_product_roadmap.txt
       Avg similarity : 0.3012
       Max similarity : 0.3341
       Chunks matched : 3

  ────────────────────────────────────────────────────────────────────────

  #3  work/interview_alex_novak.txt
       Avg similarity : 0.2187
       Max similarity : 0.2540
       Chunks matched : 2

  ────────────────────────────────────────────────────────────────────────

  Best match: work/berlin_startup_meeting.txt  (avg sim = 0.7834)
```

The score gap between #1 (0.78) and #2 (0.30) shows a confident match.

---

**Query 2 — semantic match with no shared words:**

```
$ python search.py "slow database query brought down production" --db test.db --top 3 --show-chunks

════════════════════════════════════════════════════════════════════════
  Query : 'slow database query brought down production'
  Hits  : 3
════════════════════════════════════════════════════════════════════════

  #1  work/incident_postmortem_nov2023.txt
       Avg similarity : 0.6921
       Max similarity : 0.7455
       Chunks matched : 4

       Top-3 matching chunks:
         [chunk 1/4]  sim=0.7455  "Root Cause: A new analytics query deployed in the
                                   13:45 UTC release queried the orders table without …"
         [chunk 2/4]  sim=0.7102  "Timeline: 14:02 UTC — Automated monitoring alerts
                                   fire: API error rate spikes to 94% …"
         [chunk 3/4]  sim=0.6230  "Action Items: Add index on orders.status column.
                                   Increase RDS connection pool from 50 to 200 …"

  ────────────────────────────────────────────────────────────────────────

  #2  work/q2_product_roadmap.txt
       Avg similarity : 0.2543
       Max similarity : 0.2811
       Chunks matched : 3

  ────────────────────────────────────────────────────────────────────────

  Best match: work/incident_postmortem_nov2023.txt  (avg sim = 0.6921)
```

The query contained none of the words in the document title — semantic similarity found it anyway.

---

**Query 3 — personal/unrelated topic stays isolated:**

```
$ python search.py "what to buy at the supermarket" --db test.db --top 3

  #1  personal/grocery_list.txt
       Avg similarity : 0.6103
       Max similarity : 0.6103
       Chunks matched : 1

  #2  personal/japan_trip_plan.txt
       Avg similarity : 0.2874
       Max similarity : 0.3012
       Chunks matched : 2

  #3  personal/fitness_log.txt
       Avg similarity : 0.2341
       Max similarity : 0.2601
       Chunks matched : 3

  Best match: personal/grocery_list.txt  (avg sim = 0.6103)
```

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