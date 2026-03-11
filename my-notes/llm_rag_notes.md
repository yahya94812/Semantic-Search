# Research Notes — Large Language Models & Retrieval Augmented Generation

**Date:** February 2024

---

## Paper 1: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al., 2020)

**Source:** Facebook AI Research / NeurIPS 2020

### Key Idea

Combines a parametric memory (a seq2seq transformer) with a non-parametric memory
(a dense vector index over Wikipedia). The retriever fetches relevant documents
which are concatenated to the input before generation.

### Architecture

- **Retriever:** DPR (Dense Passage Retrieval) — dual encoder, FAISS index.
- **Generator:** BART-large fine-tuned with retrieved context.
- **Two variants:** RAG-Sequence (same doc across all tokens) and RAG-Token (different docs per token).

### Results

- State-of-the-art on Natural Questions, TriviaQA, WebQuestions.
- Outperforms parametric-only models by ~10% on open-domain QA.
- Hallucination rate visibly reduced when retrieved context is relevant.

### My Takeaways

The FAISS index lookup is the bottleneck at inference time (~50ms per query).
Token-level retrieval is theoretically elegant but computationally expensive.
Good baseline architecture for our own document search feature.

---

## Paper 2: "Improving Language Models by Retrieving from Trillions of Tokens" (Borgeaud et al., 2022)

**Source:** DeepMind / ICML 2022 — RETRO model

### Key Idea

RETRO retrieves at the chunk level (every 64 tokens) during both training AND inference,
from a 2-trillion token database. The retrieved neighbours are fused via cross-attention
at each transformer layer.

### Key Results

- A 7B RETRO model matches a 175B GPT-3 on several benchmarks.
- Retrieval database acts as an implicit knowledge store, reducing need for parameter count.
- Updating knowledge requires only re-indexing, not retraining.

### Limitations

- Retrieval latency during training is non-trivial.
- The chunked cross-attention mechanism adds architectural complexity.
- Database must be kept in sync with world knowledge.

### My Takeaways

Chunk-level retrieval during training is underexplored. This makes me think our
embedding chunk size choice matters enormously for downstream quality.

---

## Open Questions I Want to Explore

1. How does chunk overlap affect retrieval recall vs. precision?
2. Can we fine-tune MiniLM on domain-specific data (medical, legal) to improve retrieval?
3. Is BM25 hybrid search (keyword + dense) worth the added complexity?
4. What is the optimal aggregation strategy — mean, max, or learned attention over chunk scores?

## Related Tools to Evaluate

| Tool                       | Purpose                              |
|----------------------------|--------------------------------------|
| LlamaIndex                 | High-level RAG orchestration         |
| LangChain                  | Popular but heavyweight              |
| Chroma / Qdrant / Weaviate | Managed vector databases             |
| FAISS                      | CPU/GPU similarity search (by Meta)  |
