# Changelog

## 0.2.0

Corpus, retrieval, verifier, evals and interfaces all expanded roughly tenfold; numpy remains the
only required runtime dependency and the whole test/eval suite still runs offline.

### Corpus
- `scripts/fetch_corpus.py`: reproducible, stdlib-only fetcher for ethereum/EIPs, ethereum/ERCs,
  ethereum/consensus-specs and a curated set of flashbots-docs pages. Writes `data/MANIFEST.json`
  (id, number, title, status, category, type, created, requires, sha256, bytes).
- Corpus grows from 23 EIPs / 497 chunks to 901 documents / ~17k chunks (11.3 MB).
- `data/LICENSE` records the CC0 / MIT provenance.

### Ingest
- Chunks carry front-matter metadata (status, category, type, requires, created, number) and a
  `kind` (prose / code / table).
- Table-aware chunking: long tables are split by rows with the header repeated on every piece.
- Code-block-aware chunking: fenced blocks are windowed by lines, never mid-line; a single giant
  line (bytecode) falls back to word windowing.
- MDX cleanup for flashbots pages (import/export lines, bare JSX tags, HTML comments).
- `CitationGraph` built from `requires` front matter and inline `EIP-N` mentions, with
  `related(doc, depth)`.

### Retrieval
- BM25 rewritten as an inverted index with BM25F-style title/section field boosting.
- Hashed "dense" baseline stored as a numpy CSC matrix (word + char-trigram features, tf-idf,
  cosine); the `Embedder` protocol and `SentenceTransformerEmbedder` are unchanged.
- Weighted RRF, EIP-number detection in queries (`boost` / `restrict` / `off`), and optional
  graph-walk expansion of results along `requires` edges.
- `LogisticReranker`: 20 interpretable cross features, numpy logistic regression, fit on the train
  split only; weights shipped in `evals/artifacts/reranker.json` for `ethrag ask --rerank`.

### Verifier
- Hex/decimal, `2**n`/value and unit (`32 ETH` = `32000000000` gwei) equivalences, with the
  match kind recorded per claim.
- Negation detection against the best-matching source sentence (answer negated, source not).
- Per-claim provenance (`VerificationResult.claims`, `provenance()`).
- Claim-coverage score per sentence and a `min_coverage` threshold for paraphrases.
- Question-relevance check (`min_relevance`) so an extractive answer to a different question is
  withheld.

### Evals
- 239 questions (144 constant, 70 concept, 25 multi-hop) with seeded, type-stratified
  train/test split; multi-hop doc-set recall; fusion weight sweep on train only; reranker fit
  on train and reported on test; verifier negation / paraphrase / relevance measurements.
- LLM-as-judge harness (`ethrag.judge`, `evals/run_judge.py`), skipped without
  `ANTHROPIC_API_KEY`, tested with a scripted judge.

### Interfaces
- `ethrag build-index` / `ethrag ask --index` (np.savez + JSON persistence).
- `ethrag serve`: stdlib HTTP JSON API (`/health`, `/ask`, `/search`, `/related`).
- `ethrag mcp`: MCP-style stdio JSON-RPC server (`ask_eips`, `search_eips`, `related_eips`).
- `ethrag related`, `ethrag stats`; `ethrag "question"` still works.

### Project
- CI matrix 3.10-3.13 with a held-out eval gate; Makefile; Dockerfile; docs/EVALUATION.md,
  docs/CORPUS.md, CONTRIBUTING.md.

## 0.1.0

Initial release: 23 core EIPs, BM25 + hashed tf-idf + RRF, deterministic citation verifier,
26 questions, 22 tests.
