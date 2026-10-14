# Changelog

## 0.2.0 (unreleased)

Corpus, retrieval, verifier, evals and interfaces expanded roughly tenfold. numpy is still the
only required runtime dependency and the whole suite runs offline.

### Corpus
- `scripts/fetch_corpus.py`: stdlib-only, reproducible fetcher for ethereum/EIPs, ethereum/ERCs,
  ethereum/consensus-specs and a curated set of flashbots-docs pages; writes
  `data/MANIFEST.json` (id, number, title, status, category, type, created, requires, sha256,
  bytes). Partial runs (`--numbers`, `--no-specs`) merge instead of pruning.
- 23 EIPs / 497 chunks became 901 documents / 17,387 chunks / 11.3 MB; `data/LICENSE` records the
  CC0 and MIT provenance; integrity is enforced by `tests/test_corpus.py`.

### Ingest
- Chunk metadata (status, category, type, requires, created, number) and a `kind`
  (prose / code / table); table-aware (header repeated on every row split) and code-aware
  (fences kept, never split mid-line) chunking; TOC bullets, MDX import/export lines and JSX
  tags removed; `Spec`/`Docs` status for documents without front matter.
- Fixed: a code window could carry two giant lines of overlap and reach 540 words (now capped at
  a quarter of the chunk size); prose beginning with "import"/"export" was deleted as MDX.

### Retrieval
- Inverted-index BM25 with title/section field boosting and query-side stop-word removal
  (chosen on train); hashed word + character-trigram dense baseline as a numpy CSC matrix;
  weighted RRF; EIP/ERC-number detection (`boost` | `restrict` | `off`; boost default 4.0);
  graph expansion along `requires` edges; `HybridIndex.search` no longer pads with zero-score
  chunks.
- `LogisticReranker` (numpy, 20 interpretable features) fit on train with grouped
  cross-validation for L2; weights in `evals/artifacts/reranker.json` for `--rerank`.
- `ethrag build-index` / `--index`: `.npz` + JSON persistence.

### Verifier
- Fixed: identifiers were stored with their underscores removed (`GAS_PER_BLOB` became
  `gasperblob`).
- New: identifier-value **binding** (a value must sit on the same line as its name, or on a line
  with no name of its own), **units** (ETH/gwei/wei, data sizes, time, gas, blocks, slots...),
  **ranges** (`1-16`, `0 to 0x7f`), negation against code lines and table rows, content
  coverage, topical **question relevance** (reference numbers and "propose/describe" words do not
  count; a named parameter must be mentioned and given a value), optional novelty, per-claim
  provenance with the source line, `ethrag ask --explain`, thousands separators (`1,000,000`,
  `1_000_000`) and `2^17`/`2**17`/`0x00FF` normalisation.
- Defaults chosen on train: `min_coverage 0.7`, `min_relevance 0.1`.
- Documented blind spots as tests (`0xEF` vs `0xFE`, two values on one line, grounded is not
  answering).

### Evals
- 504-question set mined from the corpus text and validated by program (`ethrag.evalgen`,
  `evals/build_questions.py`), seeded 70/30 split stored in the file, split by gold document for
  single-hop questions.
- `evals/run_evals.py`: BM25 / dense / hybrid / hybrid+rerank / hybrid+graph on train and test,
  R@1/3/5/10, MRR@10, nDCG@10, multi-hop doc-set recall, ablations, paired bootstrap, tier
  breakdown, gate `--min-test-recall5`.
- `evals/run_verifier_eval.py`: TPR/FPR by case class (`wrong`, `swap`, `negated`, `vague`,
  `echo`, `wrong_citation`, `wrong_q`, `uncited`, `fake_citation`) with Wilson intervals, ablation
  by check, train-only threshold sweep, extractive end-to-end; gates `--min-test-tpr`,
  `--max-test-fpr`.
- LLM-as-judge harness (`ethrag.judge`, `evals/run_judge.py`), skipped without an API key.
- Fixed: a truncated JSON judge verdict (`{"correct": `) was read as "correct".

### Interfaces
- `ethrag serve` (stdlib HTTP: `/health`, `/ask`, `/search`, `/related`; per-request `k`, no shared
  mutable state, JSON 500s); `ethrag mcp` (stdio JSON-RPC, protocol negotiation, notifications
  never answered, per-claim provenance in `ask_eips`); `ethrag related`, `ethrag stats`;
  friendly `error:` messages with exit code 1; `ethrag "question"` still works.
- Offline extractive answerer answers `NAME is value` from definition lines (including several
  constants in one question).

### Project
- ruff and pytest clean; CI matrix 3.10-3.13 with ruff, pytest + coverage and both held-out eval
  gates; Makefile; Dockerfile; docs (`EVALUATION.md`, `CORPUS.md`, `FLOWCHARTS.md`, `launch/`),
  Mermaid sources and rendered SVGs.
- Tests: 22 -> 312.

## 0.1.0

Initial release: 23 core EIPs / 497 chunks, BM25 + hashed tf-idf + RRF, deterministic citation
verifier, 26 questions, 22 tests.
