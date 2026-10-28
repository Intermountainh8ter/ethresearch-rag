# r/ethdev post (draft)

**Title:** ethrag: Q&A over EIPs/ERCs/consensus specs where a deterministic verifier can withhold the answer (with the numbers, including the ones that did not work)

Spec questions are mostly constants: `BASE_FEE_MAX_CHANGE_DENOMINATOR = 8`, opcode numbers, type
bytes, `MAX_BLOB_GAS_PER_BLOCK`. A fluent answer with the wrong constant is worse than no
answer, so ethrag treats generation as untrusted and puts a checker with no model in it between
the draft and the user:

- every sentence has a `[doc#NNN]` citation and the chunk was actually retrieved;
- every number / hex / `2**n` / `ALL_CAPS` token of the answer is a token of the cited chunk
  (`0x48` = `72`, `2**12` = `4096`, `32 ETH` = `32000000000` gwei count as equal; `8` is not found
  inside `58`);
- the value sits on the same line as the name it is claimed for, units and ranges agree,
  negation is not invented, and the answer is about the question asked.

`ethrag ask --explain` prints one row per claim with the chunk and the source line.

**Corpus:** 901 documents (379 EIPs, 458 ERCs, 38 consensus-spec files, 26 Flashbots docs), a
curated subset, sha256-listed, CC0/MIT, rebuilt by a stdlib fetcher. 17,387 chunks; tables and
code blocks are chunked atomically.

**Numbers** (504 synthetic questions mined from the corpus text and validated by program; seeded
train/test split stored in the file; weights tuned on train only; test n = 151):

| | R@1 | R@5 | nDCG@10 |
|---|---|---|---|
| BM25 | 0.424 | 0.788 | 0.622 |
| hashed dense | 0.325 | 0.609 | 0.467 |
| hybrid (RRF + EIP-number boost) | 0.437 | 0.815 | 0.649 |
| hybrid + trained reranker | 0.695 | 0.934 | 0.814 |

Things that did not go the way a README usually says:

- **Hybrid is not clearly better than BM25.** +0.026 R@5, bootstrap interval [-0.020, +0.073].
  BM25 with the same EIP-number boost matches it. The "dense" side is hashed tf-idf, not a neural
  embedding by default. I measured one real embedder once (bge-small): dense alone was worse (R@5 0.563 vs 0.609) and
  hybrid R@5 did not change (0.815), so it did not help recall here.
- The reranker's gain is partly a quirk of my question generator (some question types always
  point at Abstract chunks and the reranker learns that); without that tier it is 0.878 -> 0.931.
- Graph expansion along `requires` edges did nothing for R@5.
- The questions are synthetic and templated, and I looked at both splits while designing the
  verifier rules, so the test numbers are optimistic.

**Verifier on test:** accepts 147/151 correct reference answers; accepts 19/103 answers whose value
was swapped for another literal from the same chunk, 2/113 negated answers, 6/151 correct
answers to a *different* question. Blind spot, pinned by a test: a chunk that mentions both `0xEF`
and `0xFE` supports "code starting with 0xFE is rejected". And verified is not correct: the
offline extractive baseline verifies 138/151 but only 75 contain the reference value.

Python 3.10-3.13, numpy only, tests and evals offline. CLI, stdlib HTTP API, MCP stdio server.
Everything is MIT; `docs/EVALUATION.md` has the methodology and the "what this cannot catch" list.
I would like to hear which questions it gets wrong.

Repo: https://github.com/Intermountainh8ter/ethresearch-rag
