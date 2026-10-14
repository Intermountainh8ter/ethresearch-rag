# FAQ: the hard questions

### Does hybrid retrieval beat BM25?
Not convincingly. On the held-out split (n = 151) the tuned hybrid has Recall@5 0.815 against
0.788 for BM25: +0.026, paired-bootstrap 95% interval [-0.020, +0.073]. BM25 given the same
EIP-number boost scores 0.821, so the small edge is the boost, not the fusion. The dense side is a
hashed lexical baseline, so this says nothing about neural embeddings, which I did not evaluate.
What did help is the logistic reranker (0.815 -> 0.934). See
[EVALUATION.md](../EVALUATION.md#4-retrieval-results).

### Is the evaluation set real?
It is synthetic. Of the 504 questions, 273 were generated from the corpus text (constants found in tables,
code and prose; first sentences of Abstracts; pairs of documents linked in the citation graph)
and 231 are hand-written seeds; all were validated by program: the gold text must be in a gold
chunk, the correct answer must be supported by it, the wrong answer must not be. Real users ask
messier questions. Mined questions are templated and the set favours literal lookups. The
hand-written rows ("seed-*") are the closest to real usage and are reported separately.

### Is the test split really held out?
For the numeric knobs, yes: fusion weight, boost, reranker weights and verifier thresholds were
chosen on train, and single-hop questions are split by gold document. For design decisions, no: I
looked at failures on both splits while developing the verifier rules and retrieval features.
Treat test numbers as optimistic. The doc lists exactly what was and was not controlled.

### Was AI used to build this?
Yes. The code, the hand-written seed questions and these documents were produced with an AI coding
assistant. What you should rely on is not that, but that everything is checkable: the tests
(312, offline), the programmatic validators for every evaluation question, and the eval
scripts that regenerate every number in the README. The one place a model name appears is the
configurable default of the optional LLM generator.

### Why a deterministic verifier instead of an LLM judge?
An LLM judge is itself a model that can be wrong in fluent ways; spec questions are dominated by
literals, which a program can check exactly. The trade-off is real: the verifier cannot judge
meaning. An LLM-as-judge harness exists (`evals/run_judge.py`) for answer quality, but it needs an
API key and I report no numbers from it.

### Does "verified" mean the answer is correct?
No. It means every sentence is cited, every citation was retrieved, and every checkable token is in
the cited text, bound to its name, with consistent units, ranges, negation and topic. The offline
extractive answerer verifies 138 of 151 test questions but only 75 of those contain the reference
value. A verified answer is faithful to its source, which may be a Stagnant or Withdrawn draft.

### What can fool the verifier?
A sentence whose tokens are all in the chunk but unbound to a name: EIP-3541 bans `0xEF`; if the
chunk also mentions `0xFE`, "code starting with 0xFE is rejected" verifies (a test pins this).
Two values on one source line. Reversals without a negation word. Prose claims with no token (only
word overlap is checked). On test, 19/103 swapped-value answers, 33/151 question-restating answers
and 6/151 right-answer-to-the-wrong-question cases got through. Full list:
[EVALUATION.md](../EVALUATION.md#6-what-the-verifier-cannot-catch).

### Why are the reranker's results so much better than hybrid's? Is that leakage?
Partly regularity, not leakage in the usual sense. Documents are split between train and test and
the features contain no document ids, but my question generator produces regular shapes (purpose
questions always point at an Abstract chunk, mined constants at table or code lines) and the reranker
learns section and kind priors. Without the purpose tier the gain is R@5 0.878 -> 0.931 instead of
0.815 -> 0.934. Real traffic may differ.

### Why numpy only? Why hashed embeddings?
So `pip install` is trivial and CI needs no model download. The `Embedder` protocol and an optional
`SentenceTransformerEmbedder` (extra `embed`) exist; the latter is not run in tests. A neural
embedder is the first thing I would measure next.

### How fresh is the corpus?
A snapshot fetched on the date recorded in `data/MANIFEST.json`; refresh with
`scripts/fetch_corpus.py`. It is a curated list of numbers, not the whole repositories
([CORPUS.md](../CORPUS.md)). Statuses are kept as metadata and not filtered.

### Does it work with models other than the default?
The generator is any object with `complete(system, user) -> str` (`ethrag.llm.LLM`). The bundled
implementation calls the Anthropic API and reads the model id from `ETHRAG_MODEL`. LLM output was
not evaluated here (no API key in my environment). Expect more rejections from `min_coverage` on
paraphrased text; `--min-coverage 0.5` is the first knob.

### Is it production ready?
No. It is a careful prototype with an honest evaluation. Missing: neural retrieval, LLM-generated
answer evaluation, streaming, auth on the HTTP server, rate limiting, non-English support, and
any guarantee about spec status or fork activation.

### How do I reproduce the numbers?
`pip install -e ".[dev]"`, then `python evals/run_evals.py` and `python evals/run_verifier_eval.py`
(about a minute each, offline). `make eval-gate` runs the thresholds CI enforces.

### What are the licences?
Code MIT. Corpus documents keep upstream licences: EIPs, ERCs and consensus specs CC0 1.0,
Flashbots docs MIT (notice in `data/LICENSE`).
