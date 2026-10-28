# Contributing

Small, dependency-light project; the bar is "tests and evals stay honest".

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
make lint test           # ruff + pytest, offline, no API key, about 25 s
make cov                 # pytest with line + branch coverage
make eval-gate           # the held-out gates CI enforces (about 1.5 min)
make eval                # full retrieval tables with misses
make verifier-eval       # verifier TPR / FPR by case class
```

Python 3.10 through 3.13 are supported. `ruff` runs with `target-version = "py310"`, which flags
syntax newer than 3.10 (no `match` shortcuts, no PEP 695 generics, no PEP 701 f-strings).

## Ground rules

* **numpy stays the only required runtime dependency.** Anything else goes behind an optional
  extra (`llm`, `embed`) and an import inside the function that needs it, so `import ethrag`
  never fails on a bare interpreter.
* **Tests run offline.** No network, no API keys, no model downloads. Anything that talks to a
  service needs a stand-in (`ScriptedLLM`, `ScriptedJudge`, a monkeypatched `fetch`) and is
  skipped or stubbed without a key. HTTP and MCP tests bind to port 0 or use pipes.
* **Never tune on the test split.** `evals/run_evals.py` and `evals/run_verifier_eval.py`
  choose every numeric knob on train. If you add a knob, add it to a train-only sweep and report
  both splits. Say so in the PR if you looked at test failures while designing a rule; the docs
  list this caveat for the existing rules.
* **Do not fabricate numbers.** Tables in README.md and docs/ are pasted from real runs.
  Change code that moves a number, re-run, update the tables and CHANGELOG.md together.
* **A verifier change needs a test on both sides:** a case it must now reject and a correct
  answer it must still accept. Known weaknesses live as `test_blind_spot_*` tests in
  `tests/test_verify.py`; if one starts failing the verifier got stronger, so update
  docs/EVALUATION.md section 6 rather than deleting the test.
* **Corpus changes go through `scripts/fetch_corpus.py`.** Edit the curated list, re-run it,
  commit the regenerated `data/MANIFEST.json`, then `python evals/build_questions.py`. Keep
  `data/` under 40 MB (`tests/test_corpus.py` enforces it along with hashes and duplicates).

## Evaluation questions

`evals/questions.jsonl` is generated. Do not edit it by hand.

* Hand-written questions go in `evals/seed_questions.jsonl`:
  ```json
  {"id": "c-200", "type": "constant", "q": "What does EIP-1559 set ELASTICITY_MULTIPLIER to?",
   "gold_doc": "eip-1559", "must_contain": "ELASTICITY_MULTIPLIER = 2",
   "answer": "ELASTICITY_MULTIPLIER is 2", "wrong": "ELASTICITY_MULTIPLIER is 4"}
  ```
  Multi-hop questions use `"gold": [[doc, needle], [doc, needle]]`.
* Run `python evals/build_questions.py`. It validates every record with the production verifier
  (needle in a gold chunk, correct answer supported, wrong answer **not** supported), repairs or
  drops failures, assigns the seeded train/test split and rewrites the file. The report lists
  what was dropped and why.
* `python evals/build_questions.py --check` (also a test and a CI step) fails when the committed
  file differs from a fresh build.
* Splits are recomputed over the whole set, so adding questions can move documents between
  train and test. Re-run both eval scripts and commit `evals/results.json`,
  `evals/verifier_results.json` and `evals/artifacts/reranker.json`.

## Pull requests

* `ruff check .` and `pytest -q` pass; coverage of `src/ethrag` does not drop.
* If a retrieval or verifier number moves, paste the new table and explain why.
* One behavioural change per commit where possible; keep commit messages factual.

## Releasing

Bump `version` in `pyproject.toml` and `__version__` in `src/ethrag/__init__.py`, move the
CHANGELOG entry, tag. There is no PyPI package yet.

## Publishing to PyPI

Maintainer steps. Nothing is published until you do them; the workflow in `.github/workflows/publish.yml`
only runs when a GitHub Release is published.

1. On pypi.org, add a *pending trusted publisher* for the project name `ethrag`: owner `Intermountainh8ter`,
   repository `ethresearch-rag`, workflow `publish.yml`, environment `pypi`. (No API token is needed or stored.)
2. In this repository's Settings, create an environment named `pypi` (optionally with required reviewers).
3. Make sure `version` in `pyproject.toml` is the version you want, and that CI is green on `main`.
4. Create a GitHub Release whose tag is `v<version>` (for example `v0.2.0`). The workflow builds the
   sdist and wheel, runs `twine check`, and uploads them.
5. A PyPI version can never be re-uploaded, so check the built files first with `python -m build`
   and `twine check dist/*`.
6. The wheel contains the code only, not the 17,000-chunk corpus under data/. After pip install ethrag, point the CLI at a corpus with --data DIR (a clone of this repository has one in data/) or run scripts/fetch_corpus.py; without one, ethrag ask stops with a message saying so.
