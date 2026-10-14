# Contributing

Thanks for looking. This is a small, dependency-light project; the bar is "tests and evals
stay honest", not ceremony.

## Setup

```
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
make lint test          # ruff + pytest (offline, no API key)
make eval               # retrieval + verifier evals, prints train/test tables
```

Python 3.10 through 3.13 are supported; keep syntax compatible with 3.10 (no `match` on
anything but trivially rewritable code, no PEP 695 generics).

## Ground rules

- **numpy stays the only required runtime dependency.** Anything else goes behind an optional
  extra (`llm`, `embed`) and an import inside the function that needs it, so `import ethrag`
  never fails on a bare interpreter.
- **Tests run offline.** No network, no API keys, no model downloads. Anything that needs an LLM
  must have a scripted stand-in (`ScriptedLLM`, `ScriptedJudge`) and be skipped without a key.
- **Never tune on the test split.** `evals/run_evals.py` fits fusion weights and the reranker on
  the train split and reports both. If you add a knob, sweep it on train only.
- **Do not fabricate numbers.** README tables are pasted from a real `make eval` run; update
  them together with the code that changed them, and mention the run in CHANGELOG.md.
- **Corpus changes go through `scripts/fetch_corpus.py`.** Edit the curated number list or page
  list, re-run it, and commit the regenerated `data/MANIFEST.json`. Keep `data/` under 40 MB.
- **Eval questions must be grounded.** `tests/test_corpus.py` checks that every question's
  `must_contain` is in a chunk of its gold doc, that the reference answer's claim tokens are
  supported, and that the `wrong` answer's tokens are *not* all present (otherwise the verifier
  could not tell them apart and the question measures nothing).

## Adding a question

Append a line to `evals/questions.jsonl`:

```json
{"id": "c-145", "type": "constant", "q": "...", "gold_doc": "eip-1559",
 "must_contain": "ELASTICITY_MULTIPLIER = 2", "answer": "ELASTICITY_MULTIPLIER is 2",
 "wrong": "ELASTICITY_MULTIPLIER is 4"}
```

Multi-hop questions use `"gold": [[doc, needle], [doc, needle]]`. Ids are stable; the
train/test split is a seeded, type-stratified function of the id, so a new question lands in
one split deterministically and existing assignments do not move.

## Pull requests

- `ruff check .` and `pytest -q` must pass on 3.10 and 3.13.
- If retrieval numbers move, paste the new `make eval` tables into the README and say why.
- Keep commits focused; one behavioural change per commit is ideal.
