# Configurations tried — ethrag, 2026-10-14

Every configuration run on this machine, including the ones that were wrong.
Host: 2x AMD gfx906 / ROCm 7.2.3, Python 3.12.3. Code at `b2f504ba82` plus the
`--embedder` patch described below.

## 1. Reproduction of the committed numbers — RUN, PASSED

```
python evals/run_evals.py --min-test-recall5 0.80
```

All retrieval tables (TRAIN, TEST, multi-hop, sensitivity) reproduced
identically. Gate passed: test hybrid R@5 0.815 >= 0.80. The only difference
in `evals/results.json` was `reranker.path`, an absolute path committed
upstream as `/home/user/...`, which cannot match on any other machine.

## 2. First attempt, WRONG — my error, recorded because it changed a number

```
python evals/run_evals.py          # no --min-test-recall5
```

Produced `gate.threshold: 0.0` against the committed `0.8`. Not a repo bug:
the flag defaults to 0.0 and both `.github/workflows/ci.yml` and
`docs/EVALUATION.md` invoke it with `0.80`. Re-run as in 1. Keeping this here
because a reader comparing result files would otherwise see a threshold
change with no explanation.

## 3. Neural embedder — RUN, this is the Stage 3 answer

Added `--embedder {hashed,neural}` and `--embedder-model` to
`evals/run_evals.py`, routing through the repo's existing `Embedder` protocol
and its already-present `SentenceTransformerEmbedder`. Default stays `hashed`,
and the default path was re-verified bit-identical after the patch.

```
python evals/run_evals.py --embedder neural --min-test-recall5 0.0 --bootstrap 2000
```

No fusion weights or reranker weights were hand-set. They are fit by the
harness on TRAIN only, exactly as in the hashed run; TEST was evaluated once
per embedder. No tuning was done against TEST.

## 4. Label bug in my own patch — FIXED before publishing

The first neural run printed its dense row as `dense(hashed)` because that
label is hardcoded at the systems dict. The numbers were the neural ones, so
the table was mislabelled. Fixed so the label follows the embedder actually in
use, then re-run. The numbers are unchanged by the fix (bm25 R@1 0.424,
hybrid R@1 0.523 in both runs); only the row label differs.

## Not tried, and why

- **No other embedder models.** Only `BAAI/bge-small-en-v1.5`, the repo's own
  documented default. Sweeping embedders and picking the best would be tuning
  on a split, and there is no second held-out split to protect it.
- **No reranker hyper-parameter changes.** Used as shipped.
- **No corpus changes.** The bundled 23-EIP corpus throughout.
