"""Verifier release rate by quote shape (exact, truncated, ellipsis-joined, word-reordered).

    python3 evals/run_quote_shape_blind_spot.py --seed 7 --out results/165-quote-shape-blind-spot.json

For every constant question whose gold chunk contains the ``must_contain`` line (default verifier
settings, offline, simulated answers of the form ``"<quote>" [chunk-id]``):

* ``exact``: the value line verbatim (control; cases the verifier does not release are excluded
  from the shape denominators, so every shape is measured on the same accepted set).
* ``truncated``: the value line cut mid-sentence before the claimed value.
* ``ellipsis_join``: the preceding chunk line, ``...``, then the value line.
* ``reordered``: the words of the value line shuffled.
"""
from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ethrag.evalkit import load_questions
from ethrag.ingest import load_corpus
from ethrag.verify import verify_answer
from ethrag.verifyeval import wilson

SHAPES = ("exact", "truncated", "ellipsis_join", "reordered")


def truncate(line: str, value: str) -> str:
    """Cut ``line`` before the first occurrence of ``value`` (whole line minus last word if absent)."""
    i = line.find(value)
    cut = line[:i] if i > 0 else line.rsplit(" ", 1)[0]
    return cut.rstrip(" =:") .strip()


def ellipsis_join(prev: str, line: str) -> str:
    return f"{prev.strip()} ... {line.strip()}"


def reorder(line: str, rng: random.Random) -> str:
    words = line.split()
    out = words[:]
    for _ in range(5):
        rng.shuffle(out)
        if out != words:
            break
    return " ".join(out)


def interval(k: int, n: int) -> dict:
    p, lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": p, "wilson_lo": lo, "wilson_hi": hi}


def run(data: str, questions: str, seed: int) -> dict:
    rng = random.Random(seed)
    by_doc: dict[str, list] = {}
    for c in load_corpus(data):
        by_doc.setdefault(c.doc, []).append(c)
    qs = [q for q in load_questions(questions) if q.get("type") == "constant"]
    rel = {s: 0 for s in SHAPES}
    n = skipped = tried = 0
    for q in qs:
        gold = next((c for c in by_doc.get(q["gold_doc"], []) if q["must_contain"] in c.text), None)
        if gold is None:
            skipped += 1
            continue
        lines = gold.text.split("\n")
        idx = next(i for i, l in enumerate(lines) if q["must_contain"] in l)
        line = lines[idx]
        prev = next((l for l in reversed(lines[:idx]) if l.strip()), "")
        tried += 1
        quotes = {
            "exact": line.strip(),
            "truncated": truncate(line, q["must_contain"]),
            "ellipsis_join": ellipsis_join(prev, line),
            "reordered": reorder(line, rng),
        }
        ok = {}
        for s, quote in quotes.items():
            res = verify_answer(f'"{quote}" [{gold.id}]', {gold.id: gold}, [gold.id], question=q["q"])
            ok[s] = bool(res.ok)
        if not ok["exact"]:
            continue
        n += 1
        for s in SHAPES:
            rel[s] += ok[s]
    iv = {s: interval(k, n) for s, k in rel.items()}
    return {
        "questions_used": tried, "questions_skipped": skipped, "accepted_cases": n,
        "quote_shape": {
            "release_rate": {s: iv[s]["rate"] for s in SHAPES},
            "wilson": iv,
        },
        "exact_control_release_rate": interval(n, tried)["rate"] if tried else 0.0,
        "released_counts": rel,
        "note": "simulated quotes built from the gold chunk; shapes measured on cases where the exact quote is released",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--questions", default=str(ROOT / "evals" / "questions.jsonl"))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    t0 = time.time()
    res = run(a.data, a.questions, a.seed)
    res["provenance"] = {
        "started_at": started, "seeds": [a.seed],
        "command": "python3 evals/run_quote_shape_blind_spot.py " + " ".join(argv if argv is not None else sys.argv[1:]),
        "hardware": f"{platform.machine()} {platform.system()} cpu, python {platform.python_version()}",
        "wall_s": round(time.time() - t0, 2),
        "data_sources": [{"url": "https://github.com/ethereum/EIPs", "license": "CC0-1.0"},
                         {"url": "https://github.com/ethereum/ERCs", "license": "CC0-1.0"}],
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("questions_used", "accepted_cases", "quote_shape")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
