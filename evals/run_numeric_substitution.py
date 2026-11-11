"""Verifier false-accept rate when a constant's value is swapped for another in-chunk number.

    python3 evals/run_numeric_substitution.py --seed 7 --out results/110-numeric-substitution.json

For every constant question whose gold chunk contains the ``must_contain`` line (default verifier
settings, offline, simulated answers):

* ``in_chunk_swap``: the answer states the constant name with a different number that appears
  elsewhere in the same cited chunk. False-accept = verified.
* ``absent_value``: the answer states a number absent from the chunk. False-accept = verified.
* ``control``: the correct answer. False-withhold = not verified.
"""
from __future__ import annotations

import argparse
import json
import platform
import random
import re
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

NUM = re.compile(r"(?<![\w.])\d+(?![\w.])")


def other_numbers(text: str, value: str) -> list[str]:
    """Distinct integer tokens in ``text`` different from ``value``, in order of appearance."""
    seen: list[str] = []
    for m in NUM.findall(text):
        if m != value.strip() and m not in seen:
            seen.append(m)
    return seen


def absent_number(text: str, rng: random.Random) -> str:
    while True:
        n = str(rng.randint(1000, 99999))
        if n not in text:
            return n


def interval(k: int, n: int) -> dict:
    p, lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": p, "wilson_lo": lo, "wilson_hi": hi}


def run(data: str, questions: str, seed: int) -> dict:
    rng = random.Random(seed)
    by_doc: dict[str, list] = {}
    for c in load_corpus(data):
        by_doc.setdefault(c.doc, []).append(c)
    qs = [q for q in load_questions(questions) if q.get("type") == "constant"]
    acc = {"in_chunk_swap": 0, "absent_value": 0, "control": 0}
    n = 0
    skipped = 0
    for q in qs:
        gold = next((c for c in by_doc.get(q["gold_doc"], []) if q["must_contain"] in c.text), None)
        name, _, value = q["must_contain"].partition(" = ")
        others = other_numbers(gold.text, value) if gold and value else []
        if not others:
            skipped += 1
            continue
        swap = rng.choice(others)
        cases = {
            "in_chunk_swap": f"{name} is {swap}",
            "absent_value": f"{name} is {absent_number(gold.text, rng)}",
            "control": q["answer"],
        }
        n += 1
        for v, text in cases.items():
            res = verify_answer(f"{text} [{gold.id}]", {gold.id: gold}, [gold.id], question=q["q"])
            acc[v] += bool(res.ok)
    iv = {v: interval(k, n) for v, k in acc.items()}
    return {
        "questions_used": n, "questions_skipped": skipped,
        "numeric_substitution": {
            "false_accept_rate": iv["in_chunk_swap"]["rate"],
            "wilson": iv["in_chunk_swap"],
            "absent_value_false_accept_rate": iv["absent_value"]["rate"],
            "control_false_withhold_rate": 1 - iv["control"]["rate"],
        },
        "accepted_counts": acc,
        "note": "constructed, simulated answers; in-chunk swap uses any other integer token in the gold chunk",
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
        "command": "python3 evals/run_numeric_substitution.py " + " ".join(argv if argv is not None else sys.argv[1:]),
        "hardware": f"{platform.machine()} {platform.system()} cpu, python {platform.python_version()}",
        "wall_s": round(time.time() - t0, 2),
        "data_sources": [{"url": "https://github.com/ethereum/EIPs", "license": "CC0-1.0"},
                         {"url": "https://github.com/ethereum/ERCs", "license": "CC0-1.0"}],
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("questions_used", "numeric_substitution")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
