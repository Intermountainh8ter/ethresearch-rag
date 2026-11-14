"""Verifier pass rate when the cited chunk is a superseded variant of the passage.

    python3 evals/run_stale_version_blind_spot.py --seed 7 --out results/163-stale-version-blind-spot.json

For every constant question whose gold chunk contains the ``must_contain`` line (default verifier
settings, offline, simulated answers):

* ``stale_version``: the cited chunk keeps the value line but a status line is changed to
  Superseded and every other integer outside the value line is changed. Pass = verified.
* ``control``: the unaltered chunk. False-withhold = not verified.
"""
from __future__ import annotations

import argparse
import dataclasses
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


def make_stale(text: str, value_line: str, rng: random.Random) -> str:
    """Keep ``value_line``; shift every other integer token and prepend a Superseded status line."""
    out = []
    for line in text.split("\n"):
        if value_line in line:
            out.append(line)
        else:
            out.append(NUM.sub(lambda m: str(int(m.group(0)) + rng.randint(1, 9)), line))
    return "Status: Superseded\n" + "\n".join(out)


def interval(k: int, n: int) -> dict:
    p, lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": p, "wilson_lo": lo, "wilson_hi": hi}


def run(data: str, questions: str, seed: int) -> dict:
    rng = random.Random(seed)
    by_doc: dict[str, list] = {}
    for c in load_corpus(data):
        by_doc.setdefault(c.doc, []).append(c)
    qs = [q for q in load_questions(questions) if q.get("type") == "constant"]
    acc = {"stale_version": 0, "control": 0}
    n = 0
    skipped = 0
    for q in qs:
        gold = next((c for c in by_doc.get(q["gold_doc"], []) if q["must_contain"] in c.text), None)
        if gold is None:
            skipped += 1
            continue
        stale = dataclasses.replace(gold, text=make_stale(gold.text, q["must_contain"], rng))
        n += 1
        for v, chunk in (("stale_version", stale), ("control", gold)):
            res = verify_answer(f"{q['answer']} [{gold.id}]", {gold.id: chunk}, [gold.id], question=q["q"])
            acc[v] += bool(res.ok)
    iv = {v: interval(k, n) for v, k in acc.items()}
    return {
        "questions_used": n, "questions_skipped": skipped,
        "stale_version": {"pass_rate": iv["stale_version"]["rate"], "wilson": iv["stale_version"]},
        "answerable_control": {"false_withhold_rate": 1 - iv["control"]["rate"]},
        "accepted_counts": acc,
        "note": "constructed, simulated stale chunks; the value line is preserved, other integers shifted by 1-9",
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
        "command": "python3 evals/run_stale_version_blind_spot.py " + " ".join(argv if argv is not None else sys.argv[1:]),
        "hardware": f"{platform.machine()} {platform.system()} cpu, python {platform.python_version()}",
        "wall_s": round(time.time() - t0, 2),
        "data_sources": [{"url": "https://github.com/ethereum/EIPs", "license": "CC0-1.0"},
                         {"url": "https://github.com/ethereum/ERCs", "license": "CC0-1.0"}],
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("questions_used", "stale_version", "answerable_control")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
