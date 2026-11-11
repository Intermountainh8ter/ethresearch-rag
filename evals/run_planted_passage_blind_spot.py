"""Planted-passage blind spot of the citation verifier on constructed constant questions.

    python3 evals/run_planted_passage_blind_spot.py --seed 7 --out results/101-planted-passage-blind-spot.json

Simulated, constructed variants of every constant question that has a stored ``wrong`` value and
whose gold chunk contains the ``must_contain`` line (default verifier settings, offline):

* ``planted_agrees_with_memory``: the cited chunk is a copy of the gold chunk with the constant's
  value replaced by the wrong value; the answer states the wrong value. Wrong-accept = verified.
* ``planted_absent``: the cited chunk is the real gold chunk; the answer states the wrong value.
* ``unanswerable``: the cited chunk is an unrelated chunk from another document; the answer states
  the correct value. Wrong-accept = verified.
* ``answerable_control``: real chunk, correct answer. False-withhold = not verified.

The "common misconception" is the stored ``wrong`` value, a proxy, not a measured model belief.
"""
from __future__ import annotations

import argparse
import dataclasses
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

VARIANTS = ("planted_agrees_with_memory", "planted_absent", "unanswerable", "answerable_control")


def plant(chunk, must: str, answer: str, wrong: str):
    """Copy of ``chunk`` whose ``must`` line is rewritten to the wrong statement, or None."""
    name, _, value = must.partition(" = ")
    wname, _, wvalue = wrong.partition(" is ")
    if not value or wname.strip() != name.strip() or not wvalue or wvalue.strip() == value.strip():
        return None
    if must not in chunk.text:
        return None
    return dataclasses.replace(chunk, id="planted#000", text=chunk.text.replace(must, f"{name} = {wvalue.strip()}"))


def interval(k: int, n: int) -> dict:
    p, lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": p, "wilson_lo": lo, "wilson_hi": hi}


def run(data: str, questions: str, seed: int) -> dict:
    rng = random.Random(seed)
    chunks = load_corpus(data)
    by_doc: dict[str, list] = {}
    for c in chunks:
        by_doc.setdefault(c.doc, []).append(c)
    qs = [q for q in load_questions(questions) if q.get("type") == "constant" and q.get("wrong")]
    accepted = {v: 0 for v in VARIANTS}
    n = {v: 0 for v in VARIANTS}
    skipped = 0
    for q in qs:
        gold = next((c for c in by_doc.get(q["gold_doc"], []) if q["must_contain"] in c.text), None)
        planted = plant(gold, q["must_contain"], q["answer"], q["wrong"]) if gold else None
        if gold is None or planted is None:
            skipped += 1
            continue
        other_docs = [d for d in sorted(by_doc) if d != q["gold_doc"]]
        decoy = rng.choice(by_doc[rng.choice(other_docs)])
        cases = {
            "planted_agrees_with_memory": (q["wrong"] + f" [{planted.id}]", {planted.id: planted}, planted.id),
            "planted_absent": (q["wrong"] + f" [{gold.id}]", {gold.id: gold}, gold.id),
            "unanswerable": (q["answer"] + f" [{decoy.id}]", {decoy.id: decoy}, decoy.id),
            "answerable_control": (q["answer"] + f" [{gold.id}]", {gold.id: gold}, gold.id),
        }
        for v, (text, pool, cid) in cases.items():
            res = verify_answer(text, pool, [cid], question=q["q"])
            n[v] += 1
            accepted[v] += bool(res.ok)
    wa = {v: interval(accepted[v], n[v]) for v in VARIANTS}
    return {
        "questions_used": n["planted_agrees_with_memory"], "questions_skipped": skipped,
        "wrong_accept_rate": {v: wa[v]["rate"] for v in VARIANTS[:3]},
        "wrong_accept_wilson": {v: wa[v] for v in VARIANTS[:3]},
        "false_withhold_rate": {"answerable_control": 1 - wa["answerable_control"]["rate"]},
        "accepted_counts": accepted,
        "note": "constructed, simulated variants; the misconception is the stored wrong value, a proxy",
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
        "command": "python3 evals/run_planted_passage_blind_spot.py " + " ".join(argv if argv is not None else sys.argv[1:]),
        "hardware": f"{platform.machine()} {platform.system()} cpu, python {platform.python_version()}",
        "wall_s": round(time.time() - t0, 2),
        "data_sources": [{"url": "https://github.com/ethereum/EIPs", "license": "CC0-1.0"},
                         {"url": "https://github.com/ethereum/ERCs", "license": "CC0-1.0"}],
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("questions_used", "wrong_accept_rate", "false_withhold_rate")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
