"""Verifier false-withhold rate when a correct answer cites the gold chunk plus distractor chunks.

    python3 evals/run_multi_doc.py --seed 7 --distractors 3 --out results/434-multi-doc.json

For every constant question whose ``must_contain`` is ``NAME = <decimal integer>`` and whose gold
chunk contains that line (default verifier, offline, simulated answers), the control answer cites
the gold chunk only; the multi-document answer cites the gold chunk plus seeded distractor chunks
from other documents. All cited chunks count as retrieved. Withhold = not verified.
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

LINE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*) = (\d+)$")
MIN_N = 20


def interval(k: int, n: int) -> dict:
    p, lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": p, "wilson_lo": lo, "wilson_hi": hi}


def run(data: str, questions: str, seed: int, distractors: int) -> dict:
    rng = random.Random(seed)
    chunks = list(load_corpus(data))
    by_doc: dict[str, list] = {}
    for c in chunks:
        by_doc.setdefault(c.doc, []).append(c)
    qs = [q for q in load_questions(questions) if q.get("type") == "constant"]
    ctl = [0, 0]
    multi = [0, 0]
    skipped = 0
    for q in qs:
        m = LINE.match(q["must_contain"])
        gold = next((c for c in by_doc.get(q["gold_doc"], []) if q["must_contain"] in c.text), None)
        if not m or gold is None:
            skipped += 1
            continue
        name, val = m.group(1), m.group(2)
        pool = [c for c in chunks if c.doc != gold.doc]
        extra = rng.sample(pool, distractors)
        cited = [gold] + extra
        r0 = verify_answer(f"{name} is {val} [{gold.id}]", {gold.id: gold}, [gold.id], question=q["q"])
        ids = [c.id for c in cited]
        text = f"{name} is {val} " + "".join(f"[{i}]" for i in ids)
        r1 = verify_answer(text, {c.id: c for c in cited}, ids, question=q["q"])
        ctl[0] += not r0.ok
        ctl[1] += 1
        multi[0] += not r1.ok
        multi[1] += 1
    c, mu = interval(*ctl), interval(*multi)
    return {
        "questions_total": len(qs), "questions_skipped": skipped, "distractors": distractors,
        "control_gold_only": {"false_withhold_rate": c["rate"], **c},
        "multi_doc": {"false_withhold_rate": mu["rate"], **mu,
                      "gap_to_control": mu["rate"] - c["rate"],
                      "min_n": MIN_N, "evaluable": multi[1] >= MIN_N},
        "note": "constructed, simulated answers citing the unaltered gold chunk plus seeded distractors from other documents",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--questions", default=str(ROOT / "evals" / "questions.jsonl"))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--distractors", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    t0 = time.time()
    res = run(a.data, a.questions, a.seed, a.distractors)
    res["provenance"] = {
        "started_at": started, "seeds": [a.seed],
        "command": "python3 evals/run_multi_doc.py " + " ".join(argv if argv is not None else sys.argv[1:]),
        "hardware": f"{platform.machine()} {platform.system()} cpu, python {platform.python_version()}",
        "wall_s": round(time.time() - t0, 2),
        "data_sources": [{"url": "https://github.com/ethereum/EIPs", "license": "CC0-1.0"},
                         {"url": "https://github.com/ethereum/ERCs", "license": "CC0-1.0"}],
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("control_gold_only", "multi_doc")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
