"""Verifier false-accept rate when the correct value is stated with a unit absent from the chunk.

    python3 evals/run_unit_swap.py --seed 7 --out results/422-unit-swap.json

For every constant question whose ``must_contain`` is ``NAME = <decimal integer>`` and whose gold
chunk contains that line (default verifier, offline, simulated answers), the control answer is
``NAME is <value>`` and the swap answer is ``NAME is <value> <unit>`` where ``unit`` is a word from
UNITS that does not occur in the cited chunk (first absent unit in a seeded rotation). Questions
whose chunk contains every unit are skipped. Release = verified.
"""
from __future__ import annotations

import argparse
import json
import platform
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
UNITS = ("bytes", "gwei", "seconds", "slots")
MIN_N = 20


def pick_unit(text: str, seed: int) -> str | None:
    """First unit in the seeded rotation that does not occur as a word in ``text``."""
    low = text.lower()
    for i in range(len(UNITS)):
        u = UNITS[(seed + i) % len(UNITS)]
        if not re.search(rf"\b{u}\b", low):
            return u
    return None


def interval(k: int, n: int) -> dict:
    p, lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": p, "wilson_lo": lo, "wilson_hi": hi}


def run(data: str, questions: str, seed: int) -> dict:
    chunks = list(load_corpus(data))
    by_doc: dict[str, list] = {}
    for c in chunks:
        by_doc.setdefault(c.doc, []).append(c)
    qs = [q for q in load_questions(questions) if q.get("type") == "constant"]
    ctl = [0, 0]
    swap = [0, 0]
    unit_used: dict[str, int] = {}
    skipped = 0
    for q in qs:
        m = LINE.match(q["must_contain"])
        gold = next((c for c in by_doc.get(q["gold_doc"], []) if q["must_contain"] in c.text), None)
        unit = pick_unit(gold.text, seed) if gold is not None else None
        if not m or gold is None or unit is None:
            skipped += 1
            continue
        name, val = m.group(1), m.group(2)
        r0 = verify_answer(f"{name} is {val} [{gold.id}]", {gold.id: gold}, [gold.id], question=q["q"])
        r1 = verify_answer(f"{name} is {val} {unit} [{gold.id}]", {gold.id: gold}, [gold.id], question=q["q"])
        ctl[0] += bool(r0.ok)
        ctl[1] += 1
        if r0.ok:  # only questions where the control is released count for the swap rate
            swap[0] += bool(r1.ok)
            swap[1] += 1
            unit_used[unit] = unit_used.get(unit, 0) + 1
    return {
        "questions_total": len(qs), "questions_skipped": skipped,
        "control_exact": interval(*ctl),
        "unit_swap": {"false_accept_rate": interval(*swap)["rate"], **interval(*swap),
                      "min_n": MIN_N, "evaluable": swap[1] >= MIN_N,
                      "units": list(UNITS), "unit_used": unit_used},
        "note": "constructed, simulated answers citing the unaltered gold chunk; only questions of the form NAME = <integer>",
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
        "command": "python3 evals/run_unit_swap.py " + " ".join(argv if argv is not None else sys.argv[1:]),
        "hardware": f"{platform.machine()} {platform.system()} cpu, python {platform.python_version()}",
        "wall_s": round(time.time() - t0, 2),
        "data_sources": [{"url": "https://github.com/ethereum/EIPs", "license": "CC0-1.0"},
                         {"url": "https://github.com/ethereum/ERCs", "license": "CC0-1.0"}],
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("control_exact", "unit_swap")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
