"""Verifier release rate when the constant's name is restated in prose form.

    python3 evals/run_name_variants.py --seed 7 --out results/421-name-variants.json

For every constant question whose ``must_contain`` is ``NAME = <decimal integer>`` and whose gold
chunk contains that line (default verifier settings, offline, simulated answers), the answer
``<name form> is <value>`` cites the unaltered gold chunk. Forms: ``exact`` (control), ``spaced``
(lowercase words) and ``camel`` (camelCase). Release = verified.
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
MIN_N = 20


def name_forms(name: str) -> dict[str, str]:
    """Surface forms of an ALL_CAPS constant name."""
    parts = [p for p in name.lower().split("_") if p]
    return {"exact": name, "spaced": " ".join(parts),
            "camel": parts[0] + "".join(p.capitalize() for p in parts[1:])}


def interval(k: int, n: int) -> dict:
    p, lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": p, "wilson_lo": lo, "wilson_hi": hi}


def run(data: str, questions: str, seed: int) -> dict:
    chunks = list(load_corpus(data))
    by_doc: dict[str, list] = {}
    for c in chunks:
        by_doc.setdefault(c.doc, []).append(c)
    qs = [q for q in load_questions(questions) if q.get("type") == "constant"]
    counts: dict[str, list[int]] = {}
    skipped = 0
    for q in qs:
        m = LINE.match(q["must_contain"])
        gold = next((c for c in by_doc.get(q["gold_doc"], []) if q["must_contain"] in c.text), None)
        if not m or gold is None:
            skipped += 1
            continue
        for name, form in name_forms(m.group(1)).items():
            res = verify_answer(f"{form} is {m.group(2)} [{gold.id}]", {gold.id: gold}, [gold.id], question=q["q"])
            c = counts.setdefault(name, [0, 0])
            c[0] += bool(res.ok)
            c[1] += 1
    variants = {k: interval(*v) for k, v in counts.items() if k != "exact"}
    return {
        "questions_total": len(qs), "questions_skipped": skipped,
        "control_exact": interval(*counts.get("exact", [0, 0])),
        "name_variants": {"release_rate": {k: v["rate"] for k, v in variants.items()},
                          "by_form": variants, "min_n": MIN_N,
                          "evaluable_forms": [k for k, v in variants.items() if v["n"] >= MIN_N]},
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
        "command": "python3 evals/run_name_variants.py " + " ".join(argv if argv is not None else sys.argv[1:]),
        "hardware": f"{platform.machine()} {platform.system()} cpu, python {platform.python_version()}",
        "wall_s": round(time.time() - t0, 2),
        "data_sources": [{"url": "https://github.com/ethereum/EIPs", "license": "CC0-1.0"},
                         {"url": "https://github.com/ethereum/ERCs", "license": "CC0-1.0"}],
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("control_exact", "name_variants")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
