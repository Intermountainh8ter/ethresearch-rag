"""Verifier release rate when the cited integer value is restated in an equivalent spelling.

    python3 evals/run_spelling_variants.py --seed 7 --out results/419-spelling-variants.json

For every constant question whose ``must_contain`` is ``NAME = <decimal integer>`` and whose gold
chunk contains that line (default verifier settings, offline, simulated answers), the answer
``NAME is <spelling>`` cites the unaltered gold chunk. Spellings: ``exact`` (control, decimal as
in the source), ``hex`` (0x...), ``pow2`` (2**n, powers of two >= 2 only) and ``comma``
(thousands separators, values >= 1000 only). Release = verified.
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


def spellings(value: int) -> dict[str, str]:
    """Equivalent spellings of ``value`` that apply to it."""
    out = {"exact": str(value), "hex": hex(value)}
    if value >= 2 and value & (value - 1) == 0:
        out["pow2"] = f"2**{value.bit_length() - 1}"
    if value >= 1000:
        out["comma"] = f"{value:,}"
    return out


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
        for name, spelled in spellings(int(m.group(2))).items():
            res = verify_answer(f"{m.group(1)} is {spelled} [{gold.id}]", {gold.id: gold}, [gold.id], question=q["q"])
            c = counts.setdefault(name, [0, 0])
            c[0] += bool(res.ok)
            c[1] += 1
    variants = {k: interval(*v) for k, v in counts.items() if k != "exact"}
    rates = [v["rate"] for v in variants.values() if v["n"] > 0]
    evaluable = [k for k, v in variants.items() if v["n"] >= MIN_N]
    return {
        "questions_total": len(qs), "questions_skipped": skipped,
        "control_exact": interval(*counts.get("exact", [0, 0])),
        "spelling_variants": {"by_spelling": variants, "min_n": MIN_N,
                              "min_release_rate": min(rates) if rates else None,
                              "falsifier_evaluable_spellings": evaluable},
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
        "command": "python3 evals/run_spelling_variants.py " + " ".join(argv if argv is not None else sys.argv[1:]),
        "hardware": f"{platform.machine()} {platform.system()} cpu, python {platform.python_version()}",
        "wall_s": round(time.time() - t0, 2),
        "data_sources": [{"url": "https://github.com/ethereum/EIPs", "license": "CC0-1.0"},
                         {"url": "https://github.com/ethereum/ERCs", "license": "CC0-1.0"}],
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("control_exact", "spelling_variants")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
