"""Verifier release rate when only a different document's chunk supports the claim.

    python3 evals/run_pooled_citation_gap.py --seed 7 --out results/374-pooled-citation-gap.json

For every constant question whose ``must_contain`` line also occurs in a chunk of a *different*
document (default verifier settings, offline, simulated answers):

* ``pooled_citation``: the answer cites [gold chunk with the value line removed, other-document
  chunk with the line]. Release = verified. ``per_citation_reject`` counts the same answers
  checked against the first-cited chunk alone (it lacks the claim, so a per-citation check rejects).
* ``control``: the unaltered gold chunk only. False-withhold = not verified.
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


def strip_line(text: str, value_line: str) -> str:
    """Remove every line containing ``value_line``."""
    return "\n".join(ln for ln in text.split("\n") if value_line not in ln)


def interval(k: int, n: int) -> dict:
    p, lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": p, "wilson_lo": lo, "wilson_hi": hi}


def run(data: str, questions: str, seed: int) -> dict:
    rng = random.Random(seed)
    chunks = list(load_corpus(data))
    by_doc: dict[str, list] = {}
    for c in chunks:
        by_doc.setdefault(c.doc, []).append(c)
    qs = [q for q in load_questions(questions) if q.get("type") == "constant"]
    n = released = ctrl_ok = per_cite_reject = first_lacks = skipped = 0
    for q in qs:
        mc = q["must_contain"]
        gold = next((c for c in by_doc.get(q["gold_doc"], []) if mc in c.text), None)
        others = [c for c in chunks if c.doc != q["gold_doc"] and mc in c.text]
        if gold is None or not others:
            skipped += 1
            continue
        other = rng.choice(others)
        stripped = dataclasses.replace(gold, text=strip_line(gold.text, mc))
        n += 1
        first_lacks += mc not in stripped.text
        ans = q["answer"]
        res = verify_answer(f"{ans} [{stripped.id}] [{other.id}]", {stripped.id: stripped, other.id: other},
                            [stripped.id, other.id], question=q["q"])
        released += bool(res.ok)
        alone = verify_answer(f"{ans} [{stripped.id}]", {stripped.id: stripped}, [stripped.id], question=q["q"])
        per_cite_reject += not alone.ok
        ctrl = verify_answer(f"{ans} [{gold.id}]", {gold.id: gold}, [gold.id], question=q["q"])
        ctrl_ok += bool(ctrl.ok)
    iv = interval(released, n)
    return {
        "questions_used": n, "questions_skipped": skipped,
        "pooled_citation": {"release_rate": iv["rate"], "wilson": iv,
                            "first_cited_lacks_claim": first_lacks,
                            "first_cited_alone_rejected": per_cite_reject},
        "answerable_control": {"false_withhold_rate": 1 - ctrl_ok / n if n else 0.0},
        "note": "constructed, simulated answers; the value line is removed from the gold chunk and supplied by a chunk of another document",
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
        "command": "python3 evals/run_pooled_citation_gap.py " + " ".join(argv if argv is not None else sys.argv[1:]),
        "hardware": f"{platform.machine()} {platform.system()} cpu, python {platform.python_version()}",
        "wall_s": round(time.time() - t0, 2),
        "data_sources": [{"url": "https://github.com/ethereum/EIPs", "license": "CC0-1.0"},
                         {"url": "https://github.com/ethereum/ERCs", "license": "CC0-1.0"}],
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("questions_used", "pooled_citation", "answerable_control")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
