"""Leave-one-check-out ablation of the citation verifier's false withholds, TRAIN+TEST pooled.

    python3 evals/run_leave_one_check_out.py --seed 7 --out results/135-leave-one-check-out.json

Population: questions whose offline extractive answer contains the reference value. A "false withhold" is
one of those answers the verifier refuses. For each issue type, the arm ignores that type's issues in the
verdict of the same verification run; the delta is the absolute drop in the false-withhold rate. Relevance
and coverage only fire under the strict_combined setting, so they are ablated there. Offline, deterministic.
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

from ethrag.evalkit import load_questions, split_questions
from ethrag.ingest import load_corpus
from ethrag.pipeline import RagPipeline, extractive_answer
from ethrag.retrieval import HybridIndex
from ethrag.verify import claim_tokens, content_words
from ethrag.verifyeval import wilson

SETTINGS = {
    "default": {},
    "strict_combined": {"min_coverage": 0.95, "min_novelty": 0.3, "min_relevance": 0.25},
}
ARMS = {
    "default": ["unsupported_claim", "binding", "citation", "uncited", "unit", "range", "negation", "other"],
    "strict_combined": ["relevance", "novelty", "coverage"],
}
PREFIXES = (("answer relevance", "relevance"), ("answer novelty", "novelty"),
            ("answer does not mention", "relevance"), ("answer states no value", "relevance"),
            ("unsupported claim", "unsupported_claim"), ("unbound claim", "binding"),
            ("uncited", "uncited"), ("citation", "citation"), ("unit mismatch", "unit"),
            ("range mismatch", "range"), ("malformed range", "range"), ("negation", "negation"))


def issue_type(issue: str) -> str:
    for prefix, name in PREFIXES:
        if issue.startswith(prefix):
            return name
    if "coverage" in issue:
        return "coverage"
    if "unit" in issue:
        return "unit"
    if "range" in issue:
        return "range"
    return "other"


def answers_question(q: dict, answer_text: str) -> bool:
    ref = q.get("answer", "")
    tokens = claim_tokens(ref)
    if tokens:
        return tokens <= claim_tokens(answer_text)
    words = content_words(ref)
    return bool(words) and len(words & content_words(answer_text)) / len(words) >= 0.7


def interval(k: int, n: int) -> dict:
    p, lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": p, "wilson_lo": lo, "wilson_hi": hi}


def ablate(issue_sets: list[set[str]], check: str | None) -> int:
    """Count answers still withheld when `check` is ignored (None = full verifier)."""
    return sum(1 for s in issue_sets if s - {check})


def run(data: str, questions: str, seed: int) -> dict:
    random.seed(seed)
    chunks = load_corpus(data)
    train, test = split_questions(load_questions(questions))
    index = HybridIndex(chunks)
    pipes = {k: RagPipeline(chunks, index=index, **kw) for k, kw in SETTINGS.items()}
    base = pipes["default"]
    population = 0
    issues: dict[str, list[set[str]]] = {k: [] for k in SETTINGS}
    for q in (x for x in list(train) + list(test) if x.get("answer")):
        ctx = base.retrieve(q["q"])
        raw = extractive_answer(q["q"], ctx)
        if raw == "INSUFFICIENT_CONTEXT" or not answers_question(q, raw):
            continue
        population += 1
        ids = [c.id for c in ctx]
        for k, p in pipes.items():
            issues[k].append({issue_type(i) for i in p.verify(q["q"], raw, ids).issues})
    out: dict = {"split": "train+test", "population": population, "settings": {}}
    deltas: dict[str, float] = {}
    for setting, checks in ARMS.items():
        full = interval(ablate(issues[setting], None), population)
        arms = {}
        for c in checks:
            arm = interval(ablate(issues[setting], c), population)
            arm["delta"] = full["rate"] - arm["rate"]
            arm["interval_excludes_full"] = arm["wilson_hi"] < full["rate"] or arm["wilson_lo"] > full["rate"]
            arms[c] = arm
            deltas[f"{setting}/{c}"] = arm["delta"]
        dominant = max(arms, key=lambda c: arms[c]["delta"])
        out["settings"][setting] = {"full": full, "arms": arms, "dominant_check": dominant}
    top = max(deltas, key=deltas.get)
    out["leave_one_out"] = {"max_false_withhold_delta": deltas[top], "dominant_arm": top,
                            "default_max_delta": max(deltas[f"default/{c}"] for c in ARMS["default"])}
    return out


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
        "command": "python3 evals/run_leave_one_check_out.py " + " ".join(argv if argv is not None else sys.argv[1:]),
        "hardware": f"{platform.machine()} {platform.system()} cpu, python {platform.python_version()}",
        "wall_s": round(time.time() - t0, 2),
        "data_sources": [{"url": "https://github.com/ethereum/EIPs", "license": "CC0-1.0"}],
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"population": res["population"], "leave_one_out": res["leave_one_out"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
