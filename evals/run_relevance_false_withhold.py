"""False withholds of the citation verifier at two question-relevance thresholds, TEST split.

    python evals/run_relevance_false_withhold.py --seed 7 --out results/67-relevance-false-withhold.json

Population: TEST questions whose offline extractive answer contains the reference value. A
"false withhold" is one of those answers the verifier refuses. Compared: the shipped
``min_relevance`` 0.1 and 0.25; everything else is default. Offline and deterministic.
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
sys.path.insert(0, str(ROOT / "evals"))

from ethrag.evalkit import load_questions, split_questions
from ethrag.ingest import load_corpus
from ethrag.pipeline import RagPipeline, extractive_answer
from ethrag.retrieval import HybridIndex
from ethrag.verify import claim_tokens, content_words
from ethrag.verifyeval import wilson

THRESHOLDS = {"default_0_1": 0.1, "relevance_0_25": 0.25}


def answers_question(q: dict, answer_text: str) -> bool:
    ref = q.get("answer", "")
    tokens = claim_tokens(ref)
    if tokens:
        return tokens <= claim_tokens(answer_text)
    words = content_words(ref)
    return bool(words) and len(words & content_words(answer_text)) / len(words) >= 0.7


def issue_type(issue: str) -> str:
    for prefix, name in (("answer relevance", "relevance"), ("answer novelty", "novelty"),
                         ("answer does not mention", "relevance"), ("answer states no value", "relevance"),
                         ("unsupported claim", "unsupported_claim"), ("uncited", "uncited"),
                         ("citation", "citation"), ("unit mismatch", "unit"), ("range", "range"),
                         ("malformed range", "range"), ("negation", "negation")):
        if issue.startswith(prefix):
            return name
    if "coverage" in issue:
        return "coverage"
    return "other"


def interval(k: int, n: int) -> dict:
    p, lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": p, "wilson_lo": lo, "wilson_hi": hi}


def run(data: str, questions: str, seed: int) -> dict:
    random.seed(seed)
    chunks = load_corpus(data)
    _, test = split_questions(load_questions(questions))
    index = HybridIndex(chunks)
    pipes = {k: RagPipeline(chunks, index=index, min_relevance=v) for k, v in THRESHOLDS.items()}
    base = pipes["default_0_1"]
    population = 0
    withheld = {k: 0 for k in THRESHOLDS}
    issue_counts = {k: {} for k in THRESHOLDS}
    added = 0
    added_relevance = 0
    for q in (x for x in test if x.get("answer")):
        ctx = base.retrieve(q["q"])
        raw = extractive_answer(q["q"], ctx)
        if raw == "INSUFFICIENT_CONTEXT" or not answers_question(q, raw):
            continue
        population += 1
        ids = [c.id for c in ctx]
        res = {k: p.verify(q["q"], raw, ids) for k, p in pipes.items()}
        for k, r in res.items():
            if not r.ok:
                withheld[k] += 1
                for t in {issue_type(i) for i in r.issues}:
                    issue_counts[k][t] = issue_counts[k].get(t, 0) + 1
        if res["relevance_0_25"].ok is False and res["default_0_1"].ok:
            added += 1
            added_relevance += any(issue_type(i) == "relevance" for i in res["relevance_0_25"].issues)
    return {
        "split": "test",
        "population": population,
        "false_withhold_rate": {k: interval(withheld[k], population)["rate"] for k in THRESHOLDS},
        "false_withhold_wilson": {k: interval(withheld[k], population) for k in THRESHOLDS},
        "issue_type_counts": issue_counts,
        "added_withholds": {"n": added, "with_relevance_issue": added_relevance,
                            "relevance_majority": added > 0 and added_relevance * 2 > added},
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
        "command": "python3 evals/run_relevance_false_withhold.py " + " ".join(argv if argv is not None else sys.argv[1:]),
        "hardware": f"{platform.machine()} {platform.system()} cpu, python {platform.python_version()}",
        "wall_s": round(time.time() - t0, 2),
        "data_sources": [{"url": "https://github.com/ethereum/EIPs", "license": "CC0-1.0"}],
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("population", "false_withhold_rate", "added_withholds")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
