"""Verifier evals: TPR / FPR of the deterministic grounding check, on the stored train/test split.

    python evals/run_verifier_eval.py [--data DIR] [--min-test-tpr 0.9] [--max-test-fpr 0.05]

* Section 1 - accept rate of every case class (see ``ethrag.verifyeval``) for each cumulative
  verifier configuration, TRAIN and TEST.
* Section 2 - the two thresholds that are not obvious (content coverage, question relevance) and
  the strict-negation switch are swept on TRAIN only; the best setting by (TPR - pooled FPR) is
  reported on TEST next to the shipped defaults.
* Section 3 - end to end with the offline extractive answerer on TEST: how many answers are
  verified, how many of those actually contain the reference value, and how many answers to the
  *wrong question* get through with and without the relevance check.

Everything here is offline and deterministic. See docs/EVALUATION.md for what the numbers mean
and what the verifier cannot catch.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ethrag.evalgen import split_index
from ethrag.evalkit import load_questions, split_questions
from ethrag.ingest import load_corpus
from ethrag.pipeline import RagPipeline, extractive_answer
from ethrag.retrieval import HybridIndex
from ethrag.verify import claim_tokens, content_words
from ethrag.verifyeval import (
    CONFIGS,
    KINDS,
    NEGATIVE_KINDS,
    accept_flags,
    balanced,
    build_all,
    summarize,
    table,
)

COVERAGE = (0.5, 0.7)
RELEVANCE = (0.1, 0.25)
NOVELTY = (0.0, 0.25)
DEFAULT = {"min_coverage": 0.7, "min_relevance": 0.1, "min_novelty": 0.0, "check_negation": True}


def banner(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def score(summary) -> float:
    return summary["correct"]["rate"] - balanced(summary)


def answers_question(q: dict, answer_text: str) -> bool:
    """Does an end-to-end answer contain the reference? (all claim tokens, else content-word recall)"""
    ref = q.get("answer", "")
    tokens = claim_tokens(ref)
    if tokens:
        return tokens <= claim_tokens(answer_text)
    words = content_words(ref)
    return bool(words) and len(words & content_words(answer_text)) / len(words) >= 0.7


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--questions", default=str(ROOT / "evals" / "questions.jsonl"))
    ap.add_argument("--min-test-tpr", type=float, default=0.0)
    ap.add_argument("--max-test-fpr", type=float, default=1.0, help="ceiling for the pooled FPR on TEST")
    ap.add_argument("--no-e2e", action="store_true", help="skip the extractive end-to-end section")
    ap.add_argument("--out", default=str(ROOT / "evals" / "verifier_results.json"))
    a = ap.parse_args(argv)

    t0 = time.time()
    chunks = load_corpus(a.data)
    by_doc, by_id = split_index(chunks)
    qs = load_questions(a.questions)
    train, test = split_questions(qs)
    cases = {"train": build_all(train, by_doc, by_id), "test": build_all(test, by_doc, by_id)}
    print(f"cases: train {len(cases['train'])} / test {len(cases['test'])} from "
          f"{len(train)} / {len(test)} questions ({time.time() - t0:.1f}s)")
    results: dict = {"cases": {s: {k: sum(c.kind == k for c in v) for k in KINDS} for s, v in cases.items()}}

    # ---- 1. cumulative configurations --------------------------------------------------------
    results["configs"] = {}
    for split in ("train", "test"):
        banner(f"accept rate per case class, {split.upper()} split "
               "(correct: higher is better; every other class: lower is better)")
        rows = {name: summarize(cases[split], accept_flags(cases[split], by_id, cfg))
                for name, cfg in CONFIGS.items()}
        print(table(rows))
        n = {k: int(rows[next(iter(rows))][k]["n"]) for k in KINDS if k in rows[next(iter(rows))]}
        print("   n per class: " + ", ".join(f"{k}={v}" for k, v in n.items()))
        results["configs"][split] = rows

    # ---- 2. thresholds chosen on TRAIN --------------------------------------------------------
    banner("coverage / relevance / strict-negation sweep on TRAIN (objective: TPR - pooled FPR)")
    grid = []
    for cov in COVERAGE:
        for rel in RELEVANCE:
            for nov in NOVELTY:
                for strict in (False, True):
                    cfg = {"min_coverage": cov, "min_relevance": rel, "min_novelty": nov,
                           "check_negation": True, "strict_negation": strict}
                    s = summarize(cases["train"], accept_flags(cases["train"], by_id, cfg))
                    grid.append({"cfg": cfg, "tpr": s["correct"]["rate"], "fpr": balanced(s), "score": score(s)})
    grid.sort(key=lambda g: (-round(g["score"], 6), g["cfg"]["min_coverage"], g["cfg"]["min_novelty"]))
    for g in grid[:6]:
        c = g["cfg"]
        print(f"  coverage={c['min_coverage']} relevance={c['min_relevance']} novelty={c['min_novelty']} "
              f"strict_negation={c['strict_negation']}  TPR={g['tpr']:.3f} pooled FPR={g['fpr']:.3f} "
              f"score={g['score']:.3f}")
    best = grid[0]["cfg"]
    print(f"best on train: {best}")
    test_default = summarize(cases["test"], accept_flags(cases["test"], by_id, DEFAULT))
    test_best = summarize(cases["test"], accept_flags(cases["test"], by_id, best))
    banner("TEST split with the shipped defaults vs the setting chosen on TRAIN")
    print(table({"shipped defaults": test_default, f"best on train {best}": test_best}))
    results["sweep_train"] = grid
    results["best_on_train"] = best
    results["shipped_defaults"] = DEFAULT
    results["test_default"] = test_default
    results["test_best"] = test_best

    # Wilson intervals of the headline numbers
    banner("TEST headline numbers with 95% Wilson intervals (shipped defaults)")
    for k in KINDS:
        if k in test_default:
            r = test_default[k]
            print(f"  {k:<15} accepted {int(r['accepted']):>3}/{int(r['n']):<3} = {r['rate']:.3f}  [{r['lo']:.3f}, {r['hi']:.3f}]")

    # ---- 3. extractive end to end ----------------------------------------------------------------
    if not a.no_e2e:
        banner("end to end, offline extractive answerer, TEST split")
        index = HybridIndex(chunks)
        plain = RagPipeline(chunks, index=index, min_relevance=0.0, min_coverage=0.5)
        full = RagPipeline(chunks, index=index)
        rng = random.Random(11)
        e2e = {"n": 0, "verified_plain": 0, "verified_full": 0, "correct_plain": 0, "correct_full": 0,
               "wrong_q_plain": 0, "wrong_q_full": 0, "wrong_q_n": 0}
        pool = [q for q in test if q.get("answer")]
        by_type: dict[str, dict[str, int]] = {}
        for q in pool:
            e2e["n"] += 1
            row = by_type.setdefault(q["type"], {"n": 0, "verified": 0, "correct": 0})
            row["n"] += 1
            ctx = full.retrieve(q["q"])
            raw = extractive_answer(q["q"], ctx)
            ids = [c.id for c in ctx]
            for name, pipe in (("plain", plain), ("full", full)):
                ok = raw != "INSUFFICIENT_CONTEXT" and pipe.verify(q["q"], raw, ids).ok
                e2e[f"verified_{name}"] += ok
                e2e[f"correct_{name}"] += ok and answers_question(q, raw)
                if name == "full":
                    row["verified"] += ok
                    row["correct"] += ok and answers_question(q, raw)
            other = rng.choice([o for o in pool if o["id"] != q["id"]])
            o_ctx = full.retrieve(other["q"])
            o_raw = extractive_answer(other["q"], o_ctx)
            if o_raw != "INSUFFICIENT_CONTEXT":
                o_ids = [c.id for c in o_ctx]
                e2e["wrong_q_n"] += 1
                e2e["wrong_q_plain"] += plain.verify(q["q"], o_raw, o_ids).ok
                e2e["wrong_q_full"] += full.verify(q["q"], o_raw, o_ids).ok
        n = e2e["n"]
        print(f"  questions with a reference answer: {n}")
        print(f"  verified answers:                  without relevance check {e2e['verified_plain']}/{n}   "
              f"with {e2e['verified_full']}/{n}")
        print(f"  ...that contain the reference:     without {e2e['correct_plain']}   with {e2e['correct_full']}")
        print(f"  ...verified but NOT the answer:    without {e2e['verified_plain'] - e2e['correct_plain']}   "
              f"with {e2e['verified_full'] - e2e['correct_full']}")
        print(f"  answers to another question that still verify: without {e2e['wrong_q_plain']}/{e2e['wrong_q_n']}"
              f"   with {e2e['wrong_q_full']}/{e2e['wrong_q_n']}")
        for typ, row in sorted(by_type.items()):
            print(f"  [{typ:<9}] n={row['n']:>3}  verified {row['verified']:>3}  contain the reference {row['correct']:>3}")
        e2e["by_type"] = by_type
        results["e2e_extractive"] = e2e

    tpr = test_default["correct"]["rate"]
    fpr = balanced(test_default)
    results["gate"] = {"test_tpr": tpr, "test_pooled_fpr": fpr, "min_tpr": a.min_test_tpr, "max_fpr": a.max_test_fpr,
                       "negative_kinds": list(NEGATIVE_KINDS)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    print(f"\nwrote {a.out}   total {time.time() - t0:.1f}s")
    if tpr < a.min_test_tpr or fpr > a.max_test_fpr:
        print(f"GATE FAILED: test TPR {tpr:.3f} (min {a.min_test_tpr}), pooled FPR {fpr:.3f} (max {a.max_test_fpr})")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
