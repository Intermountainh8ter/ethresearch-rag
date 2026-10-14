"""LLM-as-judge answer-quality eval (optional: needs ANTHROPIC_API_KEY, otherwise skipped).

    ANTHROPIC_API_KEY=... python evals/run_judge.py [--llm] [--split test] [--limit 50]

Runs the pipeline (extractive baseline by default, ``--llm`` for the Anthropic generator) on the
chosen split and asks the judge model whether each *verified* answer states the same fact as the
hand-written reference. Numbers from this script are never pasted into the README unless the
run is reproducible with the committed question set; see docs/EVALUATION.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ethrag.evalkit import load_questions, split_questions
from ethrag.ingest import load_corpus
from ethrag.judge import LLMJudge, judge_available, run_judged_eval
from ethrag.pipeline import RagPipeline


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--questions", default=str(ROOT / "evals" / "questions.jsonl"))
    ap.add_argument("--split", choices=("train", "test", "all"), default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--llm", action="store_true", help="generate with Anthropic too (default: extractive)")
    ap.add_argument("--out", default=str(ROOT / "evals" / "judge_results.json"))
    a = ap.parse_args()
    if not judge_available():
        print("ANTHROPIC_API_KEY not set: judge eval skipped (exit 0).")
        return 0
    from ethrag.llm import AnthropicLLM

    qs = load_questions(a.questions)
    train, test = split_questions(qs)
    chosen = {"train": train, "test": test, "all": qs}[a.split]
    chosen = [q for q in chosen if q["answer"]]
    if a.limit:
        chosen = chosen[:a.limit]
    pipeline = RagPipeline(load_corpus(a.data), AnthropicLLM() if a.llm else None)
    summary = run_judged_eval(pipeline, chosen, LLMJudge(AnthropicLLM(max_tokens=200)))
    Path(a.out).write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
