"""LLM-as-judge harness for answer quality (optional; needs an LLM, skipped without a key).

The judge is any ``LLM`` (so ``ScriptedLLM`` drives the tests and ``AnthropicLLM`` the real
run). It compares a pipeline answer with the hand-written reference and must reply with a JSON
object ``{"correct": bool, "score": 0..1, "rationale": str}``. ``parse_verdict`` is tolerant of
prose around the JSON and never raises.
"""
from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Protocol

from .llm import LLM, api_key_available
from .pipeline import RagPipeline

JUDGE_SYSTEM = (
    "You grade answers to questions about Ethereum specifications. You are given the question, a "
    "reference answer written by a human, and a candidate answer (citations in square brackets "
    "may be ignored). Decide whether the candidate states the same fact as the reference; "
    "numbers and identifiers must match exactly. Reply with ONLY a JSON object: "
    '{"correct": true|false, "score": <0..1>, "rationale": "<one sentence>"}'
)


@dataclass
class Verdict:
    correct: bool
    score: float
    rationale: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class Judge(Protocol):
    def judge(self, question: str, reference: str, candidate: str) -> Verdict: ...


def judge_prompt(question: str, reference: str, candidate: str) -> str:
    return (f"Question: {question}\n\nReference answer: {reference}\n\n"
            f"Candidate answer: {candidate}\n\nJSON verdict:")


def parse_verdict(text: str) -> Verdict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            d = json.loads(m.group(0))
            correct = bool(d.get("correct", False))
            score = float(d.get("score", 1.0 if correct else 0.0))
            return Verdict(correct, max(0.0, min(1.0, score)), str(d.get("rationale", "")))
        except (ValueError, TypeError, AttributeError):
            pass
    if "{" in text:  # a broken/truncated JSON verdict must never be read as "correct"
        return Verdict(False, 0.0, "unparseable verdict")
    low = text.lower()
    if re.search(r"\bcorrect\b", low) and not re.search(r"\b(incorrect|not correct|wrong)\b", low):
        return Verdict(True, 1.0, "unstructured verdict: 'correct'")
    return Verdict(False, 0.0, "unparseable verdict")


class LLMJudge:
    def __init__(self, llm: LLM):
        self.llm = llm

    def judge(self, question: str, reference: str, candidate: str) -> Verdict:
        return parse_verdict(self.llm.complete(JUDGE_SYSTEM, judge_prompt(question, reference, candidate)))


class ScriptedJudge:
    def __init__(self, verdicts: Sequence[Verdict]):
        self.verdicts, self.calls = list(verdicts), []

    def judge(self, question: str, reference: str, candidate: str) -> Verdict:
        self.calls.append((question, reference, candidate))
        if not self.verdicts:
            raise RuntimeError("ScriptedJudge ran out of verdicts")
        return self.verdicts.pop(0)


def judge_available() -> bool:
    return api_key_available()


def run_judged_eval(pipeline: RagPipeline, questions: Sequence[dict], judge: Judge) -> dict:
    """Ask every question with a reference answer; judge verified answers; summarise."""
    rows = []
    for q in questions:
        if not q.get("answer"):
            continue
        a = pipeline.ask(q["q"])
        verdict = judge.judge(q["q"], q["answer"], a.text) if a.verified else None
        rows.append({"id": q["id"], "verified": a.verified, "answer": a.text,
                     "verdict": verdict.to_dict() if verdict else None})
    judged = [r for r in rows if r["verdict"]]
    correct = sum(1 for r in judged if r["verdict"]["correct"])
    return {
        "questions": len(rows),
        "verified": len(judged),
        "judged_correct": correct,
        "accuracy_of_verified": correct / len(judged) if judged else 0.0,
        "accuracy_overall": correct / len(rows) if rows else 0.0,
        "mean_score": sum(r["verdict"]["score"] for r in judged) / len(judged) if judged else 0.0,
        "rows": rows,
    }
