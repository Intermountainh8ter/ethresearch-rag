"""Verifier evaluation: how often does the deterministic checker accept good and bad answers?

For every question with a reference answer a family of *cases* is built from the gold chunk(s):

========== ====================================================================== ========
kind       what is checked                                                         expected
========== ====================================================================== ========
correct    the reference answer, citing its gold chunk(s)                          accept
wrong      the reference ``wrong`` answer (a plausible different value that is     reject
           NOT in the gold chunk)
swap       the value replaced by *another literal taken from the same cited        reject
           chunk* - the value is present, but bound to the wrong name
negated    the correct answer with "is" -> "is not"                                reject
vague      a fluent sentence with no checkable token ("as described above")       reject
echo       the question restated as a sentence, no value given                     reject
wrong_cit. the correct answer citing a different chunk of the same document        reject
wrong_q    a correct answer to a *different* question, cited properly              reject
uncited    the correct answer with no citation                                     reject
fake_cit.  the correct answer citing a chunk id that was never retrieved           reject
========== ====================================================================== ========

TPR is the accept rate of ``correct`` cases; the FPR of a class is its accept rate. ``wrong`` is
negative *by construction* of the data set (its value is chosen to be absent from the chunk), so
it measures little; ``swap``, ``negated``, ``echo`` and ``wrong_q`` are where the verifier can
actually fail. Correct cases are valid by construction at token level (see
``evalgen.validate_question``), so TPR measures what the extra gates cost.
"""
from __future__ import annotations

import math
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .constants import canon
from .evalgen import best_cites, gold_groups, value_spans
from .ingest import Chunk
from .verify import int_value, verify_answer

NEGATIVE_KINDS = ("wrong", "swap", "negated", "vague", "echo", "wrong_citation", "wrong_q",
                  "uncited", "fake_citation")
KINDS = ("correct", *NEGATIVE_KINDS)
VAGUE = "The specification defines this value as described in the relevant section"

# Cumulative verifier configurations, from the bare token check to the pipeline default.
CONFIGS: dict[str, dict] = {
    "tokens only": {"min_coverage": 0.0, "min_relevance": 0.0, "check_negation": False, "check_units": False,
                        "check_binding": False},
    "+ binding, units, ranges": {"min_coverage": 0.0, "min_relevance": 0.0, "check_negation": False},
    "+ content coverage 0.7": {"min_coverage": 0.7, "min_relevance": 0.0, "check_negation": False},
    "+ negation": {"min_coverage": 0.7, "min_relevance": 0.0, "check_negation": True},
    "+ question relevance 0.1 (default)": {"min_coverage": 0.7, "min_relevance": 0.1, "check_negation": True},
    "+ novelty 0.25 (optional)": {"min_coverage": 0.7, "min_relevance": 0.1, "min_novelty": 0.25,
                                  "check_negation": True},
}


@dataclass(frozen=True)
class Case:
    qid: str
    kind: str
    question: str
    answer: str
    retrieved: tuple[str, ...]

    @property
    def expect_accept(self) -> bool:
        return self.kind == "correct"


def _cited(text: str, cites: Sequence[str]) -> str:
    return f"{text.rstrip('. ')} " + " ".join(f"[{c}]" for c in cites) + "."


def _negate(answer: str) -> str | None:
    for word in (" is ", " are ", " was ", " does "):
        if word in answer:
            return answer.replace(word, word[:-1] + " not ", 1)
    return None


def _swap_literal(answer: str, cited_text: str, rng: random.Random) -> str | None:
    """Replace the answer's last literal by a *different* literal that occurs in the cited chunk."""
    spans = value_spans(answer)
    if not spans:
        return None
    a, b = spans[-1]
    old = answer[a:b]
    old_v = int_value(re.sub(r"[,_\s]", "", old).lower())
    pool = []
    for x, y in value_spans(cited_text):
        lit = cited_text[x:y]
        v = int_value(re.sub(r"[,_\s]", "", lit).lower())
        if canon(lit) != canon(old) and (v is None or v != old_v) and lit not in pool:
            pool.append(lit)
    if not pool:
        return None
    return answer[:a] + rng.choice(pool[:20]) + answer[b:]


def build_cases(q: dict, by_doc: Mapping[str, list[Chunk]], by_id: Mapping[str, Chunk],
                others: Sequence[dict], rng: random.Random) -> list[Case]:
    """All cases of one question (``q`` needs ``answer``; ``wrong`` is used when present)."""
    answer = q.get("answer", "")
    if not answer:
        return []
    groups = gold_groups(q, by_doc)
    cites = best_cites(answer, groups, dict(by_id))
    if cites is None:
        return []
    doc = q["gold"][0][0]
    siblings = [c.id for c in by_doc.get(doc, []) if c.id not in cites]
    retrieved = tuple(cites) + tuple(rng.sample(siblings, min(4, len(siblings))))
    question = q["q"]
    out = [Case(q["id"], "correct", question, _cited(answer, cites), retrieved)]

    def add(kind: str, text: str | None, ret: Sequence[str] = retrieved, cs: Sequence[str] = cites) -> None:
        if text:
            out.append(Case(q["id"], kind, question, _cited(text, cs) if cs else text, tuple(ret)))

    if q.get("wrong"):
        add("wrong", q["wrong"])
    cited_text = "\n".join(by_id[c].text for c in cites)
    add("swap", _swap_literal(answer, cited_text, rng))
    add("negated", _negate(answer))
    add("vague", VAGUE)
    add("echo", question.rstrip("?") + ", as the specification states")
    hard = [c for c in siblings if c not in cites]
    if hard:
        wrong_cite = rng.choice(hard)
        add("wrong_citation", answer, ret=retrieved, cs=[wrong_cite])
    add("uncited", answer, cs=[])
    add("fake_citation", answer, cs=["eip-0#999"])
    candidates = [o for o in others if o["id"] != q["id"] and o.get("answer")
                  and o["gold"][0][0] != doc]
    if candidates:
        o = rng.choice(candidates)
        o_cites = best_cites(o["answer"], gold_groups(o, by_doc), dict(by_id))
        if o_cites:
            add("wrong_q", o["answer"], ret=tuple(o_cites) + retrieved, cs=o_cites)
    return out


def build_all(questions: Sequence[dict], by_doc: Mapping[str, list[Chunk]], by_id: Mapping[str, Chunk],
              seed: int = 7) -> list[Case]:
    rng = random.Random(seed)
    cases: list[Case] = []
    for q in questions:
        cases.extend(build_cases(q, by_doc, by_id, questions, rng))
    return cases


def accept_flags(cases: Sequence[Case], by_id: Mapping[str, Chunk], cfg: Mapping) -> list[bool]:
    return [verify_answer(c.answer, by_id, c.retrieved, question=c.question, **cfg).ok for c in cases]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """(proportion, lo, hi): Wilson score interval; (0, 0, 0) when n == 0."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def summarize(cases: Sequence[Case], flags: Sequence[bool]) -> dict[str, dict[str, float]]:
    """Accept rate per kind: ``{kind: {"n", "accepted", "rate", "lo", "hi"}}``."""
    out: dict[str, dict[str, float]] = {}
    for kind in KINDS:
        sel = [f for c, f in zip(cases, flags, strict=True) if c.kind == kind]
        if not sel:
            continue
        p, lo, hi = wilson(sum(sel), len(sel))
        out[kind] = {"n": len(sel), "accepted": sum(sel), "rate": p, "lo": lo, "hi": hi}
    return out


def balanced(summary: Mapping[str, Mapping[str, float]], kinds: Sequence[str] = NEGATIVE_KINDS) -> float:
    """Mean accept rate over the negative classes that have cases (the pooled FPR)."""
    rates = [summary[k]["rate"] for k in kinds if k in summary]
    return sum(rates) / len(rates) if rates else 0.0


def table(rows: Mapping[str, Mapping[str, Mapping[str, float]]], kinds: Sequence[str] = KINDS) -> str:
    """Markdown table: one row per configuration, one column per case kind (accept rate)."""
    head = "| configuration | " + " | ".join(kinds) + " | pooled FPR |"
    sep = "|---|" + "---|" * (len(kinds) + 1)
    lines = [head, sep]
    for name, summary in rows.items():
        cells = [f"{summary[k]['rate']:.3f}" if k in summary else "-" for k in kinds]
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {balanced(summary):.3f} |")
    return "\n".join(lines)
