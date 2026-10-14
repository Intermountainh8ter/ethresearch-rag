"""Eval-set loading, seeded train/test splitting and retrieval metrics (offline, numpy only).

Question records (``evals/questions.jsonl``)::

    {"id": "c-001", "type": "constant", "q": "...", "gold_doc": "eip-1559",
     "must_contain": "BASE_FEE_MAX_CHANGE_DENOMINATOR = 8",
     "answer": "BASE_FEE_MAX_CHANGE_DENOMINATOR is 8", "wrong": "... is 16"}

Multi-hop records carry ``"gold": [[doc, needle], [doc, needle]]`` instead of a single pair.
A gold *chunk* is any chunk of the gold doc whose text contains the needle (case-insensitive).
Every record of the committed file also stores ``"split": "train" | "test"`` (see
``ethrag.evalgen.assign_splits``); ``split_questions`` honours it and only falls back to a seeded
computation for files without the field.
"""
from __future__ import annotations

import json
import math
import random
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

from .ingest import Chunk

SPLIT_SEED = 20240613
TEST_FRACTION = 0.4


def load_questions(path: str | Path) -> list[dict]:
    qs = []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        q = json.loads(line)
        q.setdefault("id", f"q-{i:03d}")
        q.setdefault("type", "multihop" if q.get("gold") else "concept")
        if q.get("gold"):
            q["gold_docs"] = [d for d, _ in q["gold"]]
            q.setdefault("gold_doc", q["gold_docs"][0])
        else:
            q["gold"] = [[q["gold_doc"], q["must_contain"]]]
            q["gold_docs"] = [q["gold_doc"]]
        qs.append(q)
    ids = [q["id"] for q in qs]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate question ids")
    return qs


def split_questions(questions: Sequence[dict], seed: int = SPLIT_SEED,
                    test_fraction: float = TEST_FRACTION) -> tuple[list[dict], list[dict]]:
    """(train, test). Uses the stored ``split`` field when every record has one.

    Otherwise: a deterministic, type-stratified split (same ids -> same split, whatever the file
    order).
    """
    if questions and all(q.get("split") in ("train", "test") for q in questions):
        return ([q for q in questions if q["split"] == "train"],
                [q for q in questions if q["split"] == "test"])
    by_type: dict[str, list[dict]] = {}
    for q in sorted(questions, key=lambda q: q["id"]):
        by_type.setdefault(q["type"], []).append(q)
    train, test = [], []
    for typ in sorted(by_type):
        group = list(by_type[typ])
        random.Random(f"{seed}:{typ}").shuffle(group)
        n_test = round(len(group) * test_fraction)
        test.extend(group[:n_test])
        train.extend(group[n_test:])
    key = {q["id"]: i for i, q in enumerate(questions)}
    return sorted(train, key=lambda q: key[q["id"]]), sorted(test, key=lambda q: key[q["id"]])


def gold_chunk_ids(chunks: Iterable[Chunk], q: dict) -> set[str]:
    out = set()
    for doc, needle in q["gold"]:
        needle = needle.lower()
        out |= {c.id for c in chunks if c.doc == doc and needle in c.text.lower()}
    return out


def gold_map(chunks: Sequence[Chunk], questions: Sequence[dict]) -> dict[str, set[str]]:
    by_doc: dict[str, list[Chunk]] = {}
    for c in chunks:
        by_doc.setdefault(c.doc, []).append(c)
    out = {}
    for q in questions:
        ids = set()
        for doc, needle in q["gold"]:
            ids |= gold_chunk_ids(by_doc.get(doc, []), {"gold": [[doc, needle]]})
        out[q["id"]] = ids
    return out


SearchFn = Callable[[str, int], list[str]]  # (query, k) -> ranked chunk ids


def dcg(ranked_relevance: Sequence[int]) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(ranked_relevance))


def per_question(search: SearchFn, questions: Sequence[dict], gold: dict[str, set[str]],
                 ks: Sequence[int] = (1, 3, 5, 10), depth: int = 10) -> list[dict]:
    """One outcome row per question: first-hit rank, reciprocal rank, nDCG and hit flags.

    ``ndcg`` is binary-relevance nDCG@``depth`` against the *set* of gold chunks, so a question
    with several equivalent gold chunks rewards ranking all of them early. For multi-hop
    questions ``docset@k`` is 1 when the top-k results cover every gold document.
    """
    rows = []
    for q in questions:
        ranked = search(q["q"], max(max(ks), depth))
        gold_ids = gold[q["id"]]
        first = next((r for r, cid in enumerate(ranked, 1) if cid in gold_ids), None)
        top = ranked[:depth]
        ideal = dcg([1] * min(len(gold_ids), depth))
        row: dict = {
            "id": q["id"], "type": q.get("type", ""), "first": first,
            "rr": 1 / first if first is not None and first <= depth else 0.0,
            "ndcg": dcg([int(cid in gold_ids) for cid in top]) / ideal if ideal else 0.0,
        }
        for k in ks:
            row[f"R@{k}"] = float(first is not None and first <= k)
        if len(q["gold_docs"]) > 1:
            for k in ks:
                docs = {cid.split("#")[0] for cid in ranked[:k]}
                row[f"docset@{k}"] = float(set(q["gold_docs"]) <= docs)
        rows.append(row)
    return rows


def aggregate(rows: Sequence[dict], ks: Sequence[int] = (1, 3, 5, 10), depth: int = 10) -> dict[str, float]:
    """Mean of the per-question rows (see ``per_question``)."""
    n = len(rows)
    if n == 0:
        return {}
    out = {f"R@{k}": sum(r[f"R@{k}"] for r in rows) / n for k in ks}
    out[f"MRR@{depth}"] = sum(r["rr"] for r in rows) / n
    out[f"nDCG@{depth}"] = sum(r["ndcg"] for r in rows) / n
    multi = [r for r in rows if f"docset@{ks[0]}" in r]
    if multi:
        out.update({f"docset@{k}": sum(r[f"docset@{k}"] for r in multi) / len(multi) for k in ks})
        out["multihop_n"] = len(multi)
    return out


def retrieval_metrics(search: SearchFn, questions: Sequence[dict], gold: dict[str, set[str]],
                      ks: Sequence[int] = (1, 3, 5, 10), depth: int = 10) -> dict[str, float]:
    """R@k = fraction of questions with a gold chunk in the top k; MRR and nDCG over ``depth``.

    For multi-hop questions ``docset@k`` is the fraction whose top-k results cover *every*
    gold document (chunk-level recall is still reported against the union of gold chunks).
    """
    return aggregate(per_question(search, questions, gold, ks, depth), ks, depth)


def paired_bootstrap(a: Sequence[float], b: Sequence[float], n_boot: int = 2000, seed: int = 0,
                     alpha: float = 0.05) -> tuple[float, float, float]:
    """(mean(a-b), lo, hi): percentile bootstrap CI of the paired per-question difference."""
    if len(a) != len(b) or not a:
        raise ValueError("paired samples must be the same non-zero length")
    diffs = [x - y for x, y in zip(a, b, strict=True)]
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return sum(diffs) / n, lo, hi


def metric_vector(rows: Sequence[dict], key: str) -> list[float]:
    """Per-question values of ``key`` (``R@5``, ``rr``, ``ndcg``, ...) for ``paired_bootstrap``."""
    return [float(r[key]) for r in rows]


def misses(search: SearchFn, questions: Sequence[dict], gold: dict[str, set[str]], k: int = 5) -> list[dict]:
    out = []
    for q in questions:
        ranked = search(q["q"], k)
        if not (set(ranked) & gold[q["id"]]):
            out.append({"id": q["id"], "q": q["q"], "top": ranked[:3]})
    return out


def format_table(rows: Sequence[dict], columns: Sequence[str], name_key: str = "system",
                 fmt: str = "{:.3f}") -> str:
    head = f"| {name_key} | " + " | ".join(columns) + " |"
    sep = "|---|" + "|".join("---" for _ in columns) + "|"
    lines = [head, sep]
    for r in rows:
        cells = []
        for c in columns:
            v = r.get(c, "")
            cells.append(fmt.format(v) if isinstance(v, float) else str(v))
        lines.append(f"| {r[name_key]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
