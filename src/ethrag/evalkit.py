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


def retrieval_metrics(search: SearchFn, questions: Sequence[dict], gold: dict[str, set[str]],
                      ks: Sequence[int] = (1, 3, 5, 10), depth: int = 10) -> dict[str, float]:
    """R@k = fraction of questions with a gold chunk in the top k; MRR over the top ``depth``.

    For multi-hop questions ``docset@k`` is the fraction whose top-k results cover *every*
    gold document (chunk-level recall is still reported against the union of gold chunks).
    """
    n = len(questions)
    if n == 0:
        return {}
    hits = {k: 0 for k in ks}
    docset = {k: 0 for k in ks}
    rr = 0.0
    multihop = 0
    for q in questions:
        ranked = search(q["q"], max(max(ks), depth))
        gold_ids = gold[q["id"]]
        first = next((r for r, cid in enumerate(ranked, 1) if cid in gold_ids), None)
        if first is not None and first <= depth:
            rr += 1 / first
        for k in ks:
            hits[k] += bool(first is not None and first <= k)
        if len(q["gold_docs"]) > 1:
            multihop += 1
            for k in ks:
                docs = {cid.split("#")[0] for cid in ranked[:k]}
                docset[k] += set(q["gold_docs"]) <= docs
    out = {f"R@{k}": hits[k] / n for k in ks}
    out[f"MRR@{depth}"] = rr / n
    if multihop:
        out.update({f"docset@{k}": docset[k] / multihop for k in ks})
        out["multihop_n"] = multihop
    return out


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
