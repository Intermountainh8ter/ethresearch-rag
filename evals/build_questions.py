"""Rebuild evals/questions.jsonl deterministically from the corpus and the curated seeds.

    python evals/build_questions.py [--seed 20240613] [--mined 200] [--check]

Sources, all re-validated against the chunk text by the production verifier:

* ``evals/seed_questions.jsonl`` - hand-written (AI-assisted) constant, concept and multi-hop
  questions. Wrong answers the verifier would accept are repaired or the question is dropped.
* mined constants - ``NAME = value`` definitions found in table rows / code / prose of the
  corpus itself (``ethrag.evalgen.mine_constants``), asked in three tiers: exact identifier +
  document, identifier only, and identifier spelled out as words + document.
* mined multi-hop - two constants from documents linked in the citation graph.
* mined "purpose" questions - the first sentence of an Abstract/Summary section.

``--check`` rebuilds in memory and fails if the committed file differs.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ethrag.evalgen import (
    TEMPLATES_PURPOSE,
    assign_splits,
    best_cites,
    clean_sentence,
    constant_question,
    doc_label,
    first_summary_sentence,
    gold_groups,
    make_wrong,
    mine_constants,
    multihop_pairs,
    multihop_question,
    pick_wrong_at,
    split_index,
    stratified_take,
    validate_question,
    value_spans,
)
from ethrag.graph import CitationGraph
from ethrag.ingest import load_corpus

SEED = 20240613


def normalise_seed(rec: dict) -> dict:
    out = dict(rec)
    out.setdefault("wrong", "")
    out.setdefault("answer", "")
    out["origin"] = "seed"
    if not out.get("gold"):
        out["gold"] = [[out["gold_doc"], out["must_contain"]]]
    return out


def repair_wrong(rec: dict, by_doc, by_id) -> str | None:
    """Re-derive a wrong answer the verifier rejects for a seed whose own wrong answer leaks."""
    groups = gold_groups(rec, by_doc)
    cites = best_cites(rec["answer"], groups, by_id)
    if cites is None:
        return None
    old = rec["wrong"]
    if old:  # perturb the literal that differs between answer and wrong
        a, w = value_spans(rec["answer"]), value_spans(old)
        for (s, e) in a:
            lit = rec["answer"][s:e]
            if lit not in old:
                cand = pick_wrong_at(rec["answer"], (s, e), cites, by_id)
                if cand:
                    return cand
        del w
    return make_wrong(rec["answer"], cites, by_id)


def build(data: Path, seed: int, n_mined: int) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    chunks = load_corpus(data)
    by_doc, by_id = split_index(chunks)
    graph = CitationGraph(chunks)
    label = {d: doc_label(cs[0]) for d, cs in by_doc.items()}
    audit: dict = {"dropped": [], "repaired": []}
    questions: list[dict] = []

    # ---- 1. seeds -------------------------------------------------------------------------
    for line in (ROOT / "evals" / "seed_questions.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = normalise_seed(json.loads(line))
        problems = validate_question(rec, by_doc, by_id)
        if problems == ["wrong answer is supported by the gold chunk(s)"]:
            fixed = repair_wrong(rec, by_doc, by_id)
            if fixed:
                rec["wrong"] = fixed
                audit["repaired"].append(rec["id"])
                problems = validate_question(rec, by_doc, by_id)
        if rec["type"] == "concept" and rec["answer"] and not rec["wrong"] and not problems:
            cites = best_cites(rec["answer"], gold_groups(rec, by_doc), by_id)
            rec["wrong"] = make_wrong(rec["answer"], cites, by_id) or ""
        if problems:
            audit["dropped"].append((rec["id"], problems))
            continue
        questions.append(rec)

    # ---- 2. mined constants -----------------------------------------------------------------
    consts = mine_constants(chunks)
    name_docs: dict[str, set[str]] = {}
    for k in consts:
        name_docs.setdefault(k.name, set()).add(k.doc)
    seed_names = {(q["gold_doc"], q["answer"].split(" is ")[0]) for q in questions if q["type"] == "constant"}
    pool = [k for k in consts if (k.doc, k.name) not in seed_names]
    group = lambda k: "cl" if k.doc.startswith("cl-") else ("fb" if k.doc.startswith("fb-") else k.doc[:3])
    chosen = stratified_take(pool, n_mined, per_doc=3, rng=rng, group_of=group)
    chosen.sort(key=lambda k: (k.doc, k.name))
    n = 0
    for k in chosen:
        text, tier = constant_question(k, label[k.doc], len(name_docs[k.name]) == 1, rng)
        answer = f"{k.name} is {k.value}"
        rec = {"type": "constant", "tier": tier, "origin": "mined", "q": text, "gold_doc": k.doc,
               "must_contain": k.needle, "answer": answer, "wrong": "",
               "gold": [[k.doc, k.needle]]}
        groups = gold_groups(rec, by_doc)
        cites = best_cites(answer, groups, by_id)
        wrong = make_wrong(answer, cites, by_id) if cites else None
        if not wrong:
            audit["dropped"].append((f"{k.doc}:{k.name}", ["no unsupported wrong value available"]))
            continue
        rec["wrong"] = wrong
        problems = validate_question(rec, by_doc, by_id)
        if problems:
            audit["dropped"].append((f"{k.doc}:{k.name}", problems))
            continue
        n += 1
        rec["id"] = f"g-{n:03d}"
        questions.append(rec)

    # ---- 3. mined multi-hop ------------------------------------------------------------------
    pairs = multihop_pairs(consts, graph, rng, limit=70)
    m = 0
    for ka, kb, rel in pairs:
        answer = f"{ka.name} is {ka.value} and {kb.name} is {kb.value}"
        text, tier = multihop_question(ka, kb, label[ka.doc], label[kb.doc], rng,
                                       len(name_docs[kb.name]) == 1)
        rec = {"type": "multihop", "tier": tier, "origin": "mined", "relation": rel, "q": text,
               "gold": [[ka.doc, ka.needle], [kb.doc, kb.needle]], "answer": answer, "wrong": ""}
        cites = best_cites(answer, gold_groups(rec, by_doc), by_id)
        wrong = make_wrong(answer, cites, by_id, prefer_last=bool(m % 2)) if cites else None
        if not wrong:
            audit["dropped"].append((f"{ka.doc}+{kb.doc}", ["no unsupported wrong value available"]))
            continue
        rec["wrong"] = wrong
        problems = validate_question(rec, by_doc, by_id)
        if problems:
            audit["dropped"].append((f"{ka.doc}+{kb.doc}", problems))
            continue
        m += 1
        rec["id"] = f"x-{m:03d}"
        questions.append(rec)

    # ---- 4. mined purpose questions ------------------------------------------------------------
    covered = {q["gold_doc"] for q in questions if q["type"] != "multihop"}
    docs = [d for d in sorted(by_doc) if not d.startswith(("fb-", "cl-")) and d not in covered]
    rng.shuffle(docs)
    p = 0
    for d in docs:
        if p >= 70:
            break
        hit = first_summary_sentence(by_doc[d])
        if hit is None:
            continue
        c, sentence = hit
        needle = sentence if len(sentence) <= 160 else sentence[:160].rsplit(" ", 1)[0]
        text = rng.choice(TEMPLATES_PURPOSE).format(label=label[d], title=c.title)
        rec = {"type": "concept", "tier": "purpose", "origin": "mined", "q": text, "gold_doc": d,
               "must_contain": needle, "answer": clean_sentence(sentence), "wrong": "",
               "gold": [[d, needle]]}
        problems = validate_question(rec, by_doc, by_id)
        if problems:
            audit["dropped"].append((f"purpose:{d}", problems))
            continue
        cites = best_cites(rec["answer"], gold_groups(rec, by_doc), by_id)
        rec["wrong"] = (make_wrong(rec["answer"], cites, by_id) if cites else None) or ""
        p += 1
        rec["id"] = f"p-{p:03d}"
        questions.append(rec)

    for q in questions:
        q.pop("gold", None) if q["type"] != "multihop" else None
    assign_splits(questions, seed=seed, test_fraction=0.3)
    order = {"constant": 0, "concept": 1, "multihop": 2}
    questions.sort(key=lambda q: (order[q["type"]], q["id"]))
    return questions, audit


def render(questions: list[dict]) -> str:
    def row(q: dict) -> dict:
        keys = ["id", "type", "tier", "origin", "relation", "split", "q", "gold_doc", "must_contain",
                "gold", "answer", "wrong"]
        return {k: q[k] for k in keys if k in q}
    return "".join(json.dumps(row(q), ensure_ascii=False) + "\n" for q in questions)


def report(questions: list[dict], audit: dict) -> None:
    print(f"questions: {len(questions)}")
    print("by type/split:", dict(Counter((q["type"], q["split"]) for q in questions)))
    print("by origin:", dict(Counter(q["origin"] for q in questions)))
    print("by tier:", dict(Counter(q.get("tier", "-") for q in questions)))
    docs = {d for q in questions for d in ([q["gold_doc"]] if q["type"] != "multihop" else [g[0] for g in q["gold"]])}
    print(f"distinct gold documents: {len(docs)}")
    print(f"repaired seeds: {audit['repaired']}")
    print(f"dropped: {len(audit['dropped'])}")
    for ident, why in audit["dropped"][:40]:
        print("  -", ident, why)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--mined", type=int, default=200)
    ap.add_argument("--out", default=str(ROOT / "evals" / "questions.jsonl"))
    ap.add_argument("--check", action="store_true", help="fail if the committed file is stale")
    a = ap.parse_args()
    questions, audit = build(Path(a.data), a.seed, a.mined)
    text = render(questions)
    report(questions, audit)
    if a.check:
        current = Path(a.out).read_text(encoding="utf-8")
        if current != text:
            print("evals/questions.jsonl is stale: re-run evals/build_questions.py")
            return 1
        print("evals/questions.jsonl is up to date")
        return 0
    Path(a.out).write_text(text, encoding="utf-8")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
