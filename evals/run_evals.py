"""Retrieval + verifier evals over the bundled EIP corpus.

    python evals/run_evals.py [--min-hybrid-recall5 0.85] [--show-misses]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ethrag.ingest import load_corpus
from ethrag.retrieval import BM25, DenseIndex, HybridIndex
from ethrag.verify import verify_answer


def gold_chunks(chunks, q):
    return [c for c in chunks if c.doc == q["gold_doc"] and q["must_contain"].lower() in c.text.lower()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-hybrid-recall5", type=float, default=0.0)
    ap.add_argument("--show-misses", action="store_true")
    a = ap.parse_args()

    chunks = load_corpus(ROOT / "data" / "eips")
    by_id = {c.id: c for c in chunks}
    qs = [json.loads(line) for line in (ROOT / "evals" / "questions.jsonl").read_text().splitlines() if line]
    print(f"corpus: {len({c.doc for c in chunks})} EIPs, {len(chunks)} chunks   questions: {len(qs)}\n")

    systems = {"bm25": BM25(chunks), "dense(hashed)": DenseIndex(chunks), "hybrid(rrf)": HybridIndex(chunks)}
    ks = (1, 3, 5)
    print("answer-chunk retrieval (hit = a retrieved chunk from the gold EIP contains the answer text)")
    print(f"{'system':<16}" + "".join(f"{'R@' + str(k):>8}" for k in ks) + f"{'MRR@10':>9}")
    hybrid_r5 = 0.0
    for name, idx in systems.items():
        hits = {k: 0 for k in ks}
        rr = 0.0
        for q in qs:
            gold_ids = {c.id for c in gold_chunks(chunks, q)}
            ranked = [c.id for c, _ in idx.search(q["q"], 10)]
            first = next((r for r, cid in enumerate(ranked, 1) if cid in gold_ids), None)
            if first:
                rr += 1 / first
                for k in ks:
                    hits[k] += first <= k
            elif a.show_misses:
                print(f"   miss [{name}] {q['q']}")
        row = {k: hits[k] / len(qs) for k in ks}
        if name.startswith("hybrid"):
            hybrid_r5 = row[5]
        print(f"{name:<16}" + "".join(f"{row[k]:>8.3f}" for k in ks) + f"{rr / len(qs):>9.3f}")

    # Verifier: cite the gold chunk. Correct statements should pass, wrong constants should fail.
    ok = tot_ok = bad_rej = tot_bad = 0
    leaks = []
    for q in qs:
        if not q["answer"]:
            continue
        gold = gold_chunks(chunks, q)
        from ethrag.verify import claim_tokens
        cite = next((c for c in gold if claim_tokens(q["answer"]) <= claim_tokens(c.text)), gold[0])
        ids = [c.id for c in gold]
        tot_ok += 1
        ok += verify_answer(f"{q['answer']} [{cite.id}].", by_id, ids).ok
        if q["wrong"]:
            tot_bad += 1
            rejected = not verify_answer(f"{q['wrong']} [{cite.id}].", by_id, ids).ok
            bad_rej += rejected
            if not rejected:
                leaks.append(q["wrong"])
    uncited = not verify_answer("BASE_FEE_MAX_CHANGE_DENOMINATOR is 8.", by_id, ["eip-1559#000"]).ok
    fabricated = not verify_answer("It is 8 [eip-1559#999].", by_id, ["eip-1559#000"]).ok
    print("\nverifier (cite the gold chunk)")
    print(f"  correct answers accepted:   {ok}/{tot_ok}")
    print(f"  wrong constants rejected:   {bad_rej}/{tot_bad}")
    print(f"  uncited sentence rejected:  {uncited}")
    print(f"  fabricated cite rejected:   {fabricated}")
    return 0 if hybrid_r5 >= a.min_hybrid_recall5 else 1


if __name__ == "__main__":
    sys.exit(main())
