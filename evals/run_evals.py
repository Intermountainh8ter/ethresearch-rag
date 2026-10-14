"""Retrieval + verifier evals over the bundled corpus with a seeded train/test split.

    python evals/run_evals.py [--data DIR | --index DIR] [--min-test-hybrid-recall5 0.80]
                              [--show-misses] [--out evals/results.json]
                              [--reranker-out evals/artifacts/reranker.json]

Everything that is *tuned* (fusion weight, reranker weights) is tuned on the TRAIN split only and
reported on both splits. The CI gate reads the HELD-OUT hybrid R@5. See docs/EVALUATION.md.
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

from ethrag.evalkit import (  # noqa: E402
    format_table, gold_map, load_questions, misses, retrieval_metrics, split_questions,
)
from ethrag.ingest import load_corpus  # noqa: E402
from ethrag.pipeline import RagPipeline, extractive_answer  # noqa: E402
from ethrag.rerank import LogisticReranker, training_pairs  # noqa: E402
from ethrag.retrieval import BM25, DenseIndex, HybridIndex  # noqa: E402
from ethrag.verify import claim_tokens, verify_answer  # noqa: E402

KS = (1, 3, 5, 10)
COLUMNS = ["R@1", "R@3", "R@5", "R@10", "MRR@10"]
WEIGHTS = [round(0.1 * i, 1) for i in range(1, 10)]


def banner(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def build_index(a: argparse.Namespace) -> HybridIndex:
    if a.index:
        from ethrag.store import load_index

        return load_index(a.index)
    return HybridIndex(load_corpus(a.data))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--index", default=None)
    ap.add_argument("--questions", default=str(ROOT / "evals" / "questions.jsonl"))
    ap.add_argument("--min-test-hybrid-recall5", type=float, default=0.0)
    ap.add_argument("--show-misses", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "evals" / "results.json"))
    ap.add_argument("--reranker-out", default=str(ROOT / "evals" / "artifacts" / "reranker.json"))
    a = ap.parse_args()

    t0 = time.time()
    index = build_index(a)
    chunks = index.chunks
    by_id = {c.id: c for c in chunks}
    qs = load_questions(a.questions)
    train, test = split_questions(qs)
    gold = gold_map(chunks, qs)
    missing = [q["id"] for q in qs if not gold[q["id"]]]
    if missing:
        print(f"ERROR: {len(missing)} questions have no gold chunk: {missing[:10]}")
        return 1
    n_docs = len(index.graph.docs)
    print(f"corpus: {n_docs} docs, {len(chunks)} chunks   questions: {len(qs)} "
          f"(train {len(train)} / test {len(test)}; "
          f"{sum(q['type'] == 'multihop' for q in qs)} multi-hop)   index built in {time.time() - t0:.1f}s")
    results: dict = {"corpus": {"docs": n_docs, "chunks": len(chunks)},
                     "questions": {"total": len(qs), "train": len(train), "test": len(test)}}

    bm25, dense = BM25(chunks) if a.index else index.bm25, index.dense

    def ids(hits):
        return [c.id for c, _ in hits]

    systems = {
        "bm25": lambda q, k: ids(bm25.search(q, k)),
        "dense(hashed)": lambda q, k: ids(dense.search(q, k)),
        "hybrid(rrf,w=0.5,no-boost)": lambda q, k: ids(index.search(q, k, bm25_weight=0.5, eip_mode="off")),
        "hybrid+eip-boost": lambda q, k: ids(index.search(q, k, bm25_weight=0.5, eip_mode="boost")),
        "hybrid+eip-restrict": lambda q, k: ids(index.search(q, k, bm25_weight=0.5, eip_mode="restrict")),
    }

    # ---- fusion weight sweep: TRAIN only ------------------------------------------------
    banner("fusion weight sweep (train split only; w = BM25 share of the RRF credit)")
    sweep_rows = []
    for w in WEIGHTS:
        m = retrieval_metrics(lambda q, k, w=w: ids(index.search(q, k, bm25_weight=w)), train, gold, KS)
        sweep_rows.append({"system": f"w={w}", **m})
    best = max(sweep_rows, key=lambda r: (r["R@5"], r["MRR@10"]))
    best_w = float(best["system"].split("=")[1])
    print(format_table(sweep_rows, COLUMNS))
    print(f"chosen on train: w*={best_w}")
    results["fusion_sweep_train"] = sweep_rows
    results["fusion_weight"] = best_w
    systems[f"hybrid+eip-boost(w*={best_w})"] = lambda q, k: ids(index.search(q, k, bm25_weight=best_w))

    # ---- reranker: fit on TRAIN, evaluate on both ---------------------------------------
    X, y = training_pairs(index, train, gold, pool=50)
    reranker = LogisticReranker().fit(X, y)
    Path(a.reranker_out).parent.mkdir(parents=True, exist_ok=True)
    reranker.save(a.reranker_out)
    systems["hybrid+reranker(train-fit)"] = lambda q, k: ids(reranker.rerank(index, q, k, pool=50))
    results["reranker"] = {"train_pairs": int(len(y)), "train_positives": int(y.sum()),
                           "final_loss": reranker.history[-1] if reranker.history else None,
                           "weights": reranker.feature_weights(), "path": str(a.reranker_out)}

    # ---- main tables -----------------------------------------------------------------------
    results["retrieval"] = {}
    for split_name, split in (("train", train), ("test", test)):
        banner(f"answer-chunk retrieval, {split_name} split (n={len(split)}; "
               "hit = a retrieved chunk of the gold doc contains the answer text)")
        rows = []
        for name, fn in systems.items():
            m = retrieval_metrics(fn, split, gold, KS)
            rows.append({"system": name, **m})
            if a.show_misses and split_name == "test" and name.startswith("hybrid+eip-boost("):
                for miss in misses(fn, split, gold, 5):
                    print(f"   miss [{name}] {miss['id']}: {miss['q']}  top={miss['top']}")
        print(format_table(rows, COLUMNS))
        results["retrieval"][split_name] = rows
        for typ in ("constant", "concept", "multihop"):
            sub = [q for q in split if q["type"] == typ]
            if sub:
                m = retrieval_metrics(systems[f"hybrid+eip-boost(w*={best_w})"], sub, gold, KS)
                print(f"   by type [{typ:<8} n={len(sub):>3}] hybrid(w*): "
                      + "  ".join(f"{c}={m[c]:.3f}" for c in COLUMNS))
                results["retrieval"].setdefault(f"{split_name}_by_type", {})[typ] = m

    # ---- multi-hop: document-set recall with and without graph expansion ---------------------
    banner("multi-hop questions: doc-set recall (every gold doc appears in the top-k results)")
    mh_rows = []
    for split_name, split in (("train", train), ("test", test)):
        mh = [q for q in split if q["type"] == "multihop"]
        if not mh:
            continue
        variants = {
            f"{split_name}: hybrid(w*) top-10": lambda q, k: ids(index.search(q, k, bm25_weight=best_w)),
            f"{split_name}: hybrid(w*) top-8 + 2 graph neighbours": lambda q, k: ids(
                index.search(q, max(1, k - 2), expand_graph=True, extra=2, bm25_weight=best_w)),
            f"{split_name}: hybrid(w*) top-6 + 4 graph neighbours": lambda q, k: ids(
                index.search(q, max(1, k - 4), expand_graph=True, extra=4, bm25_weight=best_w)),
        }
        for name, fn in variants.items():
            m = retrieval_metrics(fn, mh, gold, (5, 10))
            mh_rows.append({"system": name, "n": len(mh), "docset@5": m.get("docset@5", 0.0),
                            "docset@10": m.get("docset@10", 0.0), "R@10": m["R@10"]})
    print(format_table(mh_rows, ["n", "docset@5", "docset@10", "R@10"]))
    results["multihop"] = mh_rows

    # ---- verifier ------------------------------------------------------------------------
    banner("verifier (cite the gold chunk; all questions with a reference answer)")
    v = {"accepted": 0, "answers": 0, "rejected_wrong": 0, "wrong": 0, "negated_rejected": 0,
         "paraphrase_rejected": 0, "swap_leaks": 0}
    leaks, false_rejects = [], []
    rng = random.Random(7)
    for q in qs:
        if not q["answer"]:
            continue
        gchunks = [by_id[i] for i in sorted(gold[q["id"]])]
        cite = next((c for c in gchunks if claim_tokens(q["answer"]) <= claim_tokens(c.text)), None)
        if cite is None:  # multi-hop: tokens split across two docs -> cite both
            cites = " ".join(f"[{c.id}]" for c in gchunks[:4])
        else:
            cites = f"[{cite.id}]"
        ret = [c.id for c in gchunks]
        v["answers"] += 1
        r = verify_answer(f"{q['answer']} {cites}.", by_id, ret, question=q["q"],
                          min_coverage=0.5, min_relevance=0.25)
        v["accepted"] += r.ok
        if not r.ok:
            false_rejects.append((q["id"], r.issues[:2]))
        if q["wrong"]:
            v["wrong"] += 1
            rw = verify_answer(f"{q['wrong']} {cites}.", by_id, ret, question=q["q"],
                               min_coverage=0.5, min_relevance=0.25)
            v["rejected_wrong"] += not rw.ok
            if rw.ok:
                leaks.append(q["id"])
        negated = q["answer"].replace(" is ", " is not ", 1)
        if negated != q["answer"]:
            rn = verify_answer(f"{negated} {cites}.", by_id, ret, question=q["q"], min_coverage=0.5)
            v["negated_rejected"] += not rn.ok
            v.setdefault("negated", 0)
            v["negated"] += 1
        rp = verify_answer(f"The specification sets this parameter as described above {cites}.", by_id,
                           ret, question=q["q"], min_coverage=0.5, min_relevance=0.0)
        v["paraphrase_rejected"] += not rp.ok
    uncited = not verify_answer("BASE_FEE_MAX_CHANGE_DENOMINATOR is 8.", by_id, ["eip-1559#000"]).ok
    fabricated = not verify_answer("It is 8 [eip-1559#999].", by_id, ["eip-1559#000"]).ok
    print(f"  correct answers accepted:            {v['accepted']}/{v['answers']}")
    print(f"  wrong constants rejected:            {v['rejected_wrong']}/{v['wrong']}")
    print(f"  negated correct answers rejected:    {v['negated_rejected']}/{v.get('negated', 0)}")
    print(f"  no-token paraphrases rejected:       {v['paraphrase_rejected']}/{v['answers']}  (min_coverage=0.5)")
    print(f"  uncited sentence rejected:           {uncited}")
    print(f"  fabricated citation rejected:        {fabricated}")
    if false_rejects:
        print(f"  false rejections: {false_rejects[:5]}")
    if leaks:
        print(f"  leaks (wrong answer accepted): {leaks}")
    results["verifier"] = {**v, "uncited_rejected": uncited, "fabricated_rejected": fabricated,
                           "false_rejections": false_rejects, "leaks": leaks}

    # ---- question relevance: can the extractive baseline pass while answering the wrong question?
    banner("question-relevance check (extractive baseline, test split)")
    pipe_plain = RagPipeline(chunks, index=index, min_relevance=0.0, min_coverage=0.5, k=5)
    pipe_rel = RagPipeline(chunks, index=index, min_relevance=0.25, min_coverage=0.5, k=5)
    rel = {"n": 0, "verified_plain": 0, "verified_rel": 0, "cites_gold_plain": 0, "cites_gold_rel": 0,
           "wrongq_pass_plain": 0, "wrongq_pass_rel": 0}
    shuffled = list(test)
    rng.shuffle(shuffled)
    for q, other in zip(test, shuffled):
        rel["n"] += 1
        ctx = pipe_plain.retrieve(q["q"])
        ret = [c.id for c in ctx]
        raw = extractive_answer(q["q"], ctx)
        for name, p in (("plain", pipe_plain), ("rel", pipe_rel)):
            r = p.verify(q["q"], raw, ret)
            rel[f"verified_{name}"] += r.ok
            rel[f"cites_gold_{name}"] += r.ok and bool(set(ret) & gold[q["id"]]) and any(
                cid in gold[q["id"]] for cid in [m.strip("[]") for m in raw.split() if m.startswith("[")])
        # answer generated for a *different* question, verified against this question
        if other["id"] != q["id"]:
            o_ctx = pipe_plain.retrieve(other["q"])
            o_raw = extractive_answer(other["q"], o_ctx)
            o_ret = [c.id for c in o_ctx]
            rel["wrongq_pass_plain"] += pipe_plain.verify(q["q"], o_raw, o_ret).ok
            rel["wrongq_pass_rel"] += pipe_rel.verify(q["q"], o_raw, o_ret).ok
    print(f"  baseline answers verified:           plain {rel['verified_plain']}/{rel['n']}   "
          f"with relevance check {rel['verified_rel']}/{rel['n']}")
    print(f"  ...of which cite a gold chunk:       plain {rel['cites_gold_plain']}   "
          f"with relevance check {rel['cites_gold_rel']}")
    print(f"  wrong-question answers that pass:    plain {rel['wrongq_pass_plain']}/{rel['n']}   "
          f"with relevance check {rel['wrongq_pass_rel']}/{rel['n']}")
    results["relevance"] = rel

    test_r5 = next(r for r in results["retrieval"]["test"] if r["system"] == "hybrid+eip-boost")["R@5"]
    results["gate"] = {"test_hybrid_r5": test_r5, "threshold": a.min_test_hybrid_recall5}
    Path(a.out).write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    print(f"\nwrote {a.out}   total {time.time() - t0:.1f}s")
    if test_r5 < a.min_test_hybrid_recall5:
        print(f"GATE FAILED: test hybrid R@5 {test_r5:.3f} < {a.min_test_hybrid_recall5}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
