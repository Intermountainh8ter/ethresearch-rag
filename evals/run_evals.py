"""Retrieval evals over the bundled corpus with a stored train/test split.

    python evals/run_evals.py [--data DIR | --index DIR] [--min-test-recall5 0.80]
                              [--show-misses] [--out evals/results.json]
                              [--reranker-out evals/artifacts/reranker.json]

Everything that is *tuned* is tuned on the TRAIN split only: the fusion weight and EIP-boost of
the hybrid retriever (grid sweep) and the logistic reranker (weights fit on train candidate
pools, L2 strength chosen by document-grouped cross-validation inside train). Both splits are
reported, the CI gate reads the HELD-OUT hybrid R@5, and paired bootstrap intervals say whether
the differences between systems on the test split are distinguishable from noise.
See docs/EVALUATION.md.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ethrag.evalkit import (
    aggregate,
    format_table,
    gold_map,
    load_questions,
    metric_vector,
    misses,
    paired_bootstrap,
    per_question,
    split_questions,
)
from ethrag.ingest import load_corpus
from ethrag.rerank import LogisticReranker, build_pools, cv_select_l2, stack_pools
from ethrag.retrieval import HybridIndex

KS = (1, 3, 5, 10)
COLUMNS = ["R@1", "R@3", "R@5", "R@10", "MRR@10", "nDCG@10"]
WEIGHTS = [round(0.1 * i, 1) for i in range(11)]
BOOSTS = [0.0, 0.5, 1.0, 2.0, 4.0]
L2_GRID = [1e-3, 1e-2, 1e-1, 1.0]
POOL = 50


def banner(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def build_index(a: argparse.Namespace) -> HybridIndex:
    if a.index:
        from ethrag.store import load_index

        return load_index(a.index)
    # The dense side is swappable through the Embedder protocol. Default stays
    # HashedEmbedder so the committed numbers reproduce with no extra deps; the
    # neural option exists to answer whether a real embedder beats BM25 at all.
    embedder = None
    if getattr(a, "embedder", "hashed") == "neural":
        from ethrag.retrieval import SentenceTransformerEmbedder

        embedder = SentenceTransformerEmbedder(a.embedder_model)
    return HybridIndex(load_corpus(a.data), embedder=embedder)


class FusionCache:
    """Per-query RRF vectors computed once, so a (w, boost) grid costs one fuse per cell."""

    def __init__(self, index: HybridIndex, questions):
        self.index = index
        self.cache = {}
        for q in questions:
            rb, rd = index.rank_vectors(q["q"])
            self.cache[q["q"]] = (rb, rd, index.plan(q["q"]).docs)

    def search(self, w: float, boost: float):
        def fn(query: str, k: int) -> list[str]:
            rb, rd, docs = self.cache[query]
            fused = self.index.fuse(rb, rd, docs, bm25_weight=w, eip_boost=boost, eip_mode="boost")
            order = np.argsort(-fused, kind="stable")[:k]
            return [self.index.chunks[i].id for i in order if fused[i] > 0]
        return fn


def graph_budget_search(index: HybridIndex, w: float, boost: float):
    """Budget-neutral graph expansion: top-(k-e) hybrid hits + e `requires` neighbours."""
    def fn(q: str, k: int) -> list[str]:
        extra = 2 if k >= 5 else (1 if k >= 3 else 0)
        hits = index.search(q, k - extra, expand_graph=extra > 0, extra=extra, bm25_weight=w,
                            eip_boost=boost)
        return [c.id for c, _ in hits]
    return fn


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--index", default=None)
    ap.add_argument("--questions", default=str(ROOT / "evals" / "questions.jsonl"))
    ap.add_argument("--min-test-recall5", "--min-test-hybrid-recall5", dest="min_test_recall5",
                    type=float, default=0.0, help="exit 1 if the hybrid retriever's TEST R@5 is lower")
    ap.add_argument("--embedder", choices=("hashed", "neural"), default="hashed",
                    help="dense side of the hybrid: 'hashed' (default, no deps, what the "
                         "committed numbers use) or 'neural' (BAAI/bge-small-en-v1.5 via "
                         "sentence-transformers, through the Embedder protocol)")
    ap.add_argument("--embedder-model", default="BAAI/bge-small-en-v1.5",
                    help="model id when --embedder neural")
    ap.add_argument("--show-misses", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=2000, help="bootstrap resamples (0 = skip)")
    ap.add_argument("--out", default=str(ROOT / "evals" / "results.json"))
    ap.add_argument("--reranker-out", default=str(ROOT / "evals" / "artifacts" / "reranker.json"))
    a = ap.parse_args(argv)

    t0 = time.time()
    index = build_index(a)
    chunks = index.chunks
    qs = load_questions(a.questions)
    train, test = split_questions(qs)
    gold = gold_map(chunks, qs)
    missing = [q["id"] for q in qs if not gold[q["id"]]]
    if missing:
        print(f"ERROR: {len(missing)} questions have no gold chunk: {missing[:10]}")
        return 1
    if not train or not test:
        print("ERROR: need at least one train and one test question")
        return 1
    print(f"corpus: {len(index.graph.docs)} docs, {len(chunks)} chunks   questions: {len(qs)} "
          f"(train {len(train)} / test {len(test)}; "
          f"{sum(q['type'] == 'multihop' for q in qs)} multi-hop)   index ready in {time.time() - t0:.1f}s")
    results: dict = {
        "corpus": {"docs": len(index.graph.docs), "chunks": len(chunks)},
        "questions": {"total": len(qs), "train": len(train), "test": len(test),
                      "multihop_train": sum(q["type"] == "multihop" for q in train),
                      "multihop_test": sum(q["type"] == "multihop" for q in test)},
    }

    def ids(hits):
        return [c.id for c, _ in hits]

    dense_label = ("dense(hashed)" if getattr(a, "embedder", "hashed") == "hashed"
                   else f"dense({a.embedder_model})")

    # ---- 1. tune the hybrid on TRAIN only ---------------------------------------------------
    banner("fusion sweep on the TRAIN split (w = BM25 share of the RRF credit; boost = EIP-number boost)")
    grid = []
    cache = FusionCache(index, train)
    for boost in BOOSTS:
        for w in WEIGHTS:
            grid.append({"w": w, "boost": boost,
                         **aggregate(per_question(cache.search(w, boost), train, gold, KS), KS)})
    # A "hybrid" must fuse both rankers, so the selection only considers 0 < w < 1; the pure
    # BM25 (w=1) and pure dense (w=0) rows are still printed and reported as ablations below.
    fused_cells = [r for r in grid if 0.0 < r["w"] < 1.0]
    best = max(fused_cells, key=lambda r: (round(r["R@5"], 6), round(r["nDCG@10"], 6), -r["boost"]))
    w_star, b_star = best["w"], best["boost"]
    print(format_table([{"system": f"w={r['w']}", **r} for r in grid if r["boost"] == b_star], COLUMNS))
    print(format_table([{"system": f"boost={r['boost']}", **r} for r in grid if r["w"] == w_star], COLUMNS))
    print(f"chosen on train among 0<w<1: w*={w_star} boost*={b_star}   "
          f"(R@5 {best['R@5']:.3f}, nDCG@10 {best['nDCG@10']:.3f}); "
          f"best cell overall: w={max(grid, key=lambda r: (round(r['R@5'], 6), round(r['nDCG@10'], 6)))['w']}")
    results["fusion"] = {"w": w_star, "boost": b_star, "grid_train": grid}

    # ---- 2. fit the reranker on TRAIN only ----------------------------------------------------
    pools = build_pools(index, train, gold, pool=POOL)
    doc_of = {q["id"]: q["gold_doc"] for q in train}
    groups = [doc_of[p.qid] for p in pools]
    best_l2, cv_scores = cv_select_l2(pools, groups, L2_GRID, folds=3, k=5)
    X, y = stack_pools(pools)
    reranker = LogisticReranker().fit(X, y, l2=best_l2)
    Path(a.reranker_out).parent.mkdir(parents=True, exist_ok=True)
    reranker.save(a.reranker_out)
    print(f"\nreranker: {len(y)} train candidates ({int(y.sum())} positive), L2={best_l2} chosen by "
          f"grouped 3-fold CV on train {cv_scores}")
    top = sorted(reranker.feature_weights().items(), key=lambda kv: -abs(kv[1]))[:6]
    print("  largest standardised weights: " + ", ".join(f"{k}={v:+.2f}" for k, v in top))
    results["reranker"] = {"train_candidates": len(y), "train_positives": int(y.sum()), "l2": best_l2,
                           "cv_recall5": {str(k): v for k, v in cv_scores.items()},
                           "final_loss": reranker.history[-1] if reranker.history else None,
                           "weights": reranker.feature_weights(), "path": str(a.reranker_out)}

    # ---- 3. systems -----------------------------------------------------------------------------
    headline = {
        "bm25": lambda q, k: ids(index.bm25.search(q, k)),
        # Label follows the embedder actually in use: publishing a table that
        # says "hashed" while --embedder neural is set would misreport which
        # retriever produced the numbers.
        dense_label: lambda q, k: ids(index.dense.search(q, k)),
        "hybrid": lambda q, k: ids(index.search(q, k, bm25_weight=w_star, eip_boost=b_star)),
        "hybrid+rerank": lambda q, k: ids(reranker.rerank(index, q, k, pool=POOL)),
        "hybrid+graph": graph_budget_search(index, w_star, b_star),
    }
    ablation = {
        "bm25+eip-boost": lambda q, k: ids(index.search(q, k, bm25_weight=1.0, eip_boost=b_star)),
        "dense+eip-boost": lambda q, k: ids(index.search(q, k, bm25_weight=0.0, eip_boost=b_star)),
        "hybrid(rrf w=0.5, no boost)": lambda q, k: ids(index.search(q, k, bm25_weight=0.5, eip_mode="off")),
        "hybrid(rrf w=0.5, boost=1)": lambda q, k: ids(index.search(q, k, bm25_weight=0.5, eip_boost=1.0)),
        "hybrid(w*, restrict to named EIP)": lambda q, k: ids(
            index.search(q, k, bm25_weight=w_star, eip_mode="restrict")),
    }
    splits = {"train": train, "test": test}
    rows_by: dict[str, dict[str, list[dict]]] = {s: {} for s in splits}
    for s, split in splits.items():
        for name, fn in {**headline, **ablation}.items():
            rows_by[s][name] = per_question(fn, split, gold, KS)

    results["retrieval"] = {}
    for s, split in splits.items():
        banner(f"retrieval, {s.upper()} split (n={len(split)}; hit = a retrieved chunk contains the gold text)")
        table = [{"system": n, **aggregate(rows_by[s][n], KS)} for n in headline]
        print(format_table(table, COLUMNS))
        results["retrieval"][s] = table
        if s == "test":
            for typ in ("constant", "concept", "multihop"):
                sub = [q for q in split if q["type"] == typ]
                if not sub:
                    continue
                ids_sub = {q["id"] for q in sub}
                line = []
                for n in headline:
                    m = aggregate([r for r in rows_by[s][n] if r["id"] in ids_sub], KS)
                    line.append(f"{n}: R@5={m['R@5']:.3f}")
                print(f"   by type [{typ} n={len(sub)}]  " + "  ".join(line))
            tiers = sorted({q.get("tier") or f"{q.get('origin', 'seed')}-{q['type']}" for q in split})
            print("   by tier (test R@5, hybrid vs bm25 vs hybrid+rerank):")
            for tier in tiers:
                sub = {q["id"] for q in split if (q.get("tier") or f"{q.get('origin', 'seed')}-{q['type']}") == tier}
                cells = [aggregate([r for r in rows_by[s][n] if r["id"] in sub], KS)["R@5"]
                         for n in ("bm25", "hybrid", "hybrid+rerank")]
                print(f"     {tier:<16} n={len(sub):>3}  {cells[0]:.3f}  {cells[1]:.3f}  {cells[2]:.3f}")
                results["retrieval"].setdefault("test_by_tier", {})[tier] = {
                    "n": len(sub), "bm25": cells[0], "hybrid": cells[1], "hybrid+rerank": cells[2]}
            if a.show_misses:
                for miss in misses(headline["hybrid"], split, gold, 5):
                    print(f"   miss [hybrid] {miss['id']}: {miss['q']}  top={miss['top']}")

    # The "purpose" tier always points at an Abstract chunk, so a reranker that learns a section
    # prior is rewarded for it. Report the test split without that tier as a sensitivity check.
    no_purpose = {q["id"] for q in test if q.get("tier") != "purpose"}
    banner(f"sensitivity: TEST split without the synthetic 'purpose' tier (n={len(no_purpose)})")
    table_np = [{"system": n, **aggregate([r for r in rows_by["test"][n] if r["id"] in no_purpose], KS)}
                for n in headline]
    print(format_table(table_np, COLUMNS))
    results["retrieval"]["test_without_purpose"] = table_np

    banner("ablations (same tuned w*/boost*; shows where hybrid gains or loses against plain BM25)")
    for s in splits:
        print(f"[{s}]")
        table = [{"system": n, **aggregate(rows_by[s][n], KS)} for n in ablation]
        print(format_table(table, COLUMNS))
        results["retrieval"][f"{s}_ablation"] = table

    # ---- 4. multi-hop doc-set recall ----------------------------------------------------------------
    banner("multi-hop questions: doc-set recall (every gold document appears in the top-k results)")
    mh_table = []
    for s in splits:
        for n in headline:
            m = aggregate([r for r in rows_by[s][n] if "docset@5" in r], KS)
            if m:
                mh_table.append({"system": f"{s}: {n}", "n": int(m["multihop_n"]),
                                 "docset@5": m["docset@5"], "docset@10": m["docset@10"],
                                 "R@10": m["R@10"]})
    print(format_table(mh_table, ["n", "docset@5", "docset@10", "R@10"]))
    results["multihop"] = mh_table

    # ---- 5. significance on TEST ----------------------------------------------------------------------
    results["bootstrap_test"] = {}
    if a.bootstrap:
        banner(f"paired bootstrap on the TEST split ({a.bootstrap} resamples, 95% CI of the difference)")
        for left, right in (("hybrid", "bm25"), ("hybrid", "bm25+eip-boost"),
                            ("hybrid+rerank", "hybrid"), ("hybrid+graph", "hybrid")):
            for key, label in (("R@5", "R@5"), ("ndcg", "nDCG@10")):
                d, lo, hi = paired_bootstrap(metric_vector(rows_by["test"][left], key),
                                             metric_vector(rows_by["test"][right], key), a.bootstrap)
                verdict = "better" if lo > 0 else ("worse" if hi < 0 else "not distinguishable")
                print(f"  {left:<14} - {right:<15} {label:<8} {d:+.3f}  [{lo:+.3f}, {hi:+.3f}]  {verdict}")
                results["bootstrap_test"][f"{left} - {right} {label}"] = {"diff": d, "lo": lo, "hi": hi,
                                                                          "verdict": verdict}

    hybrid_r5 = next(r for r in results["retrieval"]["test"] if r["system"] == "hybrid")["R@5"]
    results["gate"] = {"system": "hybrid", "test_r5": hybrid_r5, "threshold": a.min_test_recall5}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    print(f"\nwrote {out}   total {time.time() - t0:.1f}s")
    if hybrid_r5 < a.min_test_recall5:
        print(f"GATE FAILED: test hybrid R@5 {hybrid_r5:.3f} < {a.min_test_recall5}")
        return 1
    if a.min_test_recall5:
        print(f"gate ok: test hybrid R@5 {hybrid_r5:.3f} >= {a.min_test_recall5}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
