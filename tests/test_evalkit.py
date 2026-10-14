import json

import pytest

from ethrag.evalkit import (
    SPLIT_SEED,
    format_table,
    gold_chunk_ids,
    gold_map,
    load_questions,
    misses,
    retrieval_metrics,
    split_questions,
)
from ethrag.ingest import Chunk


def write(tmp_path, records):
    p = tmp_path / "q.jsonl"
    p.write_text("\n".join(json.dumps(r) if isinstance(r, dict) else r for r in records) + "\n")
    return p


def test_load_questions_fills_defaults_and_normalises_multihop(tmp_path):
    p = write(tmp_path, [
        {"q": "one", "gold_doc": "eip-1", "must_contain": "ABC", "answer": "a", "wrong": "b"},
        "",
        {"id": "m-1", "q": "two", "gold": [["eip-1", "AAA"], ["eip-2", "BBB"]], "answer": "x", "wrong": "y"},
        {"id": "k-1", "type": "concept", "q": "three", "gold_doc": "eip-3", "must_contain": "C"},
    ])
    qs = load_questions(p)
    assert [q["id"] for q in qs] == ["q-000", "m-1", "k-1"]
    assert qs[0]["type"] == "concept" and qs[0]["gold"] == [["eip-1", "ABC"]] and qs[0]["gold_docs"] == ["eip-1"]
    assert qs[1]["type"] == "multihop" and qs[1]["gold_docs"] == ["eip-1", "eip-2"] and qs[1]["gold_doc"] == "eip-1"
    assert qs[2]["type"] == "concept"


def test_load_questions_rejects_duplicate_ids(tmp_path):
    rec = {"id": "x", "q": "q", "gold_doc": "d", "must_contain": "m"}
    with pytest.raises(ValueError, match="duplicate"):
        load_questions(write(tmp_path, [rec, rec]))


def _qs(n_per_type=20):
    out = []
    for typ in ("constant", "concept", "multihop"):
        for i in range(n_per_type):
            out.append({"id": f"{typ}-{i:03d}", "type": typ, "q": f"{typ} {i}"})
    return out


def test_split_is_deterministic_stratified_and_disjoint():
    qs = _qs()
    train, test = split_questions(qs)
    assert split_questions(qs) == (train, test)
    assert {q["id"] for q in train}.isdisjoint({q["id"] for q in test})
    assert len(train) + len(test) == len(qs)
    for typ in ("constant", "concept", "multihop"):
        assert sum(q["type"] == typ for q in test) == round(20 * 0.4)
    assert [q["id"] for q in train] == [q["id"] for q in qs if q in train]  # file order preserved


def test_split_does_not_depend_on_file_order_and_changes_with_seed():
    qs = _qs()
    a = split_questions(qs)
    b = split_questions(list(reversed(qs)))
    assert {q["id"] for q in a[1]} == {q["id"] for q in b[1]}
    c = split_questions(qs, seed=SPLIT_SEED + 1)
    assert {q["id"] for q in a[1]} != {q["id"] for q in c[1]}
    t0, t1 = split_questions(qs, test_fraction=0.0)
    assert t1 == [] and len(t0) == len(qs)


def test_gold_chunk_ids_and_gold_map(mini_chunks):
    q = {"id": "q", "gold": [["eip-1559", "elasticity_multiplier = 2"], ["eip-4844", "BLOB_TX_TYPE"]]}
    ids = gold_chunk_ids(mini_chunks, {"gold": [q["gold"][0]]})
    assert len(ids) == 1 and next(iter(ids)).startswith("eip-1559#")
    gm = gold_map(mini_chunks, [q, {"id": "none", "gold": [["eip-1559", "no such needle"]]}])
    assert {i.split("#")[0] for i in gm["q"]} == {"eip-1559", "eip-4844"} and gm["none"] == set()


def make_search(table):
    return lambda query, k: table[query][:k]


def test_retrieval_metrics_exact_values():
    qs = [{"id": "a", "q": "qa", "gold_docs": ["d1"]}, {"id": "b", "q": "qb", "gold_docs": ["d1"]},
          {"id": "c", "q": "qc", "gold_docs": ["d1"]}, {"id": "d", "q": "qd", "gold_docs": ["d1"]}]
    gold = {"a": {"d1#1"}, "b": {"d1#1"}, "c": {"d1#1"}, "d": {"d1#1"}}
    table = {"qa": ["d1#1", "x#0"],                    # rank 1
             "qb": ["x#0", "d1#1"],                    # rank 2
             "qc": ["x#0", "y#0", "z#0", "w#0", "d1#1"],  # rank 5
             "qd": ["x#0"] * 10}                       # miss
    m = retrieval_metrics(make_search(table), qs, gold, ks=(1, 3, 5), depth=10)
    assert m["R@1"] == 0.25 and m["R@3"] == 0.5 and m["R@5"] == 0.75
    assert m["MRR@10"] == pytest.approx((1 + 1 / 2 + 1 / 5 + 0) / 4)
    assert "docset@5" not in m and retrieval_metrics(make_search(table), [], gold) == {}


def test_hit_beyond_depth_does_not_count_for_mrr():
    qs = [{"id": "a", "q": "qa", "gold_docs": ["d"]}]
    ranked = ["x#0"] * 11 + ["d#1"]
    m = retrieval_metrics(lambda q, k: ranked[:k], qs, {"a": {"d#1"}}, ks=(10, 12), depth=10)
    assert m["R@10"] == 0.0 and m["R@12"] == 1.0 and m["MRR@10"] == 0.0


def test_multihop_docset_recall_requires_every_gold_doc():
    qs = [{"id": "a", "q": "qa", "gold_docs": ["d1", "d2"]}, {"id": "b", "q": "qb", "gold_docs": ["d1", "d2"]},
          {"id": "s", "q": "qs", "gold_docs": ["d1"]}]
    gold = {"a": {"d1#1", "d2#1"}, "b": {"d1#1", "d2#1"}, "s": {"d1#1"}}
    table = {"qa": ["d1#1", "d2#1"], "qb": ["d1#1", "x#0", "y#0", "d2#1"], "qs": ["d1#1", "x#0", "y#0", "z#0"]}
    m = retrieval_metrics(make_search(table), qs, gold, ks=(2, 4), depth=4)
    assert m["multihop_n"] == 2 and m["docset@2"] == 0.5 and m["docset@4"] == 1.0
    assert m["R@2"] == 1.0  # chunk-level recall still counts any gold chunk


def test_misses_lists_questions_without_gold_in_top_k():
    qs = [{"id": "a", "q": "qa"}, {"id": "b", "q": "qb"}]
    gold = {"a": {"g#1"}, "b": {"g#2"}}
    out = misses(make_search({"qa": ["g#1"], "qb": ["x#1", "y#1", "z#1", "w#1"]}), qs, gold, k=3)
    assert out == [{"id": "b", "q": "qb", "top": ["x#1", "y#1", "z#1"]}]


def test_format_table_renders_markdown():
    t = format_table([{"system": "bm25", "R@1": 0.5, "n": 3}, {"system": "hybrid", "R@1": 0.25}], ["R@1", "n"])
    assert t.splitlines()[0] == "| system | R@1 | n |"
    assert t.splitlines()[2] == "| bm25 | 0.500 | 3 |" and t.splitlines()[3] == "| hybrid | 0.250 |  |"


def test_chunk_import_keeps_gold_chunk_helpers_pure():
    chunks = [Chunk("eip-1#000", "eip-1", "t", "s", "Has NEEDLE inside")]
    assert gold_chunk_ids(chunks, {"gold": [["eip-1", "needle"]]}) == {"eip-1#000"}
    assert gold_chunk_ids(chunks, {"gold": [["eip-2", "needle"]]}) == set()


# ------------------------------------------------------------------ nDCG, per-question rows, bootstrap
def test_ndcg_rewards_ranking_all_gold_chunks_early():
    from ethrag.evalkit import aggregate, per_question

    qs = [{"id": "a", "q": "qa", "gold_docs": ["d"]}]
    gold = {"a": {"d#1", "d#2"}}
    best = per_question(lambda q, k: ["d#1", "d#2", "x#0"], qs, gold)[0]["ndcg"]
    late = per_question(lambda q, k: ["x#0", "d#1", "d#2"], qs, gold)[0]["ndcg"]
    one = per_question(lambda q, k: ["d#1", "x#0", "y#0"], qs, gold)[0]["ndcg"]
    ideal = 1 + 1 / 1.5849625007211562  # 1/log2(2) + 1/log2(3)
    assert best == pytest.approx(1.0)
    assert one == pytest.approx(1 / ideal)  # only one of the two gold chunks, but at rank 1
    assert late == pytest.approx((1 / 1.5849625007211562 + 0.5) / ideal)  # both, at ranks 2 and 3
    assert one < late < best
    assert aggregate([], (1,)) == {}
    miss = per_question(lambda q, k: ["x#0"], qs, gold)[0]
    assert miss["ndcg"] == 0.0 and miss["rr"] == 0.0 and miss["first"] is None


def test_ndcg_exact_value_for_single_gold_at_rank_two():
    from ethrag.evalkit import retrieval_metrics

    qs = [{"id": "a", "q": "qa", "gold_docs": ["d"]}]
    m = retrieval_metrics(lambda q, k: ["x#0", "d#1"], qs, {"a": {"d#1"}})
    assert m["nDCG@10"] == pytest.approx(1 / 1.5849625007211562)  # 1/log2(3)
    assert m["MRR@10"] == 0.5


def test_aggregate_reports_docset_only_for_multihop_rows():
    from ethrag.evalkit import aggregate, per_question

    qs = [{"id": "a", "q": "qa", "gold_docs": ["d1"]}, {"id": "b", "q": "qb", "gold_docs": ["d1", "d2"]}]
    gold = {"a": {"d1#1"}, "b": {"d1#1", "d2#1"}}
    rows = per_question(lambda q, k: ["d1#1", "d2#1"], qs, gold, ks=(1, 2), depth=2)
    assert "docset@1" not in rows[0] and rows[1]["docset@1"] == 0.0 and rows[1]["docset@2"] == 1.0
    agg = aggregate(rows, (1, 2), depth=2)
    assert agg["multihop_n"] == 1 and agg["docset@2"] == 1.0 and agg["nDCG@2"] > 0


def test_paired_bootstrap_detects_a_clear_difference_and_a_null_one():
    from ethrag.evalkit import metric_vector, paired_bootstrap

    a = [1.0] * 40 + [0.0] * 10
    b = [0.0] * 10 + [1.0] * 40  # same mean as ``a``, different questions
    d, lo, hi = paired_bootstrap(a, b, n_boot=500, seed=1)
    assert d == pytest.approx(0.0) and lo < 0 < hi  # same mean, so no significant difference
    d, lo, hi = paired_bootstrap([1.0] * 50, [0.0] * 25 + [1.0] * 25, n_boot=500, seed=1)
    assert d == pytest.approx(0.5) and lo > 0
    assert paired_bootstrap(a, b, n_boot=200, seed=3) == paired_bootstrap(a, b, n_boot=200, seed=3)
    with pytest.raises(ValueError):
        paired_bootstrap([1.0], [1.0, 0.0])
    with pytest.raises(ValueError):
        paired_bootstrap([], [])
    assert metric_vector([{"R@5": 1}, {"R@5": 0.0}], "R@5") == [1.0, 0.0]


def test_stored_split_overrides_the_computed_one(tmp_path):
    qs = [{"id": f"q{i}", "type": "constant", "split": "test" if i % 4 == 0 else "train"} for i in range(20)]
    train, test = split_questions(qs)
    assert [q["id"] for q in test] == ["q0", "q4", "q8", "q12", "q16"] and len(train) == 15
    qs[3]["split"] = "dev"  # an unknown value means the field is not trusted: fall back to computing
    train2, test2 = split_questions(qs)
    assert len(train2) + len(test2) == 20 and test2 != test
