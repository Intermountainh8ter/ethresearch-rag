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
