"""The committed evals/questions.jsonl must stay valid against the committed corpus."""
import json
import subprocess
import sys
from collections import Counter

import pytest

from ethrag.evalgen import split_index, validate_question
from ethrag.evalkit import gold_map, load_questions, retrieval_metrics, split_questions


@pytest.fixture(scope="module")
def qs(root):
    return load_questions(root / "evals" / "questions.jsonl")


def test_size_types_and_schema(qs):
    assert len(qs) >= 200
    assert Counter(q["type"] for q in qs).keys() == {"constant", "concept", "multihop"}
    assert sum(q["type"] == "multihop" for q in qs) >= 20
    assert len({q["id"] for q in qs}) == len(qs)
    for q in qs:
        assert q["q"].endswith("?") and q["split"] in ("train", "test") and q["origin"] in ("seed", "mined")
        assert q["answer"] or not q["wrong"]


def test_every_question_validates_against_the_corpus(qs, corpus):
    by_doc, by_id = split_index(corpus)
    bad = {q["id"]: p for q in qs if (p := validate_question(q, by_doc, by_id))}
    assert not bad, bad


def test_correct_and_wrong_pairs_cover_most_questions(qs):
    with_pair = [q for q in qs if q["answer"] and q["wrong"]]
    assert len(with_pair) >= 0.7 * len(qs)
    assert all(q["wrong"] for q in qs if q["type"] in ("constant", "multihop"))


def test_stored_split_is_70_30_and_documents_do_not_straddle(qs):
    train, test = split_questions(qs)
    assert len(train) + len(test) == len(qs)
    singles = [q for q in qs if q["type"] != "multihop"]
    share = sum(q["split"] == "test" for q in singles) / len(singles)
    assert 0.27 <= share <= 0.35
    sides = {}
    for q in singles:
        sides.setdefault(q["gold_doc"], set()).add(q["split"])
    assert all(len(v) == 1 for v in sides.values())
    assert sum(q["type"] == "multihop" for q in test) >= 10  # enough multi-hop questions in TEST


def test_every_question_has_a_gold_chunk_and_gold_sets_are_small(qs, corpus):
    gold = gold_map(corpus, qs)
    assert all(gold[q["id"]] for q in qs)
    assert sorted(len(v) for v in gold.values())[len(qs) // 2] <= 2  # median gold set


def test_multihop_questions_need_two_distinct_documents(qs):
    for q in qs:
        if q["type"] == "multihop":
            assert len(set(q["gold_docs"])) == 2


def test_questions_do_not_leak_into_the_corpus_by_id(qs):
    # sanity: no question text is a verbatim copy of its own needle (would be a copy-paste gold)
    for q in qs:
        assert q["q"].strip().lower() != q["must_contain"].strip().lower() if q.get("must_contain") else True


def test_retrieval_metrics_run_on_the_real_set(qs, corpus, corpus_index):
    gold = gold_map(corpus, qs)
    _, test = split_questions(qs)
    m = retrieval_metrics(lambda q, k: [c.id for c, _ in corpus_index.search(q, k)], test[:40], gold)
    assert 0 < m["R@10"] <= 1 and m["R@1"] <= m["R@3"] <= m["R@5"] <= m["R@10"]


def test_committed_file_matches_a_fresh_rebuild(root):
    out = subprocess.run([sys.executable, str(root / "evals" / "build_questions.py"), "--check"],
                         check=False, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stdout[-800:]


def test_every_line_is_valid_json_with_expected_keys(root):
    for line in (root / "evals" / "questions.jsonl").read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        assert {"id", "type", "origin", "split", "q", "answer", "wrong"} <= set(rec)
