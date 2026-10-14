import importlib.util
import json
import random

import pytest

from ethrag.evalgen import split_index
from ethrag.evalkit import load_questions
from ethrag.verifyeval import (
    CONFIGS,
    KINDS,
    NEGATIVE_KINDS,
    VAGUE,
    Case,
    _negate,
    _swap_literal,
    accept_flags,
    balanced,
    build_all,
    build_cases,
    summarize,
    table,
    wilson,
)

MINI_Q = [
    ("t1", "train", "What is BASE_FEE_MAX_CHANGE_DENOMINATOR in EIP-1559?", "eip-1559",
     "BASE_FEE_MAX_CHANGE_DENOMINATOR = 8", "BASE_FEE_MAX_CHANGE_DENOMINATOR is 8", "BASE_FEE_MAX_CHANGE_DENOMINATOR is 16"),
    ("t2", "train", "What is GAS_PER_BLOB for blob transactions?", "eip-4844", "GAS_PER_BLOB",
     "GAS_PER_BLOB is 2**17", "GAS_PER_BLOB is 2**18"),
    ("t3", "train", "What is COLD_SLOAD_COST in EIP-2929?", "eip-2929", "COLD_SLOAD_COST",
     "COLD_SLOAD_COST is 2100", "COLD_SLOAD_COST is 2200"),
    ("e1", "test", "What is ELASTICITY_MULTIPLIER in EIP-1559?", "eip-1559", "ELASTICITY_MULTIPLIER = 2",
     "ELASTICITY_MULTIPLIER is 2", "ELASTICITY_MULTIPLIER is 4"),
    ("e2", "test", "What is MAX_BLOB_GAS_PER_BLOCK in EIP-4844?", "eip-4844", "MAX_BLOB_GAS_PER_BLOCK",
     "MAX_BLOB_GAS_PER_BLOCK is 786432", "MAX_BLOB_GAS_PER_BLOCK is 786433"),
    ("e3", "test", "What is WARM_STORAGE_READ_COST in EIP-2929?", "eip-2929", "WARM_STORAGE_READ_COST",
     "WARM_STORAGE_READ_COST is 100", "WARM_STORAGE_READ_COST is 200"),
]


@pytest.fixture(scope="module")
def qfile(tmp_path_factory):
    recs = [{"id": i, "type": "constant", "origin": "seed", "split": sp, "q": q, "gold_doc": d, "must_contain": n,
             "answer": a, "wrong": w} for i, sp, q, d, n, a, w in MINI_Q]
    recs.append({"id": "c1", "type": "concept", "origin": "seed", "split": "test",
                 "q": "What happens to the base fee?", "gold_doc": "eip-1559",
                 "must_contain": "The base fee is not paid to the miner; it is burned",
                 "answer": "The base fee is not paid to the miner; it is burned", "wrong": ""})
    p = tmp_path_factory.mktemp("vq") / "q.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    return p


@pytest.fixture(scope="module")
def mini_cases(qfile, mini_chunks):
    by_doc, by_id = split_index(mini_chunks)
    qs = load_questions(qfile)
    return qs, by_doc, by_id, build_all(qs, by_doc, by_id)


def test_wilson_interval():
    assert wilson(0, 0) == (0.0, 0.0, 0.0)
    p, lo, hi = wilson(5, 10)
    assert p == 0.5 and lo == pytest.approx(0.2366, abs=1e-3) and hi == pytest.approx(0.7634, abs=1e-3)
    p, lo, hi = wilson(0, 10)
    assert p == 0.0 and lo == 0.0 and hi == pytest.approx(0.2775, abs=1e-3)
    p, lo, hi = wilson(10, 10)
    assert hi == 1.0 and lo == pytest.approx(0.7225, abs=1e-3)
    assert wilson(50, 100)[2] - wilson(50, 100)[1] < wilson(5, 10)[2] - wilson(5, 10)[1]  # more data, tighter


def test_every_kind_is_built_for_a_constant_question(mini_cases):
    qs, _, _, cases = mini_cases
    mine = [c for c in cases if c.qid == "t1"]
    assert {c.kind for c in mine} == set(KINDS)
    by_kind = {c.kind: c for c in mine}
    assert by_kind["correct"].answer.startswith("BASE_FEE_MAX_CHANGE_DENOMINATOR is 8 [eip-1559#")
    assert by_kind["correct"].expect_accept and not by_kind["swap"].expect_accept
    assert by_kind["wrong"].answer.startswith("BASE_FEE_MAX_CHANGE_DENOMINATOR is 16 [")
    assert "is not 8" in by_kind["negated"].answer
    assert by_kind["vague"].answer.startswith(VAGUE)
    assert by_kind["echo"].answer.startswith("What is BASE_FEE_MAX_CHANGE_DENOMINATOR in EIP-1559, as the specification")
    assert "[" not in by_kind["uncited"].answer and "[eip-0#999]" in by_kind["fake_citation"].answer
    assert "eip-0#999" not in by_kind["fake_citation"].retrieved
    swapped = by_kind["swap"].answer.split(" is ")[1].split(" ")[0]
    assert swapped not in ("8",) and swapped != "16"
    wc = by_kind["wrong_citation"].answer.rsplit("[", 1)[1].rstrip("].")
    assert wc not in by_kind["correct"].answer and wc in by_kind["wrong_citation"].retrieved
    other = by_kind["wrong_q"]
    assert other.question == qs[0]["q"] and other.answer.split(" is ")[0] != "BASE_FEE_MAX_CHANGE_DENOMINATOR"
    assert any(r.startswith("eip-") and r in other.answer for r in other.retrieved)


def test_questions_without_an_answer_or_a_swappable_value_produce_fewer_cases(mini_cases):
    qs, by_doc, by_id, _ = mini_cases
    assert build_cases({**qs[0], "answer": ""}, by_doc, by_id, qs, random.Random(0)) == []
    concept = [c for c in build_all([q for q in qs if q["id"] == "c1"], by_doc, by_id)]
    kinds = {c.kind for c in concept}
    assert "swap" not in kinds and "wrong" not in kinds and {"correct", "vague", "echo", "uncited"} <= kinds
    assert "negated" in kinds  # "... it is burned" -> "... it is not burned"
    broken = {**qs[0], "gold": [["eip-1559", "BASE_FEE_MAX_CHANGE_DENOMINATOR = 8"]], "answer": "X is 99"}
    assert build_cases(broken, by_doc, by_id, qs, random.Random(0)) == []  # answer not supported by its gold


def test_helpers_negate_and_swap():
    assert _negate("X is 8") == "X is not 8" and _negate("They are 3") == "They are not 3"
    assert _negate("nothing to flip") is None
    rng = random.Random(1)
    assert _swap_literal("X is 8", "a = 8; b = 16; c = 0x10", rng) in ("X is 16", "X is 0x10")
    assert _swap_literal("X is 8", "only 8 here", rng) is None and _swap_literal("no literal", "9", rng) is None
    assert _swap_literal("X is 0x10", "b = 16", rng) is None  # 16 == 0x10 is the same value, not a swap


def test_default_configuration_accepts_correct_and_rejects_wrong_and_uncited(mini_cases):
    _, _, by_id, cases = mini_cases
    flags = accept_flags(cases, by_id, CONFIGS["+ question relevance 0.1 (default)"])
    summary = summarize(cases, flags)
    assert summary["correct"]["rate"] == 1.0 and summary["correct"]["n"] == 7
    for kind in ("wrong", "vague", "uncited", "fake_citation", "wrong_citation"):
        assert summary[kind]["accepted"] == 0, kind
    assert 0 <= balanced(summary) <= 1 and set(summary) <= set(KINDS)


def test_each_added_gate_never_accepts_more_than_the_previous(mini_cases):
    _, _, by_id, cases = mini_cases
    prev = None
    for name in list(CONFIGS)[:5]:
        n = sum(accept_flags(cases, by_id, CONFIGS[name]))
        assert prev is None or n <= prev, name
        prev = n


def test_table_rendering_and_balanced():
    rows = {"a": {"correct": {"rate": 1.0}, "wrong": {"rate": 0.2}, "swap": {"rate": 0.4}},
            "b": {"correct": {"rate": 0.5}}}
    t = table(rows, kinds=("correct", "wrong", "swap")).splitlines()
    assert t[0].startswith("| configuration | correct | wrong | swap | pooled FPR |")
    assert t[2] == "| a | 1.000 | 0.200 | 0.400 | 0.300 |" and t[3] == "| b | 0.500 | - | - | 0.000 |"
    assert balanced(rows["a"]) == pytest.approx(0.3) and balanced({}) == 0.0
    assert set(NEGATIVE_KINDS) | {"correct"} == set(KINDS)
    assert Case("q", "correct", "?", "a", ("x",)).expect_accept


# ------------------------------------------------------------------------------- the script
@pytest.fixture(scope="module")
def script(root):
    spec = importlib.util.spec_from_file_location("run_verifier_eval_script", root / "evals" / "run_verifier_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_verifier_eval_script_prints_tables_and_writes_results(script, mini_dir, qfile, tmp_path, capsys):
    out = tmp_path / "v.json"
    rc = script.main(["--data", str(mini_dir), "--questions", str(qfile), "--out", str(out)])
    text = capsys.readouterr().out
    assert rc == 0
    for needle in ("TRAIN split", "TEST split", "tokens only", "sweep on TRAIN", "Wilson", "end to end",
                   "answers to another question that still verify"):
        assert needle in text
    res = json.loads(out.read_text())
    assert res["cases"]["test"]["correct"] == 4 and res["configs"]["test"]["+ negation"]["correct"]["rate"] == 1.0
    assert res["shipped_defaults"]["min_coverage"] == 0.7 and res["gate"]["test_tpr"] > 0.5
    assert res["e2e_extractive"]["n"] == 4 and set(res["e2e_extractive"]["by_type"]) == {"constant", "concept"}


def test_verifier_eval_gate_fails_when_thresholds_are_not_met(script, mini_dir, qfile, tmp_path, capsys):
    rc = script.main(["--data", str(mini_dir), "--questions", str(qfile), "--out", str(tmp_path / "v.json"),
                      "--no-e2e", "--min-test-tpr", "1.01"])
    assert rc == 1 and "GATE FAILED" in capsys.readouterr().out
    rc = script.main(["--data", str(mini_dir), "--questions", str(qfile), "--out", str(tmp_path / "v.json"),
                      "--no-e2e", "--max-test-fpr", "0.0"])
    assert rc in (0, 1)  # depends on whether any negative class slips through on the toy corpus
    assert "end to end" not in capsys.readouterr().out


def test_answers_question_helper(script):
    q = {"answer": "X_Y is 8"}
    assert script.answers_question(q, "X_Y is 8 [a#000].") and not script.answers_question(q, "X_Y is 9 [a#000].")
    prose = {"answer": "The base fee is burned and never paid to miners"}
    assert script.answers_question(prose, "The base fee is burned and never paid to miners [a#000].")
    assert not script.answers_question(prose, "Withdrawals are pushed [a#000].")
    assert not script.answers_question({"answer": ""}, "anything")


def test_summarize_skips_classes_without_cases():
    cases = [Case("a", "correct", "?", "x", ()), Case("b", "correct", "?", "y", ())]
    s = summarize(cases, [True, False])
    assert set(s) == {"correct"} and s["correct"]["accepted"] == 1 and s["correct"]["rate"] == 0.5
