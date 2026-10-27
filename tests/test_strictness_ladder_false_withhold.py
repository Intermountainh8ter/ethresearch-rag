import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "run_strictness_ladder_false_withhold", Path(__file__).resolve().parents[1] / "evals" / "run_strictness_ladder_false_withhold.py")
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_issue_type_buckets():
    assert mod.issue_type("answer relevance 0.05 < 0.25: shares too few terms") == "relevance"
    assert mod.issue_type("answer does not mention ['x']") == "relevance"
    assert mod.issue_type("unsupported claim '16' not found") == "unsupported_claim"
    assert mod.issue_type("sentence 0 coverage 0.10 < 0.70: x") == "coverage"
    assert mod.issue_type("something else") == "other"


def test_interval_empty_and_full():
    assert mod.interval(0, 0)["rate"] == 0.0
    r = mod.interval(5, 10)
    assert r["rate"] == 0.5 and r["wilson_lo"] < 0.5 < r["wilson_hi"]


def test_answers_question_token_match():
    assert mod.answers_question({"answer": "8"}, "It is 8.")
    assert not mod.answers_question({"answer": "8"}, "It is 16.")


def test_ladder_is_cumulative():
    s = mod.SETTINGS
    assert s["default"] == {}
    assert s["strict_combined"] == {"min_coverage": 0.95, "min_novelty": 0.3, "min_relevance": 0.25}
    assert s["strict_novelty"].items() <= s["strict_combined"].items()
