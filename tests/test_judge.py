import pytest

from ethrag import RagPipeline
from ethrag.judge import (
    JUDGE_SYSTEM,
    LLMJudge,
    ScriptedJudge,
    Verdict,
    judge_available,
    judge_prompt,
    parse_verdict,
    run_judged_eval,
)
from ethrag.llm import ScriptedLLM


def test_parse_verdict_strict_json_and_clamping():
    v = parse_verdict('{"correct": true, "score": 0.9, "rationale": "same value"}')
    assert v == Verdict(True, 0.9, "same value")
    assert parse_verdict('{"correct": true, "score": 7}').score == 1.0
    assert parse_verdict('{"correct": false, "score": -3}').score == 0.0
    assert parse_verdict('{"correct": true}').score == 1.0 and parse_verdict('{"correct": false}').score == 0.0


def test_parse_verdict_tolerates_prose_around_json():
    v = parse_verdict('Sure! Here is my verdict:\n{"correct": false, "score": 0.2, "rationale": "16 != 8"}\nThanks')
    assert v.correct is False and v.score == 0.2 and "16" in v.rationale


def test_parse_verdict_unstructured_fallbacks():
    assert parse_verdict("The answer is correct.").correct is True
    assert parse_verdict("That is incorrect.").correct is False
    assert parse_verdict("not correct at all").correct is False
    v = parse_verdict("¯\\_(ツ)_/¯")
    assert v.correct is False and v.rationale == "unparseable verdict"
    assert parse_verdict('{"correct": ').correct is False  # truncated JSON never raises
    assert parse_verdict('{"score": "abc"}').correct is False


def test_judge_prompt_contains_all_three_fields():
    p = judge_prompt("Q?", "REF", "CAND")
    assert "Q?" in p and "REF" in p and "CAND" in p and p.rstrip().endswith("JSON verdict:")
    assert "JSON" in JUDGE_SYSTEM


def test_llm_judge_uses_the_llm_and_parses():
    llm = ScriptedLLM(['{"correct": true, "score": 1, "rationale": "ok"}'])
    v = LLMJudge(llm).judge("Q", "ref", "cand")
    assert v.correct and llm.calls[0][0] == JUDGE_SYSTEM and "cand" in llm.calls[0][1]


def test_scripted_judge_replays_and_runs_out():
    j = ScriptedJudge([Verdict(True, 1.0)])
    assert j.judge("q", "r", "c").correct and j.calls == [("q", "r", "c")]
    with pytest.raises(RuntimeError, match="ran out"):
        j.judge("q", "r", "c")


def test_judge_available_follows_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert judge_available() is False
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    assert judge_available() is True


def test_run_judged_eval_scores_only_verified_answers(mini_chunks, mini_index, cid):
    good = f"BASE_FEE_MAX_CHANGE_DENOMINATOR is 8 [{cid('eip-1559', 'BASE_FEE_MAX')}]."
    bad = f"BASE_FEE_MAX_CHANGE_DENOMINATOR is 16 [{cid('eip-1559', 'BASE_FEE_MAX')}]."
    pipe = RagPipeline(mini_chunks, ScriptedLLM([good, bad, good]), index=mini_index, k=5)
    qs = [{"id": "a", "q": "What is BASE_FEE_MAX_CHANGE_DENOMINATOR in EIP-1559?", "answer": "it is 8"},
          {"id": "b", "q": "What is BASE_FEE_MAX_CHANGE_DENOMINATOR in EIP-1559?", "answer": "it is 8"},
          {"id": "skip", "q": "no reference answer", "answer": ""},
          {"id": "c", "q": "What is BASE_FEE_MAX_CHANGE_DENOMINATOR in EIP-1559?", "answer": "it is 8"}]
    judge = ScriptedJudge([Verdict(True, 1.0, "match"), Verdict(False, 0.25, "mismatch")])
    out = run_judged_eval(pipe, qs, judge)
    assert out["questions"] == 3 and out["verified"] == 2 and out["judged_correct"] == 1
    assert out["accuracy_of_verified"] == 0.5 and out["accuracy_overall"] == pytest.approx(1 / 3)
    assert out["mean_score"] == pytest.approx(0.625)
    rows = {r["id"]: r for r in out["rows"]}
    assert rows["b"]["verified"] is False and rows["b"]["verdict"] is None  # withheld: never judged
    assert len(judge.calls) == 2


def test_run_judged_eval_with_nothing_to_judge(mini_chunks, mini_index):
    out = run_judged_eval(RagPipeline(mini_chunks, None, index=mini_index), [], ScriptedJudge([]))
    assert out["questions"] == 0 and out["accuracy_overall"] == 0.0 and out["mean_score"] == 0.0
