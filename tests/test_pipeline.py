from pathlib import Path

import pytest

from ethrag import RagPipeline, load_corpus
from ethrag.llm import ScriptedLLM

DATA = Path(__file__).resolve().parents[1] / "data" / "eips"


@pytest.fixture(scope="module")
def chunks():
    return load_corpus(DATA)


def test_corpus_loads(chunks):
    assert len({c.doc for c in chunks}) >= 20


def test_eval_questions_have_gold_chunks(chunks):
    import json
    qs = [json.loads(line) for line in (DATA.parents[1] / "evals" / "questions.jsonl").read_text().splitlines() if line]
    for q in qs:
        assert any(c.doc == q["gold_doc"] and q["must_contain"].lower() in c.text.lower() for c in chunks), q["q"]


def _find(p, needle):
    return next(c.id for c in p.chunks.values() if c.doc == "eip-1559" and needle in c.text)


def test_llm_answer_verified(chunks):
    p = RagPipeline(chunks, ScriptedLLM([""]))
    cid = _find(p, "BASE_FEE_MAX_CHANGE_DENOMINATOR = 8")
    p.llm = ScriptedLLM([f"BASE_FEE_MAX_CHANGE_DENOMINATOR is 8 [{cid}]."])
    p.k = 20
    a = p.ask("What is BASE_FEE_MAX_CHANGE_DENOMINATOR in EIP-1559?")
    assert a.verified and "8" in a.text


def test_hallucinated_constant_is_withheld(chunks):
    p = RagPipeline(chunks, None, k=20)
    cid = _find(p, "BASE_FEE_MAX_CHANGE_DENOMINATOR = 8")
    p.llm = ScriptedLLM([f"BASE_FEE_MAX_CHANGE_DENOMINATOR is 16 [{cid}]."])
    a = p.ask("What is BASE_FEE_MAX_CHANGE_DENOMINATOR in EIP-1559?")
    assert not a.verified and "16" not in a.text and a.issues


def test_insufficient_context_passthrough(chunks):
    p = RagPipeline(chunks, ScriptedLLM(["INSUFFICIENT_CONTEXT"]))
    a = p.ask("What is the airspeed velocity of an unladen swallow?")
    assert not a.verified


def test_no_llm_baseline_runs_end_to_end(chunks):
    a = RagPipeline(chunks).ask("What opcode number does BASEFEE use?")
    assert a.retrieved and a.raw.endswith("].")
