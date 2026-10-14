import pytest

from ethrag import RagPipeline
from ethrag.llm import ScriptedLLM


@pytest.fixture(scope="module")
def chunks(corpus):
    return corpus


@pytest.fixture()
def pipe(corpus, corpus_index):
    return RagPipeline(corpus, None, index=corpus_index, k=20)


def test_corpus_loads(chunks):
    assert len({c.doc for c in chunks}) >= 800


def test_eval_questions_have_gold_chunks(chunks, root):
    # Questions span EIPs, ERCs, consensus specs and Flashbots docs and include multi-hop
    # records, so the whole data/ tree is loaded and every (doc, needle) pair must resolve.
    from ethrag.evalkit import gold_map, load_questions

    qs = load_questions(root / "evals" / "questions.jsonl")
    gold = gold_map(chunks, qs)
    empty = [q["id"] for q in qs if not gold[q["id"]]]
    assert not empty, f"questions with no gold chunk: {empty}"
    for q in qs:
        for doc, needle in q["gold"]:
            assert any(c.doc == doc and needle.lower() in c.text.lower() for c in chunks), q["q"]


def _find(p, needle):
    return next(c.id for c in p.chunks.values() if c.doc == "eip-1559" and needle in c.text)


def test_llm_answer_verified(pipe):
    cid = _find(pipe, "BASE_FEE_MAX_CHANGE_DENOMINATOR = 8")
    pipe.llm = ScriptedLLM([f"BASE_FEE_MAX_CHANGE_DENOMINATOR is 8 [{cid}]."])
    a = pipe.ask("What is BASE_FEE_MAX_CHANGE_DENOMINATOR in EIP-1559?")
    assert a.verified and "8" in a.text


def test_hallucinated_constant_is_withheld(pipe):
    cid = _find(pipe, "BASE_FEE_MAX_CHANGE_DENOMINATOR = 8")
    pipe.llm = ScriptedLLM([f"BASE_FEE_MAX_CHANGE_DENOMINATOR is 16 [{cid}]."])
    a = pipe.ask("What is BASE_FEE_MAX_CHANGE_DENOMINATOR in EIP-1559?")
    assert not a.verified and "16" not in a.text and a.issues


def test_insufficient_context_passthrough(corpus, corpus_index):
    p = RagPipeline(corpus, ScriptedLLM(["INSUFFICIENT_CONTEXT"]), index=corpus_index)
    a = p.ask("What is the airspeed velocity of an unladen swallow?")
    assert not a.verified


def test_no_llm_baseline_runs_end_to_end(corpus, corpus_index):
    a = RagPipeline(corpus, None, index=corpus_index).ask("What opcode number does BASEFEE use?")
    assert a.retrieved and a.raw.endswith("].")
