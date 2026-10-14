import json

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


# ------------------------------------------------------------------------------- mini corpus
def test_answer_serialisation_and_provenance(mini_chunks, mini_index, repeat_llm, cid):
    reply = f"BASE_FEE_MAX_CHANGE_DENOMINATOR is 8 [{cid('eip-1559', 'BASE_FEE_MAX')}]."
    a = RagPipeline(mini_chunks, repeat_llm(reply), index=mini_index).ask(
        "What is BASE_FEE_MAX_CHANGE_DENOMINATOR in EIP-1559?")
    d = a.to_dict()
    assert d["verified"] and d["provenance"] == a.provenance and d["verification"]["ok"]
    assert cid("eip-1559", "BASE_FEE_MAX") in a.provenance
    json.dumps(d)  # fully JSON-serialisable


def test_withheld_answer_never_leaks_the_draft(mini_chunks, mini_index, repeat_llm, cid):
    reply = f"BASE_FEE_MAX_CHANGE_DENOMINATOR is 16 [{cid('eip-1559', 'BASE_FEE_MAX')}]."
    a = RagPipeline(mini_chunks, repeat_llm(reply), index=mini_index).ask(
        "What is BASE_FEE_MAX_CHANGE_DENOMINATOR in EIP-1559?")
    assert not a.verified and a.raw == reply and "16" not in a.text and "withheld" in a.text
    assert any("'16'" in i for i in a.issues) and a.verification is not None and not a.verification.ok


def test_insufficient_context_message(mini_chunks, mini_index):
    a = RagPipeline(mini_chunks, ScriptedLLM(["INSUFFICIENT_CONTEXT"]), index=mini_index).ask("blob?")
    assert a.issues == ["model reported insufficient context"] and a.verification is None
    assert a.provenance == {}


def test_k_override_and_default(mini_chunks, mini_index):
    p = RagPipeline(mini_chunks, None, index=mini_index, k=4)
    assert len(p.retrieve("blob transactions kzg commitments")) <= 4
    assert len(p.retrieve("blob transactions kzg commitments", k=1)) == 1
    assert p.ask("blob transactions kzg commitments", k=2).retrieved.__len__() <= 2 and p.k == 4


def test_graph_expansion_and_reranker_paths(mini_chunks, mini_index):
    from ethrag.rerank import LogisticReranker

    base = RagPipeline(mini_chunks, None, index=mini_index, k=1)
    exp = RagPipeline(mini_chunks, None, index=mini_index, k=1, expand_graph=True)
    q = "blob transaction GAS_PER_BLOB"
    assert len(exp.retrieve(q)) > len(base.retrieve(q))
    rr = RagPipeline(mini_chunks, None, index=mini_index, k=3, reranker=LogisticReranker())
    assert len(rr.retrieve(q)) == 3 and rr.search(q)[0][1] > 0


def test_related_and_graph_properties(mini_chunks, mini_index):
    p = RagPipeline(mini_chunks, None, index=mini_index)
    assert p.graph is mini_index.graph and ("eip-2718", 1) in p.related("eip-1559")


def test_constructors(mini_dir, tmp_path):
    from ethrag.store import save_index

    a = RagPipeline.from_directory(mini_dir, k=2)
    assert a.k == 2 and len(a.index.graph.docs) == 7
    save_index(a.index, tmp_path / "ix")
    b = RagPipeline.from_index(tmp_path / "ix", k=3)
    assert b.k == 3 and set(b.chunks) == set(a.chunks)


def test_extractive_answer_variants(mini_chunks):
    from ethrag.pipeline import extractive_answer

    ctx = [c for c in mini_chunks if c.doc == "eip-1559"]
    out = extractive_answer("Is the base fee paid to the miner or burned?", ctx)
    assert "burned" in out and out.endswith("].")
    assert extractive_answer("zzz qqq", ctx) == "INSUFFICIENT_CONTEXT"
    assert extractive_answer("anything", []) == "INSUFFICIENT_CONTEXT"
    table_only = [c for c in mini_chunks if c.doc == "eip-2929"][:1]
    assert extractive_answer("COLD_SLOAD_COST", table_only).endswith("].")


def test_generate_prompt_contains_context_ids(mini_chunks, mini_index, repeat_llm):
    llm = repeat_llm("INSUFFICIENT_CONTEXT")
    RagPipeline(mini_chunks, llm, index=mini_index).ask("What is GAS_PER_BLOB?")
    system, user = llm.calls[0]
    assert "INSUFFICIENT_CONTEXT" in system and "[eip-4844#" in user and "Question: What is GAS_PER_BLOB?" in user


def test_extractive_skips_fragments_shorter_than_three_tokens():
    from ethrag.ingest import Chunk
    from ethrag.pipeline import extractive_answer

    c = Chunk("eip-1#000", "eip-1", "T", "S", "EIP-1 T / S\nYes burn.\n\nThe fee burn is mandatory for all blocks.")
    out = extractive_answer("fee burn", [c])
    assert out.startswith("The fee burn is mandatory") and out.endswith("[eip-1#000].")
