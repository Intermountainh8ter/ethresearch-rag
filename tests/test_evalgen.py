import random

import pytest

from ethrag.evalgen import (
    Constant,
    assign_splits,
    best_cites,
    clean_sentence,
    constant_question,
    doc_label,
    first_summary_sentence,
    gold_groups,
    make_wrong,
    mine_constants,
    multihop_pairs,
    multihop_question,
    perturb,
    pick_wrong,
    pick_wrong_at,
    replace_last,
    split_index,
    stratified_take,
    supported,
    validate_question,
    value_spans,
)
from ethrag.graph import CitationGraph
from ethrag.ingest import Chunk, chunk_document

# ------------------------------------------------------------------------------ perturbations


def test_perturb_power_of_two_neighbours_first():
    assert perturb("2**17")[:2] == ["2**18", "2**16"]
    assert all(p != "2**1" for p in perturb("2**1"))
    assert all(int(p.split("**")[1]) >= 0 for p in perturb("2**0"))


def test_perturb_decimal_keeps_separator_style():
    assert perturb("8")[:3] == ["16", "4", "9"]
    assert perturb("2,675,000")[0] == "5,350,000"
    assert perturb("10_485_760")[0] == "20_971_520"
    assert "1" not in perturb("1") and "-1" not in perturb("1")


def test_perturb_hex_keeps_width_and_case():
    assert perturb("0x03")[:2] == ["0x04", "0x02"]
    assert perturb("0xFF") == [x for x in perturb("0xFF") if len(x) == 4]
    assert all(c.startswith("0x") and c[2:].upper() == c[2:] for c in perturb("0x0A"))
    assert perturb("0x" + "f" * 40)  # long literals flip a nibble instead of overflowing
    assert all(len(x) == 42 for x in perturb("0x" + "f" * 40))


def test_perturb_is_empty_for_unrecognised_values():
    assert perturb("banana") == [] and perturb("") == []


def test_replace_last_and_value_spans():
    assert replace_last("a 8 b 8", "8", "9") == "a 8 b 9" and replace_last("abc", "z", "y") == "abc"
    text = "uint256 EIP-1559 sets X to 0x03, Y to 2**17 and Z to 1,000"
    spans = [text[a:b] for a, b in value_spans(text)]
    assert spans == ["0x03", "2**17", "1,000"]  # no "256", no "1559"


def test_make_wrong_prefers_later_literal_and_is_unsupported(mini_chunks):
    _, by_id = split_index(mini_chunks)
    cid = next(c.id for c in mini_chunks if "BASE_FEE_MAX_CHANGE_DENOMINATOR = 8" in c.text)
    answer = "BASE_FEE_MAX_CHANGE_DENOMINATOR is 8"
    assert supported(answer, [cid], by_id)
    wrong = make_wrong(answer, [cid], by_id)
    assert wrong and wrong != answer and not supported(wrong, [cid], by_id)
    assert pick_wrong(answer, "8", [cid], by_id) == wrong
    a, b = value_spans(answer)[0]
    assert pick_wrong_at(answer, (a, b), [cid], by_id).startswith("BASE_FEE_MAX_CHANGE_DENOMINATOR is ")
    assert make_wrong("no literals here", [cid], by_id) is None


# --------------------------------------------------------------------------------------- mining


def test_mine_constants_from_rows_code_and_prose(mini_chunks):
    found = {(k.doc, k.name): k for k in mine_constants(mini_chunks)}
    k = found[("eip-4844", "GAS_PER_BLOB")]
    assert k.value == "2**17" and k.kind == "row" and "`GAS_PER_BLOB`" in k.needle
    assert found[("eip-1559", "BASE_FEE_MAX_CHANGE_DENOMINATOR")].kind == "code"
    assert found[("eip-1559", "BASE_FEE_MAX_CHANGE_DENOMINATOR")].value == "8"
    assert found[("cl-phase0-beacon-chain", "SLOTS_PER_EPOCH")].value == "2**5"
    assert found[("eip-2929", "COLD_SLOAD_COST")].value == "2100"
    assert found[("eip-4844", "BLOB_TX_TYPE")].value == "0x03"
    by_id = {c.id: c for c in mini_chunks}
    assert all(c.needle in by_id[c.chunk_id].text for c in found.values())


def test_mine_constants_skips_ambiguous_trivial_and_non_constant_rows():
    raw = ("---\ntitle: T\n---\n## Spec\n\n| Name | Value |\n|---|---|\n"
           "| `DUP_VALUE` | 5 |\n| `DUP_VALUE` | 6 |\n| `TINY_ONE` | 1 |\n| `SLOAD` | 5 |\n"
           "| `OK_VALUE` | `uint64(7)` (= 7) |\n| `EXPR_VALUE` | 2 * 3 |\n")
    names = {k.name for k in mine_constants(chunk_document("eip-77", raw))}
    assert names == {"OK_VALUE"}


def test_mine_constants_prose_form():
    raw = "---\ntitle: T\n---\n## Spec\nUse `CHAIN_ID`: 61 and `FORK_BLKNUM` is 2,675,000 here.\n"
    got = {k.name: k.value for k in mine_constants(chunk_document("eip-155", raw))}
    assert got == {"CHAIN_ID": "61", "FORK_BLKNUM": "2,675,000"}


def test_doc_label_variants(mini_chunks):
    by_doc, _ = split_index(mini_chunks)
    assert doc_label(by_doc["eip-1559"][0]) == "EIP-1559" and doc_label(by_doc["erc-20"][0]) == "ERC-20"
    assert doc_label(by_doc["cl-phase0-beacon-chain"][0]) == "the consensus spec 'Phase 0 -- The Beacon Chain'"
    fb = Chunk("fb-x#000", "fb-x", "Overview", "s", "t")
    assert doc_label(fb) == "the Flashbots doc 'Overview'"
    assert doc_label(Chunk("misc#000", "misc", "Misc", "s", "t")) == "Misc"


def test_constant_question_tiers_and_wording():
    k = Constant("eip-1", "eip-1#000", "MAX_THING_SIZE", "8", "x", "row")
    seen = {}
    for seed in range(60):
        text, tier = constant_question(k, "EIP-1", True, random.Random(seed))
        seen[tier] = text
        assert text.endswith("?")
    assert set(seen) == {"exact-id", "name-only", "words"}
    assert "MAX_THING_SIZE" in seen["exact-id"] and "max thing size" in seen["words"]
    assert "EIP-1" not in seen["name-only"]
    tiers = {constant_question(k, "EIP-1", False, random.Random(s))[1] for s in range(60)}
    assert "name-only" not in tiers


# ------------------------------------------------------------------------------------ validation


def _q(**kw):
    base = {"id": "t", "type": "constant", "q": "What is BASE_FEE_MAX_CHANGE_DENOMINATOR in EIP-1559?",
            "gold": [["eip-1559", "BASE_FEE_MAX_CHANGE_DENOMINATOR = 8"]],
            "answer": "BASE_FEE_MAX_CHANGE_DENOMINATOR is 8", "wrong": "BASE_FEE_MAX_CHANGE_DENOMINATOR is 16"}
    base.update(kw)
    return base


@pytest.fixture(scope="module")
def idx(mini_chunks):
    return split_index(mini_chunks)


def test_valid_question_has_no_problems(idx):
    assert validate_question(_q(), *idx) == []


@pytest.mark.parametrize("patch, fragment", [
    ({"type": "weird"}, "bad type"),
    ({"q": "Not a question"}, "must end with"),
    ({"gold": [["eip-1559", "NOT IN ANY CHUNK"]]}, "not found"),
    ({"gold": [["eip-9999", "x"]]}, "not found"),
    ({"answer": "BASE_FEE_MAX_CHANGE_DENOMINATOR is 7"}, "not supported"),
    ({"wrong": "BASE_FEE_MAX_CHANGE_DENOMINATOR is 8"}, "equals the correct"),
    ({"gold": [["eip-2930", "ACCESS_LIST_ADDRESS_COST = 2400"]],
      "q": "What does an address in an access list cost in EIP-2930?",
      "answer": "ACCESS_LIST_ADDRESS_COST is 2400", "wrong": "ACCESS_LIST_ADDRESS_COST is 1900"},
     "supported by the gold"),  # both values sit on one line: the verifier cannot tell them apart
    ({"answer": "", "wrong": "x is 3"}, "without a correct"),
    ({"q": "BASE_FEE_MAX_CHANGE_DENOMINATOR is 8, right?"}, "already contains"),
])
def test_each_problem_is_reported(idx, patch, fragment):
    problems = validate_question(_q(**patch), *idx)
    assert any(fragment in p for p in problems), problems


def test_concept_questions_skip_the_leak_check_and_may_lack_answers(idx):
    q = _q(type="concept", q="What does EIP-1559 do?", answer="", wrong="",
           gold=[["eip-1559", "burned"]])
    assert validate_question(q, *idx) == []
    q["answer"] = "The base fee is not paid to the miner; it is burned"
    assert validate_question(q, *idx) == []


def test_multihop_validation_cites_one_chunk_per_pair(idx):
    q = {"id": "m", "type": "multihop", "q": "What are GAS_PER_BLOB and COLD_SLOAD_COST?",
         "gold": [["eip-4844", "GAS_PER_BLOB"], ["eip-2929", "COLD_SLOAD_COST"]],
         "answer": "GAS_PER_BLOB is 2**17 and COLD_SLOAD_COST is 2100",
         "wrong": "GAS_PER_BLOB is 2**17 and COLD_SLOAD_COST is 2200"}
    assert validate_question(q, *idx) == []
    groups = gold_groups(q, idx[0])
    assert len(groups) == 2 and best_cites("GAS_PER_BLOB is 2**16", groups, idx[1]) is None


# ------------------------------------------------------------------------------ splits / sets


def _pool(n_docs=20, per=3):
    qs = []
    for d in range(n_docs):
        for i in range(per):
            qs.append({"id": f"c-{d}-{i}", "type": "constant", "gold_doc": f"eip-{d}"})
    qs += [{"id": f"m-{i}", "type": "multihop", "gold": [["eip-1", "a"], ["eip-2", "b"]]} for i in range(10)]
    return qs


def test_assign_splits_is_deterministic_and_never_splits_a_document():
    a = assign_splits(_pool(), seed=1)
    b = assign_splits(_pool(), seed=1)
    assert [q["split"] for q in a] == [q["split"] for q in b]
    sides = {}
    for q in a:
        if q["type"] != "multihop":
            sides.setdefault(q["gold_doc"], set()).add(q["split"])
    assert all(len(v) == 1 for v in sides.values())
    test_share = sum(q["split"] == "test" for q in a if q["type"] != "multihop") / 60
    assert 0.25 <= test_share <= 0.40
    assert sum(q["split"] == "test" for q in a if q["type"] == "multihop") == 3
    c = assign_splits(_pool(), seed=2)
    assert [q["split"] for q in a] != [q["split"] for q in c]


def test_multihop_pairs_follow_graph_edges(mini_chunks):
    graph = CitationGraph(mini_chunks)
    consts = mine_constants(mini_chunks)
    pairs = multihop_pairs(consts, graph, random.Random(0), limit=10)
    assert pairs
    for a, b, rel in pairs:
        assert a.doc != b.doc and a.name != b.name
        assert (b.doc in graph.requires[a.doc]) or (b.doc in graph.mentions[a.doc])
        assert rel in ("requires", "mentions")
    uses = {}
    for a, b, _ in pairs:
        for d in (a.doc, b.doc):
            uses[d] = uses.get(d, 0) + 1
    assert max(uses.values()) <= 2
    assert len(multihop_pairs(consts, graph, random.Random(0), limit=1)) == 1


def test_multihop_question_tiers():
    a = Constant("eip-1", "eip-1#000", "AAA_SIZE", "3", "x", "row")
    b = Constant("eip-2", "eip-2#000", "BBB_SIZE", "4", "y", "row")
    tiers = {}
    for s in range(40):
        text, tier = multihop_question(a, b, "EIP-1", "EIP-2", random.Random(s), b_is_unique=True)
        tiers[tier] = text
        assert text.endswith("?") and "AAA_SIZE" in text and "BBB_SIZE" in text
    assert "EIP-2" in tiers["both-named"] and "EIP-2" not in tiers["second-by-id"]
    assert multihop_question(a, b, "EIP-1", "EIP-2", random.Random(0))[1] == "both-named"


def test_stratified_take_caps_per_doc_and_round_robins_groups():
    items = [Constant(f"eip-{d}", f"eip-{d}#000", f"NAME_{d}_{i}", "5", "x", "row")
             for d in range(6) for i in range(5)]
    items += [Constant(f"cl-{d}", f"cl-{d}#000", f"CL_{d}_{i}", "5", "x", "row")
              for d in range(2) for i in range(5)]
    got = stratified_take(items, 12, per_doc=2, rng=random.Random(0), group_of=lambda k: k.doc[:2])
    assert len(got) == 12
    per = {}
    for k in got:
        per[k.doc] = per.get(k.doc, 0) + 1
    assert max(per.values()) <= 2
    assert sum(k.doc.startswith("cl") for k in got) == 4  # both groups drained fairly
    assert len(stratified_take(items, 1000, per_doc=1, rng=random.Random(0), group_of=lambda k: "g")) == 8


# ------------------------------------------------------------------------------------ purpose


def test_first_summary_sentence_and_cleaning(mini_chunks):
    by_doc, _ = split_index(mini_chunks)
    c, s = first_summary_sentence(by_doc["eip-1559"])
    assert c.section == "Abstract" and s.startswith("A base fee per gas")
    assert first_summary_sentence(by_doc["eip-2929"]) is None  # no Abstract section
    assert first_summary_sentence([]) is None
    assert clean_sentence("- See [EIP-1559](./eip-1559.md) for `X` and **Y**.") == "See EIP-1559 for X and Y"
