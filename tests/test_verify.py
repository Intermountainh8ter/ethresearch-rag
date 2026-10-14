import json

import pytest

from ethrag.ingest import Chunk
from ethrag.verify import (
    answer_novelty,
    best_source_sentence,
    bound_pairs,
    claim_tokens,
    content_words,
    has_negation,
    int_value,
    number_variants,
    question_identifiers,
    question_relevance,
    question_terms,
    range_claims,
    sentences,
    stem,
    unit_claims,
    unit_variants,
    verify_answer,
)

C0 = Chunk("eip-1559#000", "eip-1559", "t", "s", "BASE_FEE_MAX_CHANGE_DENOMINATOR = 8 and ELASTICITY_MULTIPLIER = 2")
C1 = Chunk("eip-4844#001", "eip-4844", "t", "s", "GAS_PER_BLOB is 2**17 and BLOB_TX_TYPE is 0x03")
CH = {c.id: c for c in (C0, C1)}
RET = list(CH)


def chunk(text, cid="eip-9#000"):
    return Chunk(cid, cid.split("#")[0], "T", "S", text)


def check(answer, text, **kw):
    c = chunk(text)
    return verify_answer(answer, {c.id: c}, [c.id], **kw)


# ----------------------------------------------------------------------------- claim tokens
def test_claim_tokens_numbers_idents_hex_pow():
    t = claim_tokens("GAS_PER_BLOB is 2 ** 17, type 0x03, and 4,096 elements. [eip-4844#001]")
    # hex is normalised (0x03 == 0x3) so spellings with/without leading zeros compare equal
    assert {"gas_per_blob", "2**17", "0x3", "4096"} <= t
    assert not any("eip" in x for x in t)


def test_identifiers_keep_their_underscores():
    assert claim_tokens("GAS_PER_BLOB") == {"gas_per_blob"}
    assert claim_tokens("GAS_PER_BLOB") != claim_tokens("GASPERBLOB") == {"gasperblob"}


@pytest.mark.parametrize("text, expected", [
    ("1,000,000", {"1000000"}), ("1_000_000", {"1000000"}), ("1,2,3", {"1", "2", "3"}),
    ("0x00FF", {"0xff"}), ("0x0", {"0x0"}), ("2^17", {"2**17"}), ("2 ** 17", {"2**17"}),
    ("2e18", {"2e18"}), ("3.14", {"3.14"}), ("ends with 8.", {"8"}),
])
def test_number_normalisation(text, expected):
    assert claim_tokens(text) == expected


def test_all_caps_stop_words_are_not_claims():
    assert claim_tokens("The tx MUST NOT be sent via RPC to the EVM per EIP") == set()


def test_number_variants_and_int_value():
    assert {"72", "0x48"} <= number_variants("0x48") and {"4096", "0x1000", "2**12"} <= number_variants("2**12")
    assert "2**4" in number_variants("16") and "0x10" in number_variants("16")
    assert "2000000000000000000" in number_variants("2e18")
    assert number_variants("3.14") == {"3.14"}
    assert [int_value(x) for x in ("0xff", "2**10", "7", "1e3", "3.5", "2**9999x")] == [255, 1024, 7, 1000, None, None]
    assert int_value("2**9999") is None  # absurd exponents are not expanded


def test_sentences_and_stem():
    assert len(sentences("A is 1 [eip-1#000]. B is 2 [eip-1#001].")) == 2
    assert sentences("one\ntwo") == ["one", "two"] and sentences("  ") == []
    assert stem("burns") == "burn" and stem("transactions") == "transaction" and stem("is") == "is"
    assert stem("fees") == "fees"  # words of four letters or fewer are left alone


def test_content_words_ignore_stop_words_and_citations():
    assert content_words("The base fee is burned [eip-1559#000]") == {"base", "fee", "burn"}


# --------------------------------------------------------------------------------- basic gate
def test_correct_answer_passes():
    r = verify_answer("BASE_FEE_MAX_CHANGE_DENOMINATOR is 8 [eip-1559#000].", CH, RET)
    assert r.ok and r.support_ratio == 1.0 and r.min_coverage == 1.0


def test_wrong_number_fails():
    r = verify_answer("BASE_FEE_MAX_CHANGE_DENOMINATOR is 16 [eip-1559#000].", CH, RET)
    assert not r.ok and any("'16'" in i for i in r.issues)


def test_digit_inside_larger_number_does_not_count():
    # '8' must be a token of the chunk, not merely a substring of e.g. '58'
    c = chunk("value 58 here", "eip-1#000")
    assert not verify_answer("It is 8 [eip-1#000].", {c.id: c}, [c.id]).ok


def test_uncited_sentence_fails():
    r = verify_answer("It is 8.", CH, RET)
    assert not r.ok and r.issues[0].startswith("uncited sentence") and r.coverage == [0.0]


def test_citation_must_be_retrieved():
    r = verify_answer("It is 8 [eip-1559#000].", CH, ["eip-4844#001"])
    assert not r.ok and any("not retrieved" in i for i in r.issues)
    assert not verify_answer("It is 8 [eip-1559#999].", CH, ["eip-1559#999"]).ok  # unknown chunk id


def test_claim_must_be_in_cited_not_just_any_chunk():
    r = verify_answer("GAS_PER_BLOB is 2**17 [eip-1559#000].", CH, RET)
    assert not r.ok


def test_every_sentence_is_checked_separately():
    r = verify_answer("ELASTICITY_MULTIPLIER is 2 [eip-1559#000]. GAS_PER_BLOB is 2**16 [eip-4844#001].", CH, RET)
    assert not r.ok and r.supported_tokens == 3 and r.checked_tokens == 4
    assert [(c.sentence, c.token) for c in r.claims if not c.supported] == [(1, "2**16")]


def test_citations_may_be_combined_and_duplicated():
    r = verify_answer("GAS_PER_BLOB is 2**17 and ELASTICITY_MULTIPLIER is 2 [eip-4844#001][eip-1559#000][eip-1559#000].",
                      CH, RET)
    assert r.ok and r.provenance() == {"eip-4844#001": ["2**17", "gas_per_blob"],
                                       "eip-1559#000": ["2", "elasticity_multiplier"]}


def test_equivalent_spellings_are_supported_with_their_kind():
    c = chunk("Opcode 0x48 costs 4096 gas, limit 2**20, price 1_000_000.")
    r = check("Opcode 72 costs 0x1000 gas, limit 1048576, price 1,000,000 [eip-9#000].", c.text)
    kinds = {x.token: x.kind for x in r.claims}
    assert r.ok and kinds["72"] == "hex-dec" and kinds["0x1000"] == "hex-dec" and kinds["1048576"] == "pow2"
    r = check("Limit is 2**12 [eip-9#000].", "limit is 4096")
    assert r.ok and r.claims[0].kind == "pow2"
    assert r.provenance() == {"eip-9#000": ["2**12"]}


# ------------------------------------------------------------------------- units and ranges
def test_unit_claims_and_variants():
    cl = unit_claims("Stake 32 ETH, wait 12 seconds, send 1 KiB, pay 5 gwei, cap 2**17 gas, 3.5 ETH")
    assert [(c.literal, c.unit, c.dimension) for c in cl] == [
        ("32", "eth", "value"), ("12", "seconds", "time"), ("1", "kib", "data"), ("5", "gwei", "value"),
        ("2**17", "gas", "gas")]
    assert unit_variants("32 ETH") == {"32": {"32000000000000000000", "32000000000"}}
    assert unit_variants("1 KiB") == {"1": {"1024"}} and unit_variants("no units") == {}
    assert unit_claims("5 wei")[0].base == 5 and unit_claims("2 minutes")[0].base == 120


def test_eth_is_supported_by_a_gwei_spelling_and_vice_versa():
    assert check("The deposit is 32 ETH [eip-9#000].", "MAX_EFFECTIVE_BALANCE = 32000000000 Gwei").ok
    assert check("It is 32000000000 gwei [eip-9#000].", "The deposit is 32 ETH.").ok


def test_unit_mismatch_is_reported_with_both_phrases():
    r = check("The deposit is 32 gwei [eip-9#000].", "The deposit is 32 ETH.")
    assert not r.ok
    assert any(i == "unit mismatch: answer says '32 gwei' but the cited chunk says '32 ETH'" for i in r.issues)


def test_same_quantity_in_another_unit_is_not_a_mismatch():
    assert check("The window is 1024 bytes [eip-9#000].", "The window is 1 KiB.").ok
    assert check("It takes 120 seconds [eip-9#000].", "It takes 2 minutes.").ok


def test_unit_check_stays_silent_when_the_chunk_gives_the_number_without_a_unit():
    assert check("The value is 32 ETH [eip-9#000].", "MAX_BALANCE | 32 |").ok  # no unit to contradict
    assert check("The value is 32 gwei [eip-9#000].", "The deposit is 32 ETH.", check_units=False).ok


def test_range_claims_forms():
    assert range_claims("between 1 and 16") == [("1", "16")]
    assert range_claims("0 to 0x7f, 1-16, 5–9, 3..4") == [("0", "0x7f"), ("1", "16"), ("5", "9"), ("3", "4")]
    assert range_claims("[1,000-2,000]") == [("1000", "2000")]


@pytest.mark.parametrize("text", [
    "nonce is 2^64-1", "costs 4/16 to 10/40 gas", "created 2020-09-01", "EIP-1559 and ERC-20", "item 3-4x", "x**2-1",
])
def test_arithmetic_dates_and_fractions_are_not_ranges(text):
    assert range_claims(text) == []


def test_range_mismatch_and_matching_ranges():
    src = "The type is a number between 0 and 0x7f."
    assert check("Types run from 0 to 127 [eip-9#000].", src).ok  # same numbers, different spelling
    r = check("Types run from 1 to 0x7f [eip-9#000].", src)
    assert not r.ok and any(i.startswith("range mismatch: answer says 1 to 0x7f but the cited chunk says 0 to 0x7f")
                            for i in r.issues)
    r = check("Types run from 0 to 0x80 [eip-9#000].", src + " 0x80 is reserved.")
    assert not r.ok and any("range mismatch" in i for i in r.issues)


def test_range_without_a_range_in_the_chunk_falls_back_to_tokens_and_backwards_ranges_fail():
    assert check("It is 1-16 [eip-9#000].", "MIN is 1, MAX is 16").ok
    r = check("It is 16-1 [eip-9#000].", "MIN is 1, MAX is 16")
    assert not r.ok and any(i.startswith("malformed range") for i in r.issues)


# ------------------------------------------------------------------------------------ binding
def test_bound_pairs_attribute_numbers_to_the_identifier_just_before():
    assert bound_pairs("BLOB_TX_TYPE is 0x03") == {"0x3": {"blob_tx_type"}}
    assert bound_pairs("Access list transactions are type 0x01 and BLOB_TX_TYPE is 0x03") == {"0x3": {"blob_tx_type"}}
    assert bound_pairs("A_B is 9 and C_D is 9") == {"9": {"a_b", "c_d"}}
    assert bound_pairs("TARGET_X is 6 while EIP-4844 set it") == {"6": {"target_x"}}  # references are not values
    assert bound_pairs("no identifiers 1 2 3") == {}
    assert bound_pairs("the MUST value is 5") == {}  # stop-word caps are not identifiers


TABLE = "| `NAME_A` | 8 |\n| `NAME_B` | 16 |\n| `NAME_C` | 32 |"


def test_value_from_a_neighbouring_table_row_does_not_bind():
    assert check("NAME_A is 8 [eip-9#000].", TABLE).ok
    r = check("NAME_A is 16 [eip-9#000].", TABLE)
    assert not r.ok and any(i.startswith("unbound claim '16'") for i in r.issues)
    assert r.claims[-1].bound is False or any(c.bound is False for c in r.claims)
    assert check("NAME_A is 16 [eip-9#000].", TABLE, check_binding=False).ok  # token check alone is fooled


def test_every_identifier_sharing_a_value_must_be_bound():
    text = "| `MAX_X` | 9 |\n| `TARGET_X` | 6 |"
    assert check("MAX_X is 9 and TARGET_X is 6 [eip-9#000].", text).ok
    assert not check("MAX_X is 9 and TARGET_X is 9 [eip-9#000].", text).ok


def test_a_continuation_line_borrows_the_identifier_of_its_neighbour_but_not_across_rows():
    text = "The value of `STATIC_FLAG`\n\nOpcode: `0xfa`.\n\nOther text."
    assert check("STATIC_FLAG is 0xfa [eip-9#000].", text).ok
    c2 = "STATIC_FLAG = 0x01\nOpcode: 0xfa\nOTHER_FLAG = 0x02"
    assert check("STATIC_FLAG is 0xfa [eip-9#000].", c2).ok
    c3 = "STATIC_FLAG = 0x01\nOTHER_FLAG = 0xfa"
    assert not check("STATIC_FLAG is 0xfa [eip-9#000].", c3).ok


def test_sentences_mixing_subjects_only_bind_the_numbers_that_follow_an_identifier():
    text = "Access list transactions use type 0x01.\nBLOB_TX_TYPE = 0x03"
    assert check("Access list transactions are type 0x01 and BLOB_TX_TYPE is 0x03 [eip-9#000].", text).ok


def test_provenance_snippets_and_explain():
    r = check("NAME_B is 16 [eip-9#000].", TABLE)
    assert r.ok
    by_tok = {c.token: c for c in r.claims}
    assert by_tok["16"].snippet == "| `NAME_B` | 16 |" and by_tok["16"].bound is True
    assert by_tok["name_b"].bound is None and by_tok["16"].chunk == "eip-9#000"
    text = r.explain()
    assert text.startswith("verdict: VERIFIED") and "| `NAME_B` | 16 |" in text and "eip-9#000 (exact)" in text
    bad = check("NAME_B is 99 [eip-9#000].", TABLE)
    assert "verdict: WITHHELD" in bad.explain() and "[FAIL]" in bad.explain() and "issue: unsupported claim" in bad.explain()
    assert "UNBOUND" in check("NAME_A is 16 [eip-9#000].", TABLE).explain()


def test_to_dict_is_json_serialisable_and_complete():
    d = check("NAME_B is 16 [eip-9#000].", TABLE, question="What is NAME_B?", min_relevance=0.1).to_dict()
    json.dumps(d)
    assert d["ok"] and d["support_ratio"] == 1.0 and d["claims"][0]["token"] and d["relevance"] == 1.0


# -------------------------------------------------------------------------------- negation
def test_negation_words():
    assert all(has_negation(t) for t in ["does not burn", "never", "can't", "cannot", "no longer", "without gas",
                                          "nor", "is disallowed", "forbidden", "unable"])
    assert not has_negation("The fee is burned [eip-1559#000]") and not has_negation("notable nothing")


def test_negation_in_answer_without_negated_source_is_rejected():
    src = "The base fee is burned and never paid to miners."
    assert check("The base fee is burned [eip-9#000].", src).ok
    r = check("The base fee is not burned [eip-9#000].", src.replace("never paid to miners", "paid to nobody"))
    assert not r.ok and any(i.startswith("negation in answer sentence 0 has no negated source") for i in r.issues)
    assert check("The base fee is not burned [eip-9#000].", "The base fee is not burned in this EIP.").ok
    assert check("The base fee is not burned [eip-9#000].", "The base fee is burned.", check_negation=False).ok


def test_dropped_negation_warns_by_default_and_fails_when_strict():
    src = "Contracts MUST NOT deploy code starting with the byte."
    r = check("Contracts MUST deploy code starting with the byte [eip-9#000].", src)
    assert r.ok and any(w.startswith("source sentence for 0 is negated") for w in r.warnings)
    r = check("Contracts MUST deploy code starting with the byte [eip-9#000].", src, strict_negation=True)
    assert not r.ok and any(i.startswith("negation dropped in answer sentence 0") for i in r.issues)


def test_negated_claim_about_a_code_line_is_caught():
    text = "```python\nFEE_MARKET = 0x02\nACCESS_LIST = 0x01\n```"
    r = check("FEE_MARKET is not 0x02 [eip-9#000].", text)
    assert not r.ok and any("negation in answer sentence 0" in i for i in r.issues)
    assert best_source_sentence("FEE_MARKET is not 0x02", [text])[0] == "FEE_MARKET = 0x02"
    assert best_source_sentence("zzz qqq", [text]) == ("", 0.0) and best_source_sentence("", [text]) == ("", 0.0)


def test_table_row_is_a_negation_source_when_no_prose_matches():
    r = check("FEE_MARKET is not 0x02 [eip-9#000].", "| `FEE_MARKET` | 0x02 |")
    assert not r.ok


# ------------------------------------------------------------------------------------ coverage
def test_content_coverage_rejects_token_free_paraphrase():
    src = "The elasticity multiplier lets blocks be twice the target size."
    vague = "The specification sets this parameter as described above [eip-9#000]."
    assert check(vague, src).ok  # no checkable token and no coverage requirement
    r = check(vague, src, min_coverage=0.5)
    assert not r.ok and any(i.startswith("sentence 0 coverage 0.") for i in r.issues)
    assert check("The elasticity multiplier lets blocks be twice the target size [eip-9#000].", src,
                 min_coverage=0.9).ok
    assert r.coverage and r.min_coverage < 0.5


# ----------------------------------------------------------------------------------- relevance
def test_question_terms_drop_references_and_meta_words():
    assert question_terms("What does ERC-3009 propose?") == set()
    assert question_terms("What is the value of EIP-4844 BLOB_TX_TYPE?") >= {"blob_tx_type", "blob"}
    assert "4844" not in question_terms("What is the value of EIP-4844 BLOB_TX_TYPE?")
    assert question_identifiers("What are TLOAD and BASE_FEE_X in EIP-1153?") == {"tload", "base_fee_x"}


def test_question_relevance_scores_topical_overlap():
    q = "How is the base fee burned?"
    assert question_relevance(q, "The base fee per gas is burned [x#000].") == 1.0
    assert question_relevance(q, "Withdrawals are pushed from the beacon chain") == 0.0
    assert question_relevance("What does ERC-3009 propose?", "anything at all") == 1.0


def test_off_topic_answer_with_valid_citation_is_rejected_by_relevance():
    text = "Withdrawals are pushed from the beacon chain to the execution layer."
    q = "How is the base fee burned?"
    ans = "Withdrawals are pushed from the beacon chain to the execution layer [eip-9#000]."
    assert check(ans, text, question=q).ok  # relevance is off by default
    r = check(ans, text, question=q, min_relevance=0.25)
    assert not r.ok and r.relevance == 0.0 and any(i.startswith("answer relevance 0.00 < 0.25") for i in r.issues)


def test_eip_numbers_in_the_question_do_not_count_as_topic():
    text = "Using the pseudocode language of EIP-1559, the updated base fee calculation becomes: below."
    q = "What is the BASE_FEE_MAX_CHANGE_DENOMINATOR in EIP-1559?"
    r = check("Using the pseudocode language of EIP-1559, the updated base fee calculation becomes [eip-9#000].",
              text, question=q, min_relevance=0.1)
    assert not r.ok and any("does not mention ['base_fee_max_change_denominator']" in i for i in r.issues)


def test_question_naming_a_parameter_needs_a_value_not_a_restatement():
    text = "| `NAME_B` | 16 |"
    q = "What is NAME_B in EIP-9?"
    r = check("NAME_B in EIP-9 is as specified [eip-9#000].", text, question=q, min_relevance=0.1)
    assert not r.ok and any("states no value for ['name_b']" in i for i in r.issues)
    assert check("NAME_B is 16 [eip-9#000].", text, question=q, min_relevance=0.1).ok


def test_naming_a_document_the_answer_does_not_cite():
    text = "Blob transactions carry commitments and versioned hashes."
    ans = "Blob transactions carry commitments and versioned hashes [eip-9#000]."
    # topical words present: only a warning
    r = check(ans, text, question="What do blob transactions carry in EIP-4844?", min_relevance=0.1)
    assert r.ok and any("the question names ['eip-4844', 'erc-4844']" in w for w in r.warnings)
    # nothing but the reference says what the question is about: citing another document fails
    r = check(ans, text, question="What does EIP-4844 propose?", min_relevance=0.1)
    assert not r.ok and any(i.startswith("the question names") for i in r.issues)
    c = Chunk("eip-4844#000", "eip-4844", "T", "S", text)
    assert verify_answer("Blob transactions carry commitments and versioned hashes [eip-4844#000].", {c.id: c},
                         [c.id], question="What does EIP-4844 propose?", min_relevance=0.1).ok


def test_novelty_flags_answers_that_only_restate_the_question():
    q = "What changes did the committee make to blob handling in the second release?"
    assert answer_novelty(q, "NAME_B is 16") > 0.9 and answer_novelty("", "") == 1.0
    echo = "What changes did the committee make to blob handling in the second release, as stated"
    assert answer_novelty(q, echo) < 0.25
    text = "committee blob handling second release changes stated make did"
    r = check(echo + " [eip-9#000].", text, question=q, min_novelty=0.25)
    assert not r.ok and any(i.startswith("answer novelty") for i in r.issues)
    assert check(echo + " [eip-9#000].", text, question=q).ok


# ----------------------------------------------------------------------------- blind spots
def test_blind_spot_0xef_vs_0xfe():
    """KNOWN LIMITATION (docs/EVALUATION.md): presence is not binding.

    EIP-3541 forbids new code starting with 0xEF; the same chunk legitimately mentions 0xFE (the
    INVALID opcode). An answer that swaps the two is "grounded": every token is in the cited
    chunk, no identifier ties the number to a subject, the words overlap. The verifier accepts it.
    If this test ever fails, the verifier got stronger - update the docs, do not just delete it.
    """
    text = ("New contract code starting with the 0xEF byte MUST be rejected. "
            "The 0xFE byte is the designated INVALID opcode and remains allowed.")
    wrong = "New contract code starting with the 0xFE byte is rejected [eip-9#000]."
    r = check(wrong, text, min_coverage=0.7)
    assert r.ok, r.issues  # <- the blind spot
    assert not check("New contract code starting with the 0xEE byte is rejected [eip-9#000].", text).ok


def test_blind_spot_binding_stops_at_identifiers_so_naming_the_subject_helps():
    text = "BAD_PREFIX = 0xEF\nINVALID_OPCODE = 0xFE"
    assert check("BAD_PREFIX is 0xEF [eip-9#000].", text).ok
    r = check("BAD_PREFIX is 0xFE [eip-9#000].", text)
    assert not r.ok and any(i.startswith("unbound claim '0xfe'") for i in r.issues)


def test_blind_spot_two_values_on_one_line_cannot_be_told_apart():
    text = "BASE_FEE = 8, ELASTICITY = 2"  # same line: binding cannot separate them
    assert check("BASE_FEE is 2 [eip-9#000].", text).ok


def test_blind_spot_grounded_is_not_the_same_as_answering():
    text = "The base fee is burned. Priority fees are paid to the miner. The block gas limit is 30000000."
    q = "How large is the block gas limit?"
    r = check("The base fee is burned [eip-9#000].", text, question=q, min_coverage=0.7, min_relevance=0.1)
    assert not r.ok  # caught here only because it shares no topical word with the question
    related = "The block gas limit is burned [eip-9#000]."
    assert check(related, text.replace("30000000", "burned"), question=q, min_coverage=0.7, min_relevance=0.1).ok
