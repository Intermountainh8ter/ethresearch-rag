from ethrag.ingest import Chunk
from ethrag.verify import claim_tokens, sentences, verify_answer

C0 = Chunk("eip-1559#000", "eip-1559", "t", "s", "BASE_FEE_MAX_CHANGE_DENOMINATOR = 8 and ELASTICITY_MULTIPLIER = 2")
C1 = Chunk("eip-4844#001", "eip-4844", "t", "s", "GAS_PER_BLOB is 2**17 and BLOB_TX_TYPE is 0x03")
CH = {c.id: c for c in (C0, C1)}
RET = list(CH)


def test_claim_tokens_numbers_idents_hex_pow():
    t = claim_tokens("GAS_PER_BLOB is 2 ** 17, type 0x03, and 4,096 elements. [eip-4844#001]")
    assert {"gas_per_blob", "2**17", "0x03", "4096"} <= t
    assert not any("eip" in x for x in t)


def test_correct_answer_passes():
    r = verify_answer("BASE_FEE_MAX_CHANGE_DENOMINATOR is 8 [eip-1559#000].", CH, RET)
    assert r.ok and r.support_ratio == 1.0


def test_wrong_number_fails():
    r = verify_answer("BASE_FEE_MAX_CHANGE_DENOMINATOR is 16 [eip-1559#000].", CH, RET)
    assert not r.ok and any("'16'" in i for i in r.issues)


def test_digit_inside_larger_number_does_not_count():
    # '8' must be a token of the chunk, not merely a substring of e.g. '58'
    c = Chunk("eip-1#000", "eip-1", "t", "s", "value 58 here")
    assert not verify_answer("It is 8 [eip-1#000].", {c.id: c}, [c.id]).ok


def test_uncited_sentence_fails():
    assert not verify_answer("It is 8.", CH, RET).ok


def test_citation_must_be_retrieved():
    r = verify_answer("It is 8 [eip-1559#000].", CH, ["eip-4844#001"])
    assert not r.ok and any("not retrieved" in i for i in r.issues)


def test_claim_must_be_in_cited_not_just_any_chunk():
    r = verify_answer("GAS_PER_BLOB is 2**17 [eip-1559#000].", CH, RET)
    assert not r.ok


def test_sentence_split_keeps_trailing_citation_attached():
    assert len(sentences("A is 1 [eip-1#000]. B is 2 [eip-1#001].")) == 2
