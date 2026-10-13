from ethrag.ingest import Chunk
from ethrag.retrieval import BM25, DenseIndex, HybridIndex, tokenize


def mk(i, text):
    return Chunk(f"eip-1#{i:03d}", "eip-1", "t", "s", text)


CH = [
    mk(0, "The BASE_FEE_MAX_CHANGE_DENOMINATOR constant controls base fee adjustment speed"),
    mk(1, "Blob transactions carry KZG commitments and versioned hashes"),
    mk(2, "Withdrawals are pushed from the beacon chain to the execution layer"),
]


def test_tokenize_splits_identifiers_but_keeps_whole():
    t = tokenize("BASE_FEE_MAX 0x48")
    assert "base_fee_max" in t and "base" in t and "0x48" in t


def test_bm25_ranks_exact_identifier_first():
    assert BM25(CH).search("BASE_FEE_MAX_CHANGE_DENOMINATOR", 1)[0][0].id == "eip-1#000"


def test_dense_ranks_related_chunk_first():
    assert DenseIndex(CH).search("KZG versioned hashes for blob transactions", 1)[0][0].id == "eip-1#001"


def test_hybrid_returns_k_distinct_results():
    res = HybridIndex(CH).search("beacon chain withdrawals", 2)
    assert len(res) == 2 and res[0][0].id == "eip-1#002"
