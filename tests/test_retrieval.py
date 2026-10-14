import numpy as np
import pytest

from ethrag.ingest import Chunk
from ethrag.retrieval import (
    BM25,
    DenseIndex,
    HashedEmbedder,
    HybridIndex,
    QueryPlan,
    content_tokens,
    log_len,
    normalize_scores,
    plan_query,
    rrf_vector,
    tokenize,
)


def mk(i, text, doc="eip-1", title="t", section="s"):
    return Chunk(f"{doc}#{i:03d}", doc, title, section, text)


CH = [
    mk(0, "The BASE_FEE_MAX_CHANGE_DENOMINATOR constant controls base fee adjustment speed"),
    mk(1, "Blob transactions carry KZG commitments and versioned hashes"),
    mk(2, "Withdrawals are pushed from the beacon chain to the execution layer"),
]


def test_tokenize_splits_identifiers_but_keeps_whole():
    t = tokenize("BASE_FEE_MAX 0x48")
    assert "base_fee_max" in t and "base" in t and "0x48" in t
    assert tokenize("version 1.2 and x") == ["version", "1.2", "and", "x"]


def test_content_tokens_drop_stopwords_and_single_letters():
    assert content_tokens("What is the base fee of a block?") == ["base", "fee", "block"]


def test_bm25_ranks_exact_identifier_first():
    assert BM25(CH).search("BASE_FEE_MAX_CHANGE_DENOMINATOR", 1)[0][0].id == "eip-1#000"


def test_bm25_unknown_terms_and_k_bounds():
    b = BM25(CH)
    assert not b.scores("zzzzqqqq").any()
    assert b.search("beacon", 0) == []
    assert len(b.search("beacon", 99)) == 3


def test_bm25_title_field_boost_lifts_matching_document():
    chunks = [mk(0, "gas gas gas", doc="eip-7", title="Other"),
              mk(0, "gas", doc="eip-8", title="Gas repricing")]
    assert BM25(chunks, title_boost=5.0).search("repricing", 1)[0][0].doc == "eip-8"


def test_bm25_empty_corpus():
    b = BM25([])
    assert b.scores("x").shape == (0,) and b.search("x") == []


def test_bm25_arrays_round_trip():
    b = BM25(CH)
    arrays, vocab = b.to_arrays()
    b2 = BM25(CH, arrays=arrays, vocab=vocab)
    assert np.allclose(b.scores("blob kzg"), b2.scores("blob kzg"))


def test_dense_ranks_related_chunk_first():
    assert DenseIndex(CH).search("KZG versioned hashes for blob transactions", 1)[0][0].id == "eip-1#001"


def test_dense_empty_query_and_oov_query_score_zero():
    d = DenseIndex(CH)
    assert not d.scores("").any() and not d.scores("!!!").any()


def test_hashed_embedder_sparse_and_dense_views_agree():
    e = HashedEmbedder(dim=1024)
    texts = ["blob transactions", "beacon chain"]
    indptr, indices, counts = e.embed_sparse(texts)
    m = e.embed(texts)
    for r in range(2):
        row = np.zeros(1024, dtype=np.float32)
        row[indices[indptr[r]:indptr[r + 1]]] = counts[indptr[r]:indptr[r + 1]]
        assert np.array_equal(row, m[r])
    ids, cnt = e.features("")
    assert ids.size == 0 and cnt.size == 0
    assert e.embed_sparse([])[1].size == 0


class DenseOnly:
    def embed(self, texts):
        return HashedEmbedder(dim=256).embed(texts)


def test_dense_index_with_non_sparse_embedder():
    d = DenseIndex(CH, embedder=DenseOnly())
    assert not d.sparse and d.search("KZG blob", 1)[0][0].id == "eip-1#001"
    assert "dense_m" in d.to_arrays()
    d2 = DenseIndex(CH, embedder=DenseOnly(), arrays=d.to_arrays())
    assert np.allclose(d.scores("blob"), d2.scores("blob"))


def test_plan_query_resolves_named_documents():
    plan = plan_query("Compare EIP-1559 with eip 4844 and EIP-1559 again", {1559: "eip-1559", 4844: "eip-4844"})
    assert isinstance(plan, QueryPlan)
    assert plan.numbers == [1559, 4844] and plan.docs == ["eip-1559", "eip-4844"]
    assert plan_query("EIP-9", None).docs == [] and plan_query("EIP-9", {}).docs == []


def test_rrf_vector_credits_only_top_positive_scores():
    v = rrf_vector(np.array([0.0, 3.0, 1.0, 2.0], dtype=np.float32), rrf_k=60, top=2)
    assert v[1] == pytest.approx(1 / 61) and v[3] == pytest.approx(1 / 62)
    assert v[0] == 0 and v[2] == 0
    assert not rrf_vector(np.zeros(4, dtype=np.float32), 60, 3).any()
    assert rrf_vector(np.array([], dtype=np.float32), 60, 3).size == 0
    assert rrf_vector(np.array([1.0, 2.0], dtype=np.float32), 60, 0).sum() == 0


def test_normalize_scores_and_log_len():
    assert np.allclose(normalize_scores(np.array([1.0, 3.0, 2.0])), [0, 1, 0.5])
    assert not normalize_scores(np.array([2.0, 2.0])).any() and normalize_scores(np.array([])).size == 0
    assert log_len(mk(0, "one two three")) > 0


def test_hybrid_returns_k_distinct_results():
    res = HybridIndex(CH).search("beacon chain withdrawals", 2)
    assert len(res) == 2 and res[0][0].id == "eip-1#002"


def test_hybrid_rejects_unknown_eip_mode():
    with pytest.raises(ValueError):
        HybridIndex(CH, eip_mode="bogus")


def test_named_eip_is_boosted_or_restricted(mini_index):
    q = "What is the cost of an access list in EIP-2930?"
    boosted = mini_index.search(q, 3, eip_mode="boost")
    restricted = mini_index.search(q, 3, eip_mode="restrict")
    assert boosted[0][0].doc == "eip-2930"
    assert {c.doc for c, _ in restricted} == {"eip-2930"}
    off = mini_index.fused_scores(q, eip_mode="off")
    on = mini_index.fused_scores(q, eip_mode="boost", eip_boost=2.0)
    idx = mini_index.doc_chunks["eip-2930"]
    assert np.all(on[idx] >= off[idx]) and (on[idx] > off[idx]).any()


def test_fuse_ignores_unknown_named_docs(mini_index):
    rb, rd = mini_index.rank_vectors("blob")
    assert np.allclose(mini_index.fuse(rb, rd, ["eip-0000"]), mini_index.fuse(rb, rd, []))


def test_bm25_weight_extremes_follow_single_ranker(mini_index):
    q = "kzg versioned hashes"
    only_b = mini_index.search(q, 1, bm25_weight=1.0)[0][0].id
    only_d = mini_index.search(q, 1, bm25_weight=0.0)[0][0].id
    assert only_b == mini_index.bm25.search(q, 1)[0][0].id
    assert only_d == mini_index.dense.search(q, 1)[0][0].id


def test_graph_expansion_adds_required_documents(mini_index):
    base = mini_index.search("blob transaction GAS_PER_BLOB", 1)
    assert base[0][0].doc == "eip-4844"
    expanded = mini_index.search("blob transaction GAS_PER_BLOB", 1, expand_graph=True, extra=2)
    assert len(expanded) == 3 and expanded[0] == base[0]
    docs = {c.doc for c, _ in expanded}
    assert "eip-4844" in docs and docs - {"eip-4844"} <= {"eip-1559", "eip-2718", "eip-2930"}
    assert len({c.id for c, _ in expanded}) == 3


def test_graph_expansion_depth_two_reaches_second_hop(mini_index):
    res = mini_index.search("blob transaction GAS_PER_BLOB", 1, expand_graph=True, extra=10, depth=2)
    assert "eip-2929" in {c.doc for c, _ in res}  # eip-4844 -> eip-2930 -> eip-2929


def test_search_empty_corpus_and_nothing_to_expand():
    h = HybridIndex([])
    assert h.search("x", 3) == []
    assert h.search("x", 3, expand_graph=True) == []


def test_shared_neighbour_is_expanded_once(mini_index):
    # eip-2718 is required by both eip-1559 and eip-4844; it must be added a single time.
    res = mini_index.search("EIP-1559 EIP-4844 blob base fee", 2, expand_graph=True, extra=5)
    docs = [c.doc for c, _ in res]
    assert docs.count("eip-2718") <= 1 and len({c.id for c, _ in res}) == len(res)


def test_bm25_drops_stop_words_from_the_query_only():
    chunks = [mk(0, "the of is a an and"), mk(1, "blob transactions carry kzg commitments")]
    on, off = BM25(chunks), BM25(chunks, drop_stopwords=False)
    assert on.query_terms("What is the blob?") == ["blob"]
    assert off.query_terms("What is the blob?") == ["what", "is", "the", "blob"]
    assert on.query_terms("what is the") == ["what", "is", "the"]  # all-stop-word queries still score
    assert on.search("what is the blob", 1)[0][0].id == "eip-1#001"
    assert off.search("what is the blob", 1)[0][0].id == "eip-1#000"  # noise wins when stop words count
    assert "the" in on.vocab  # the index itself still contains them
