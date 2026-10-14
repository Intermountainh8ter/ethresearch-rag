import numpy as np
import pytest

from ethrag.rerank import (
    FEATURE_NAMES,
    LogisticReranker,
    chunk_features,
    query_signals,
    section_type,
    status_class,
    training_pairs,
)


def test_section_type_and_status_class():
    assert section_type("Specification") == "spec" and section_type("Abstract") == "abstract"
    assert section_type("Summary") == "abstract" and section_type("Security Considerations") == "security"
    assert section_type("Motivation") == "motivation" and section_type("Rationale") == "rationale"
    assert section_type("Misc") == "other"
    assert status_class("Final") == "final" and status_class("Last Call") == "final"
    assert status_class("Draft") == "draft" and status_class("Withdrawn") == "stagnant"
    assert status_class("") == "stagnant"


def test_query_signals_extracts_identifiers_and_numbers():
    content, idents, nums = query_signals("What is BASE_FEE_MAX in EIP-1559 for 0x48?")
    assert "base_fee_max" in idents and {"1559", "0x48"} <= nums
    assert "base_fee_max" in content and "what" not in content


def _toy(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(FEATURE_NAMES)))
    y = (X[:, 0] * 2.0 - X[:, 3] + 0.1 * rng.normal(size=n) > 0.5).astype(float)
    return X, y


def test_fit_learns_a_separable_signal_and_loss_decreases():
    X, y = _toy()
    r = LogisticReranker().fit(X, y)
    p = r.predict(X)
    assert p.shape == (len(y),) and np.all((0 < p) & (p < 1))
    assert r.history[-1] < r.history[0]
    pos, neg = p[y == 1].mean(), p[y == 0].mean()
    assert pos > neg + 0.3
    w = r.feature_weights()
    assert w["bm25"] > 0.5 and w["lex_overlap"] < -0.2  # recovers the generating signs
    assert set(w) == set(FEATURE_NAMES)


def test_fit_without_balancing_and_shape_validation():
    X, y = _toy(100)
    LogisticReranker().fit(X, y, balance=False, epochs=20)
    with pytest.raises(ValueError):
        LogisticReranker().fit(X[:, :3], y)
    with pytest.raises(ValueError):
        LogisticReranker().fit(X, y[:-1])


def test_dict_and_file_round_trip(tmp_path):
    X, y = _toy(120)
    r = LogisticReranker().fit(X, y, epochs=50)
    path = tmp_path / "w.json"
    r.save(path)
    r2 = LogisticReranker.load(path)
    assert np.allclose(r.predict(X), r2.predict(X))
    d = r.to_dict()
    d["features"] = d["features"][:-1]
    with pytest.raises(ValueError, match="different feature set"):
        LogisticReranker.from_dict(d)


def test_chunk_features_shape_and_flags(mini_index):
    q = "What is BASE_FEE_MAX_CHANGE_DENOMINATOR in EIP-1559?"
    bm25, dense = mini_index.bm25.scores(q), mini_index.dense.scores(q)
    fused = mini_index.fused_scores(q)
    cand = list(range(len(mini_index.chunks)))
    X = chunk_features(mini_index, q, cand, bm25, dense, fused)
    assert X.shape == (len(cand), len(FEATURE_NAMES))
    col = {n: i for i, n in enumerate(FEATURE_NAMES)}
    named = [i for i, c in enumerate(mini_index.chunks) if c.doc == "eip-1559"]
    assert all(X[i, col["eip_named"]] == 1.0 for i in named)
    assert X[:, col["eip_named"]].sum() == len(named)
    code = [i for i, c in enumerate(mini_index.chunks) if c.kind == "code"]
    assert all(X[i, col["kind_code"]] == 1.0 for i in code)
    ident_hit = [i for i in cand if X[i, col["ident_match"]] == 1.0]
    assert ident_hit and all("BASE_FEE_MAX_CHANGE_DENOMINATOR" in mini_index.chunks[i].text for i in ident_hit)


def _gold(mini_chunks):
    gid = next(c.id for c in mini_chunks if "BASE_FEE_MAX_CHANGE_DENOMINATOR = 8" in c.text)
    return [{"id": "q1", "q": "What is BASE_FEE_MAX_CHANGE_DENOMINATOR?"},
            {"id": "q2", "q": "Which ELASTICITY_MULTIPLIER does the base fee use?"}], \
        {"q1": {gid}, "q2": {gid}}, gid


def test_training_pairs_labels_gold_chunks(mini_index, mini_chunks):
    qs, gold, _ = _gold(mini_chunks)
    X, y = training_pairs(mini_index, qs, gold, pool=10)
    assert X.shape[1] == len(FEATURE_NAMES) and len(X) == len(y) and y.sum() == 2
    X0, y0 = training_pairs(mini_index, [{"id": "z", "q": "qqqqqq"}], {"z": set()})
    assert X0.shape == (0, len(FEATURE_NAMES)) and y0.size == 0


def test_rerank_returns_sorted_top_k_and_can_promote_gold(mini_index, mini_chunks):
    qs, gold, gid = _gold(mini_chunks)
    X, y = training_pairs(mini_index, qs, gold, pool=10)
    r = LogisticReranker().fit(X, y, epochs=300)
    out = r.rerank(mini_index, qs[0]["q"], k=3)
    scores = [s for _, s in out]
    assert len(out) == 3 and scores == sorted(scores, reverse=True)
    assert out[0][0].id == gid
    assert LogisticReranker().rerank(mini_index, "zzzzqqq", k=3) is not None


def test_rerank_on_empty_index_returns_nothing():
    from ethrag.retrieval import HybridIndex

    assert LogisticReranker().rerank(HybridIndex([]), "anything", k=3) == []
