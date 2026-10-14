import json

import numpy as np
import pytest

from ethrag.retrieval import HashedEmbedder, HybridIndex
from ethrag.store import INDEX_VERSION, is_index_dir, load_index, save_index

QUERIES = ["BASE_FEE_MAX_CHANGE_DENOMINATOR", "blob transactions KZG", "slots per epoch",
           "EIP-2930 access list cost", "totally unrelated words zzz"]


def test_round_trip_preserves_every_ranking_and_score(mini_index, tmp_path):
    save_index(mini_index, tmp_path / "idx")
    assert is_index_dir(tmp_path / "idx")
    loaded = load_index(tmp_path / "idx")
    assert [c.id for c in loaded.chunks] == [c.id for c in mini_index.chunks]
    assert loaded.chunks == mini_index.chunks  # frozen dataclasses compare by value
    for q in QUERIES:
        a, b = mini_index.search(q, 5), loaded.search(q, 5)
        assert [c.id for c, _ in a] == [c.id for c, _ in b]
        assert np.allclose([s for _, s in a], [s for _, s in b])
        assert np.allclose(mini_index.bm25.scores(q), loaded.bm25.scores(q))
        assert np.allclose(mini_index.dense.scores(q), loaded.dense.scores(q), atol=1e-6)


def test_round_trip_preserves_params_and_graph(mini_chunks, tmp_path):
    idx = HybridIndex(mini_chunks, rrf_k=30, bm25_weight=0.7, eip_boost=2.5, eip_mode="restrict", top=77)
    save_index(idx, tmp_path)
    loaded = load_index(tmp_path)
    assert (loaded.rrf_k, loaded.bm25_weight, loaded.eip_boost, loaded.eip_mode, loaded.top) == (
        30, 0.7, 2.5, "restrict", 77)
    assert loaded.graph.requires == idx.graph.requires
    assert loaded.bm25.k1 == idx.bm25.k1 and loaded.bm25.title_boost == idx.bm25.title_boost


def test_missing_or_partial_directory_raises(tmp_path):
    assert not is_index_dir(tmp_path)
    with pytest.raises(FileNotFoundError, match="not an ethrag index"):
        load_index(tmp_path)
    (tmp_path / "meta.json").write_text("{}")
    assert not is_index_dir(tmp_path)


def test_version_mismatch_is_rejected(mini_index, tmp_path):
    save_index(mini_index, tmp_path)
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["version"] == INDEX_VERSION
    meta["version"] = INDEX_VERSION + 1
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="rebuild"):
        load_index(tmp_path)


class TinyEmbedder:
    """Deterministic dense embedder (no sparse path) to exercise the dense-matrix persistence."""

    def __init__(self):
        self.h = HashedEmbedder(dim=512)

    def embed(self, texts):
        return self.h.embed(texts)


def test_custom_embedder_round_trip_needs_the_same_embedder(mini_chunks, tmp_path):
    emb = TinyEmbedder()
    idx = HybridIndex(mini_chunks, embedder=emb)
    save_index(idx, tmp_path)
    with pytest.raises(ValueError, match="custom embedder"):
        load_index(tmp_path)
    loaded = load_index(tmp_path, embedder=emb)
    q = "blob transaction type byte"
    assert [c.id for c, _ in idx.search(q, 3)] == [c.id for c, _ in loaded.search(q, 3)]


def test_saved_files_are_plain_npz_and_json(mini_index, tmp_path):
    save_index(mini_index, tmp_path)
    assert {p.name for p in tmp_path.iterdir()} == {"arrays.npz", "chunks.json", "meta.json"}
    with np.load(tmp_path / "arrays.npz", allow_pickle=False) as z:
        assert "bm25_idx" in z.files and "dense_vals" in z.files
