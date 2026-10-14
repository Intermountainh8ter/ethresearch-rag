"""Index persistence: numpy arrays in one ``.npz`` plus JSON for chunks, vocabulary and params.

``ethrag build-index`` writes a directory; ``ethrag ask --index DIR`` loads it without touching
the markdown sources or recomputing anything. Only the ``HashedEmbedder`` dense index is
persisted as sparse arrays; a neural embedder's matrix is stored dense and needs the same
embedder at load time to embed queries.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .graph import CitationGraph
from .ingest import Chunk
from .retrieval import BM25, DenseIndex, Embedder, HashedEmbedder, HybridIndex

INDEX_VERSION = 2
_FILES = ("arrays.npz", "chunks.json", "meta.json")


def chunk_to_dict(c: Chunk) -> dict:
    d = c.__dict__.copy()
    d["requires"] = list(c.requires)
    return d


def chunk_from_dict(d: dict) -> Chunk:
    d = dict(d)
    d["requires"] = tuple(d.get("requires", ()))
    return Chunk(**d)


def save_index(index: HybridIndex, path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    bm25_arrays, vocab = index.bm25.to_arrays()
    arrays = {**bm25_arrays, **index.dense.to_arrays()}
    np.savez_compressed(path / "arrays.npz", **arrays)
    (path / "chunks.json").write_text(
        json.dumps([chunk_to_dict(c) for c in index.chunks], separators=(",", ":")), encoding="utf-8")
    emb = index.dense.embedder
    meta = {
        "version": INDEX_VERSION,
        "chunks": len(index.chunks),
        "docs": len({c.doc for c in index.chunks}),
        "bm25": {"k1": index.bm25.k1, "b": index.bm25.b, "title_boost": index.bm25.title_boost,
                 "section_boost": index.bm25.section_boost},
        "hybrid": {"rrf_k": index.rrf_k, "bm25_weight": index.bm25_weight,
                   "eip_boost": index.eip_boost, "eip_mode": index.eip_mode, "top": index.top},
        "embedder": {"kind": "hashed", "dim": emb.dim} if isinstance(emb, HashedEmbedder)
        else {"kind": type(emb).__name__},
        "vocab": vocab,
    }
    (path / "meta.json").write_text(json.dumps(meta, separators=(",", ":")), encoding="utf-8")
    return path


def is_index_dir(path: str | Path) -> bool:
    p = Path(path)
    return all((p / f).is_file() for f in _FILES)


def load_index(path: str | Path, embedder: Embedder | None = None) -> HybridIndex:
    path = Path(path)
    if not is_index_dir(path):
        raise FileNotFoundError(f"{path} is not an ethrag index directory (expected {_FILES})")
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    if meta.get("version") != INDEX_VERSION:
        raise ValueError(f"index version {meta.get('version')} != {INDEX_VERSION}; rebuild it")
    chunks = [chunk_from_dict(d) for d in json.loads((path / "chunks.json").read_text(encoding="utf-8"))]
    with np.load(path / "arrays.npz") as z:
        arrays = {k: z[k] for k in z.files}
    bm25 = BM25(chunks, arrays=arrays, vocab=meta["vocab"], **meta["bm25"])
    if embedder is None:
        if meta["embedder"]["kind"] != "hashed":
            raise ValueError("index was built with a custom embedder; pass the same embedder to load_index")
        embedder = HashedEmbedder(meta["embedder"]["dim"])
    dense = DenseIndex(chunks, embedder, arrays=arrays)
    return HybridIndex(chunks, bm25=bm25, dense=dense, graph=CitationGraph(chunks), **meta["hybrid"])
