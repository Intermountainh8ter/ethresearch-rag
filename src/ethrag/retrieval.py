"""Field-boosted BM25, a sparse hashed "dense" baseline, weighted RRF, and query planning.

* ``BM25`` is an inverted index with BM25F-style field boosting: tokens from the document title
  (and id) and the section heading are added to the body term frequencies with a weight, so
  "EIP-1559" or "Specification" in a query lifts the right chunks without a separate index.
* ``HashedEmbedder`` / ``DenseIndex`` are a dependency-free stand-in for a neural embedder: hashed
  tf-idf over word + character-trigram features, cosine-compared. Rows are stored as a numpy
  CSC matrix so a 900-document corpus fits comfortably in memory. A real model plugs in through
  the ``Embedder`` protocol (``SentenceTransformerEmbedder``, not exercised in CI).
* ``HybridIndex`` fuses the two with weighted reciprocal-rank fusion, boosts or restricts to
  documents named in the query ("EIP-4844", "ERC-20"), and can expand results along the
  ``requires`` citation graph.
"""
from __future__ import annotations

import math
import re
import zlib
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .graph import CitationGraph, mentioned_numbers
from .ingest import Chunk

_TOKEN = re.compile(r"[A-Za-z0-9_]+(?:\.[0-9]+)?|0x[0-9a-fA-F]+")

_STOP_TEXT = """
a an the and or of to in on for with by as at from is are was were be been being this that these
those it its into than then there their they them we you your our he she his her which who whom
whose what when where why how do does did done not no nor but if so such can could should would
may might must will shall also any all each both either more most other some only own same too
very just about above after again against before below between during over under until up down
out off while because through per via i e g eg ie etc
"""
STOPWORDS = frozenset(_STOP_TEXT.split())


def tokenize(text: str) -> list[str]:
    toks = []
    for t in _TOKEN.findall(text):
        t = t.lower()
        toks.append(t)
        if "_" in t:  # BASE_FEE_MAX -> also base, fee, max
            toks.extend(p for p in t.split("_") if p)
    return toks


def content_tokens(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in STOPWORDS and len(t) > 1]


def _top(chunks: Sequence[Chunk], scores: np.ndarray, k: int) -> list[tuple[Chunk, float]]:
    k = max(0, min(k, len(chunks)))
    if k == 0:
        return []
    idx = np.argsort(-scores, kind="stable")[:k]
    return [(chunks[i], float(scores[i])) for i in idx]


# --------------------------------------------------------------------------------------- BM25
class BM25:
    """Inverted-index BM25 with title/section field boosting."""

    def __init__(self, chunks: Sequence[Chunk], k1: float = 1.5, b: float = 0.75,
                 title_boost: float = 2.0, section_boost: float = 1.0,
                 arrays: dict[str, np.ndarray] | None = None, vocab: list[str] | None = None):
        self.chunks = list(chunks)
        self.k1, self.b, self.title_boost, self.section_boost = k1, b, title_boost, section_boost
        if arrays is not None and vocab is not None:
            self._load(arrays, vocab)
        else:
            self._build()

    def _field_counts(self, c: Chunk) -> Counter[str]:
        tf: Counter[str] = Counter(tokenize(c.body))
        for t in tokenize(f"{c.doc} {c.title}"):
            tf[t] += self.title_boost
        for t in tokenize(c.section):
            tf[t] += self.section_boost
        return tf

    def _build(self) -> None:
        n = len(self.chunks)
        self.vocab: dict[str, int] = {}
        term_ids: list[int] = []
        chunk_ids: list[int] = []
        weights: list[float] = []
        lens = np.zeros(n, dtype=np.float32)
        for i, c in enumerate(self.chunks):
            tf = self._field_counts(c)
            lens[i] = len(tokenize(c.body))
            for t, f in tf.items():
                term_ids.append(self.vocab.setdefault(t, len(self.vocab)))
                chunk_ids.append(i)
                weights.append(f)
        t_arr = np.asarray(term_ids, dtype=np.int64)
        order = np.argsort(t_arr, kind="stable")
        self.idx = np.asarray(chunk_ids, dtype=np.int32)[order]
        self.tf = np.asarray(weights, dtype=np.float32)[order]
        counts = np.bincount(t_arr, minlength=len(self.vocab))
        self.indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        self.idf = np.log(1 + (n - counts + 0.5) / (counts + 0.5)).astype(np.float32)
        self.len = lens
        self._finish()

    def _finish(self) -> None:
        self.avg = float(self.len.mean()) if len(self.len) else 0.0
        denom = self.avg if self.avg > 0 else 1.0
        self.norm = (self.k1 * (1 - self.b + self.b * self.len / denom)).astype(np.float32)

    def _load(self, arrays: dict[str, np.ndarray], vocab: list[str]) -> None:
        self.vocab = {t: i for i, t in enumerate(vocab)}
        self.idx, self.tf = arrays["bm25_idx"], arrays["bm25_tf"]
        self.indptr, self.idf, self.len = arrays["bm25_indptr"], arrays["bm25_idf"], arrays["bm25_len"]
        self._finish()

    def to_arrays(self) -> tuple[dict[str, np.ndarray], list[str]]:
        vocab = [""] * len(self.vocab)
        for t, i in self.vocab.items():
            vocab[i] = t
        return ({"bm25_idx": self.idx, "bm25_tf": self.tf, "bm25_indptr": self.indptr,
                 "bm25_idf": self.idf, "bm25_len": self.len}, vocab)

    def scores(self, query: str) -> np.ndarray:
        s = np.zeros(len(self.chunks), dtype=np.float32)
        for w in tokenize(query):
            t = self.vocab.get(w)
            if t is None:
                continue
            lo, hi = self.indptr[t], self.indptr[t + 1]
            idx, f = self.idx[lo:hi], self.tf[lo:hi]
            s[idx] += self.idf[t] * f * (self.k1 + 1) / (f + self.norm[idx])
        return s

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        return _top(self.chunks, self.scores(query), k)


# -------------------------------------------------------------------------------------- dense
class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


class HashedEmbedder:
    """Hashed word + char-trigram features. ``embed_sparse`` is what ``DenseIndex`` uses."""

    def __init__(self, dim: int = 2**18):
        self.dim = dim
        self._cache: dict[str, np.ndarray] = {}

    def _token_feats(self, tok: str) -> np.ndarray:
        arr = self._cache.get(tok)
        if arr is None:
            feats = [tok]
            if len(tok) >= 3 and not tok[0].isdigit():
                p = f"<{tok}>"
                feats.extend(p[i:i + 3] for i in range(len(p) - 2))
            arr = np.fromiter((zlib.crc32(f.encode()) % self.dim for f in feats), dtype=np.int64,
                              count=len(feats))
            self._cache[tok] = arr
        return arr

    def features(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        """(unique feature ids, counts) for one text."""
        toks = tokenize(text)
        if not toks:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
        ids, counts = np.unique(np.concatenate([self._token_feats(t) for t in toks]), return_counts=True)
        return ids, counts.astype(np.float32)

    def embed_sparse(self, texts: Sequence[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """CSR (indptr, indices, counts) over ``texts``."""
        parts = [self.features(t) for t in texts]
        lengths = np.array([len(p[0]) for p in parts], dtype=np.int64)
        indptr = np.concatenate([[0], np.cumsum(lengths)])
        indices = np.concatenate([p[0] for p in parts]) if parts else np.zeros(0, dtype=np.int64)
        counts = np.concatenate([p[1] for p in parts]) if parts else np.zeros(0, dtype=np.float32)
        return indptr, indices, counts

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        m = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            ids, counts = self.features(text)
            m[i, ids] = counts
        return m


class SentenceTransformerEmbedder:  # pragma: no cover - optional dependency, not run in CI
    def __init__(self, model: str = "BAAI/bge-small-en-v1.5"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(self.model.encode(list(texts), normalize_embeddings=True))


class DenseIndex:
    """Cosine index. Sparse CSC for ``HashedEmbedder``; dense matrix for any other embedder."""

    def __init__(self, chunks: Sequence[Chunk], embedder: Embedder | None = None,
                 arrays: dict[str, np.ndarray] | None = None):
        self.chunks = list(chunks)
        self.embedder = embedder or HashedEmbedder()
        self.sparse = hasattr(self.embedder, "embed_sparse")
        if arrays is not None:
            self._load(arrays)
        elif self.sparse:
            self._build_sparse()
        else:
            m = self.embedder.embed([c.text for c in self.chunks]).astype(np.float32)
            self.m = m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)

    def _build_sparse(self) -> None:
        n, dim = len(self.chunks), self.embedder.dim
        indptr, indices, counts = self.embedder.embed_sparse([c.text for c in self.chunks])
        df = np.bincount(indices, minlength=dim)
        self.idf = (np.log((1 + n) / (1 + df)) + 1.0).astype(np.float32)
        vals = np.log1p(counts) * self.idf[indices]
        rows = np.repeat(np.arange(n, dtype=np.int32), np.diff(indptr))
        norms = np.sqrt(np.bincount(rows, weights=vals * vals, minlength=n))
        vals = (vals / np.maximum(norms[rows], 1e-12)).astype(np.float32)
        order = np.argsort(indices, kind="stable")
        self.col_indptr = np.concatenate([[0], np.cumsum(np.bincount(indices, minlength=dim))]).astype(np.int64)
        self.rows = rows[order]
        self.vals = vals[order]

    def _load(self, arrays: dict[str, np.ndarray]) -> None:
        if "dense_m" in arrays:
            self.sparse, self.m = False, arrays["dense_m"]
            return
        self.sparse = True
        self.idf, self.col_indptr = arrays["dense_idf"], arrays["dense_col_indptr"]
        self.rows, self.vals = arrays["dense_rows"], arrays["dense_vals"]

    def to_arrays(self) -> dict[str, np.ndarray]:
        if not self.sparse:
            return {"dense_m": self.m}
        return {"dense_idf": self.idf, "dense_col_indptr": self.col_indptr,
                "dense_rows": self.rows, "dense_vals": self.vals}

    def scores(self, query: str) -> np.ndarray:
        n = len(self.chunks)
        if not self.sparse:
            q = self.embedder.embed([query]).astype(np.float32)
            q = q / max(float(np.linalg.norm(q)), 1e-12)
            return (self.m @ q.T).ravel()
        ids, counts = self.embedder.features(query)
        if len(ids) == 0:
            return np.zeros(n, dtype=np.float32)
        qv = np.log1p(counts) * self.idf[ids]
        qv = qv / max(float(np.linalg.norm(qv)), 1e-12)
        lo, hi = self.col_indptr[ids], self.col_indptr[ids + 1]
        lengths = hi - lo
        if lengths.sum() == 0:
            return np.zeros(n, dtype=np.float32)
        gather = np.concatenate([np.arange(a, b) for a, b in zip(lo, hi)])
        weights = self.vals[gather] * np.repeat(qv, lengths)
        return np.bincount(self.rows[gather], weights=weights, minlength=n).astype(np.float32)

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        return _top(self.chunks, self.scores(query), k)


# ------------------------------------------------------------------------------- query planning
@dataclass
class QueryPlan:
    query: str
    terms: list[str]
    numbers: list[int]
    docs: list[str] = field(default_factory=list)  # named documents present in the corpus


def plan_query(query: str, by_number: dict[int, str] | None = None) -> QueryPlan:
    numbers = []
    for n in mentioned_numbers(query):
        if n not in numbers:
            numbers.append(n)
    docs = [by_number[n] for n in numbers if by_number and n in by_number]
    return QueryPlan(query, tokenize(query), numbers, docs)


def rrf_vector(scores: np.ndarray, rrf_k: int, top: int) -> np.ndarray:
    """Reciprocal-rank credit for the ``top`` best positions of ``scores`` (0 elsewhere)."""
    out = np.zeros(len(scores), dtype=np.float32)
    top = min(top, len(scores))
    if top == 0:
        return out
    order = np.argpartition(-scores, top - 1)[:top] if top < len(scores) else np.arange(len(scores))
    order = order[np.argsort(-scores[order], kind="stable")]
    nonzero = scores[order] > 0
    order = order[nonzero]
    out[order] = 1.0 / (rrf_k + np.arange(1, len(order) + 1))
    return out


# ------------------------------------------------------------------------------------- hybrid
class HybridIndex:
    """Weighted RRF of BM25 and dense rankings with EIP-number boosting and graph expansion."""

    def __init__(self, chunks: Sequence[Chunk], embedder: Embedder | None = None, rrf_k: int = 60,
                 bm25_weight: float = 0.5, eip_boost: float = 1.0, eip_mode: str = "boost",
                 top: int = 200, bm25: BM25 | None = None, dense: DenseIndex | None = None,
                 graph: CitationGraph | None = None):
        if eip_mode not in ("boost", "restrict", "off"):
            raise ValueError("eip_mode must be boost | restrict | off")
        self.chunks = list(chunks)
        self.bm25 = bm25 or BM25(self.chunks)
        self.dense = dense or DenseIndex(self.chunks, embedder)
        self.graph = graph or CitationGraph(self.chunks)
        self.rrf_k, self.bm25_weight, self.eip_boost, self.eip_mode, self.top = (
            rrf_k, bm25_weight, eip_boost, eip_mode, top)
        self.doc_chunks: dict[str, np.ndarray] = {}
        buckets: dict[str, list[int]] = {}
        for i, c in enumerate(self.chunks):
            buckets.setdefault(c.doc, []).append(i)
        self.doc_chunks = {d: np.asarray(ix, dtype=np.int64) for d, ix in buckets.items()}

    def plan(self, query: str) -> QueryPlan:
        return plan_query(query, self.graph.by_number)

    def rank_vectors(self, query: str) -> tuple[np.ndarray, np.ndarray]:
        return (rrf_vector(self.bm25.scores(query), self.rrf_k, self.top),
                rrf_vector(self.dense.scores(query), self.rrf_k, self.top))

    def fuse(self, rrf_b: np.ndarray, rrf_d: np.ndarray, named_docs: Sequence[str] = (),
             bm25_weight: float | None = None, eip_boost: float | None = None,
             eip_mode: str | None = None) -> np.ndarray:
        w = self.bm25_weight if bm25_weight is None else bm25_weight
        boost = self.eip_boost if eip_boost is None else eip_boost
        mode = self.eip_mode if eip_mode is None else eip_mode
        fused = (2 * w) * rrf_b + (2 * (1 - w)) * rrf_d  # w=0.5 is plain RRF
        if named_docs and mode != "off":
            parts = [self.doc_chunks[d] for d in named_docs if d in self.doc_chunks]
            idx = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)
            if idx.size:
                if mode == "restrict":
                    mask = np.zeros(len(fused), dtype=bool)
                    mask[idx] = True
                    fused = np.where(mask, fused, 0.0)
                else:
                    fused[idx] *= 1.0 + boost
        return fused

    def fused_scores(self, query: str, **kw) -> np.ndarray:
        rrf_b, rrf_d = self.rank_vectors(query)
        return self.fuse(rrf_b, rrf_d, self.plan(query).docs, **kw)

    def search(self, query: str, k: int = 5, expand_graph: bool = False, extra: int = 2,
               depth: int = 1, **kw) -> list[tuple[Chunk, float]]:
        fused = self.fused_scores(query, **kw)
        # A fused score of 0 means neither ranker surfaced the chunk (or "restrict" masked it):
        # padding the list with such chunks would only add arbitrary text.
        results = [(c, s) for c, s in _top(self.chunks, fused, k) if s > 0]
        if expand_graph and results:
            results += self.expand(results, fused, extra=extra, depth=depth)
        return results

    def expand(self, results: list[tuple[Chunk, float]], fused: np.ndarray, extra: int = 2,
               depth: int = 1, kinds: Sequence[str] = ("requires", "required_by")) -> list[tuple[Chunk, float]]:
        """Best chunk from each graph-neighbour of the top documents, up to ``extra`` of them."""
        present = {c.doc for c, _ in results}
        chosen_ids = {c.id for c, _ in results}
        candidates: dict[str, int] = {}
        for c, _ in results:
            for doc, _dist in self.graph.related(c.doc, depth=depth, kinds=kinds):
                if doc in present or doc in candidates or doc not in self.doc_chunks:
                    continue
                ix = self.doc_chunks[doc]
                best = int(ix[np.argmax(fused[ix])])
                if fused[best] > 0 and self.chunks[best].id not in chosen_ids:
                    candidates[doc] = best
        ranked = sorted(candidates.values(), key=lambda i: -fused[i])[:extra]
        return [(self.chunks[i], float(fused[i])) for i in ranked]


def normalize_scores(s: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0, 1]; constant vectors map to 0."""
    lo, hi = float(s.min()) if s.size else 0.0, float(s.max()) if s.size else 0.0
    if hi - lo < 1e-12:
        return np.zeros_like(s, dtype=np.float32)
    return ((s - lo) / (hi - lo)).astype(np.float32)


def log_len(c: Chunk) -> float:
    return math.log1p(len(c.body.split()))
