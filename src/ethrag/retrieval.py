"""BM25, a hashed char-n-gram dense baseline, and reciprocal-rank fusion.

The dense retriever here is a *hashed TF-IDF over word + char-trigram features*, L2-normalised
and compared by cosine. It is a cheap, dependency-free stand-in for a neural embedder; plug a
real one in via the ``Embedder`` protocol (see ``SentenceTransformerEmbedder``).
"""
from __future__ import annotations

import math
import re
import zlib
from collections import Counter
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from .ingest import Chunk

_TOKEN = re.compile(r"[A-Za-z0-9_]+(?:\.[0-9]+)?|0x[0-9a-fA-F]+")


def tokenize(text: str) -> list[str]:
    toks = []
    for t in _TOKEN.findall(text):
        t = t.lower()
        toks.append(t)
        if "_" in t:  # BASE_FEE_MAX -> also base, fee, max
            toks.extend(p for p in t.split("_") if p)
    return toks


class BM25:
    def __init__(self, chunks: Sequence[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = list(chunks)
        self.k1, self.b = k1, b
        self.tf = [Counter(tokenize(c.text)) for c in self.chunks]
        self.len = np.array([sum(t.values()) for t in self.tf], dtype=float)
        self.avg = float(self.len.mean()) if len(self.len) else 0.0
        df: Counter[str] = Counter()
        for t in self.tf:
            df.update(t.keys())
        n = len(self.chunks)
        self.idf = {w: math.log(1 + (n - d + 0.5) / (d + 0.5)) for w, d in df.items()}

    def scores(self, query: str) -> np.ndarray:
        q = tokenize(query)
        s = np.zeros(len(self.chunks))
        for i, tf in enumerate(self.tf):
            norm = self.k1 * (1 - self.b + self.b * self.len[i] / self.avg)
            for w in q:
                f = tf.get(w)
                if f:
                    s[i] += self.idf[w] * f * (self.k1 + 1) / (f + norm)
        return s

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        return _top(self.chunks, self.scores(query), k)


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


class HashedEmbedder:
    def __init__(self, dim: int = 2048):
        self.dim = dim

    def _feats(self, text: str) -> list[str]:
        toks = tokenize(text)
        feats = list(toks)
        for t in toks:
            p = f"<{t}>"
            feats.extend(p[i:i + 3] for i in range(len(p) - 2))
        return feats

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        m = np.zeros((len(texts), self.dim))
        for i, text in enumerate(texts):
            for f in self._feats(text):
                m[i, zlib.crc32(f.encode()) % self.dim] += 1.0
        return m


class SentenceTransformerEmbedder:  # pragma: no cover - optional dependency, not run in CI
    def __init__(self, model: str = "BAAI/bge-small-en-v1.5"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(self.model.encode(list(texts), normalize_embeddings=True))


class DenseIndex:
    def __init__(self, chunks: Sequence[Chunk], embedder: Embedder | None = None):
        self.chunks = list(chunks)
        self.embedder = embedder or HashedEmbedder()
        m = self.embedder.embed([c.text for c in self.chunks])
        if isinstance(self.embedder, HashedEmbedder):  # tf-idf weighting + l2
            df = (m > 0).sum(axis=0)
            self.idf = np.log((1 + len(m)) / (1 + df)) + 1.0
            m = np.log1p(m) * self.idf
        self.m = m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)

    def scores(self, query: str) -> np.ndarray:
        q = self.embedder.embed([query])
        if isinstance(self.embedder, HashedEmbedder):
            q = np.log1p(q) * self.idf
        q = q / max(float(np.linalg.norm(q)), 1e-12)
        return (self.m @ q.T).ravel()

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        return _top(self.chunks, self.scores(query), k)


def _top(chunks: Sequence[Chunk], scores: np.ndarray, k: int) -> list[tuple[Chunk, float]]:
    idx = np.argsort(-scores, kind="stable")[:k]
    return [(chunks[i], float(scores[i])) for i in idx]


class HybridIndex:
    """Reciprocal-rank fusion of BM25 and dense rankings."""

    def __init__(self, chunks: Sequence[Chunk], embedder: Embedder | None = None, rrf_k: int = 60):
        self.chunks = list(chunks)
        self.bm25, self.dense, self.rrf_k = BM25(chunks), DenseIndex(chunks, embedder), rrf_k

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        fused = np.zeros(len(self.chunks))
        for s in (self.bm25.scores(query), self.dense.scores(query)):
            order = np.argsort(-s, kind="stable")
            for rank, i in enumerate(order[:100]):
                fused[i] += 1.0 / (self.rrf_k + rank + 1)
        return _top(self.chunks, fused, k)
