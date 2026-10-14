"""A lightweight cross-feature reranker trained with numpy logistic regression.

For a (query, candidate chunk) pair the features are cheap, interpretable signals: normalised
BM25 / dense / fused scores, lexical overlap with the query, exact identifier and number
matches, whether the chunk's document is named in the query, section-type priors, document
status priors, and chunk kind / length. Weights are fit on the TRAIN split of the eval set and
reported on the HELD-OUT split; see docs/EVALUATION.md.
"""
from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .ingest import Chunk
from .retrieval import HybridIndex, content_tokens, log_len, normalize_scores, tokenize

FEATURE_NAMES = [
    "bm25", "dense", "fused", "lex_overlap", "lex_coverage", "ident_match", "num_match",
    "eip_named", "sec_spec", "sec_abstract", "sec_motivation", "sec_rationale", "sec_security",
    "sec_other", "status_final", "status_draft", "status_stagnant", "kind_code", "kind_table",
    "log_len",
]
_IDENT = re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b|\b[A-Z]{3,}\b")
_NUMLIKE = re.compile(r"0x[0-9a-fA-F]+|\b\d+\b")


def section_type(section: str) -> str:
    s = section.lower()
    for key, name in (("specification", "spec"), ("abstract", "abstract"), ("summary", "abstract"),
                      ("motivation", "motivation"), ("rationale", "rationale"),
                      ("security", "security")):
        if key in s:
            return name
    return "other"


def status_class(status: str) -> str:
    s = status.lower()
    if s in ("final", "last call", "living", "spec", "docs"):
        return "final"
    if s in ("draft", "review"):
        return "draft"
    return "stagnant"  # stagnant, withdrawn, unknown


def query_signals(query: str) -> tuple[set[str], set[str], set[str]]:
    """(content tokens, identifiers, number-like tokens) of the query, lower-cased."""
    idents = {m.lower() for m in _IDENT.findall(query)}
    nums = {m.lower() for m in _NUMLIKE.findall(query)}
    return set(content_tokens(query)), idents, nums


def chunk_features(index: HybridIndex, query: str, cand: Sequence[int], bm25: np.ndarray,
                   dense: np.ndarray, fused: np.ndarray) -> np.ndarray:
    """Feature matrix (len(cand), len(FEATURE_NAMES)) for candidate chunk indices."""
    q_content, q_idents, q_nums = query_signals(query)
    named = set(index.plan(query).docs)
    nb, nd, nf = normalize_scores(bm25), normalize_scores(dense), normalize_scores(fused)
    X = np.zeros((len(cand), len(FEATURE_NAMES)), dtype=np.float32)
    for r, i in enumerate(cand):
        c: Chunk = index.chunks[i]
        toks = set(tokenize(c.text))
        overlap = len(q_content & toks)
        sec, st = section_type(c.section), status_class(c.status)
        X[r] = [
            nb[i], nd[i], nf[i],
            overlap / max(1, len(q_content)),
            overlap / max(1, len(toks)) * 10.0,
            float(bool(q_idents) and q_idents <= toks),
            float(bool(q_nums) and bool(q_nums & toks)),
            float(c.doc in named),
            float(sec == "spec"), float(sec == "abstract"), float(sec == "motivation"),
            float(sec == "rationale"), float(sec == "security"), float(sec == "other"),
            float(st == "final"), float(st == "draft"), float(st == "stagnant"),
            float(c.kind == "code"), float(c.kind == "table"),
            log_len(c),
        ]
    return X


class LogisticReranker:
    def __init__(self, weights: np.ndarray | None = None, bias: float = 0.0,
                 mean: np.ndarray | None = None, std: np.ndarray | None = None):
        n = len(FEATURE_NAMES)
        self.w = np.zeros(n, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
        self.b = float(bias)
        self.mean = np.zeros(n) if mean is None else np.asarray(mean, dtype=np.float64)
        self.std = np.ones(n) if std is None else np.asarray(std, dtype=np.float64)
        self.history: list[float] = []

    def _z(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=np.float64) - self.mean) / self.std

    def fit(self, X: np.ndarray, y: np.ndarray, l2: float = 1e-2, lr: float = 0.1,
            epochs: int = 400, balance: bool = True) -> LogisticReranker:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != len(FEATURE_NAMES) or len(y) != len(X):
            raise ValueError("X must be (n, len(FEATURE_NAMES)) and y of length n")
        self.mean = X.mean(axis=0)
        self.std = np.where(X.std(axis=0) > 1e-9, X.std(axis=0), 1.0)
        Z = self._z(X)
        pos = max(1.0, y.sum())
        sw = np.where(y > 0, (len(y) - pos) / pos, 1.0) if balance else np.ones(len(y))
        sw = sw / sw.mean()
        self.w = np.zeros(Z.shape[1])
        self.b = 0.0
        self.history = []
        for _ in range(epochs):
            p = 1.0 / (1.0 + np.exp(-(Z @ self.w + self.b)))
            g = (p - y) * sw
            self.w -= lr * (Z.T @ g / len(y) + l2 * self.w)
            self.b -= lr * float(g.mean())
            eps = 1e-9
            self.history.append(float(-np.mean(sw * (y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(self._z(X) @ self.w + self.b)))

    def feature_weights(self) -> dict[str, float]:
        return {n: float(w) for n, w in zip(FEATURE_NAMES, self.w)}

    def rerank(self, index: HybridIndex, query: str, k: int = 5, pool: int = 50) -> list[tuple[Chunk, float]]:
        bm25, dense = index.bm25.scores(query), index.dense.scores(query)
        fused = index.fused_scores(query)
        pool = min(pool, len(index.chunks))
        cand = np.argsort(-fused, kind="stable")[:pool]
        cand = [int(i) for i in cand if fused[i] > 0] or [int(i) for i in cand]
        if not cand:
            return []
        p = self.predict(chunk_features(index, query, cand, bm25, dense, fused))
        order = np.argsort(-p, kind="stable")[:k]
        return [(index.chunks[cand[i]], float(p[i])) for i in order]

    def to_dict(self) -> dict:
        return {"features": FEATURE_NAMES, "weights": self.w.tolist(), "bias": self.b,
                "mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> LogisticReranker:
        if d.get("features") != FEATURE_NAMES:
            raise ValueError("reranker weights were trained with a different feature set")
        return cls(np.asarray(d["weights"]), d["bias"], np.asarray(d["mean"]), np.asarray(d["std"]))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> LogisticReranker:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class QuestionPool:
    """Candidate chunks of one question with their features and gold labels."""

    qid: str
    cand: list[int]
    X: np.ndarray
    y: np.ndarray


def build_pools(index: HybridIndex, questions: Sequence[dict], gold_ids: dict[str, set[str]],
                pool: int = 50) -> list[QuestionPool]:
    """Feature matrix + labels over the fused top-``pool`` candidates of every question."""
    out = []
    for q in questions:
        bm25, dense = index.bm25.scores(q["q"]), index.dense.scores(q["q"])
        fused = index.fused_scores(q["q"])
        cand = [int(i) for i in np.argsort(-fused, kind="stable")[:pool] if fused[i] > 0]
        if not cand:
            continue
        gold = gold_ids.get(q["id"], set())
        out.append(QuestionPool(
            q["id"], cand, chunk_features(index, q["q"], cand, bm25, dense, fused),
            np.array([1.0 if index.chunks[i].id in gold else 0.0 for i in cand])))
    return out


def stack_pools(pools: Sequence[QuestionPool]) -> tuple[np.ndarray, np.ndarray]:
    if not pools:
        return np.zeros((0, len(FEATURE_NAMES))), np.zeros(0)
    return np.vstack([p.X for p in pools]), np.concatenate([p.y for p in pools])


def training_pairs(index: HybridIndex, questions: Sequence[dict], gold_ids: dict[str, set[str]],
                   pool: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Stack (features, label) rows over the candidate pool of every question."""
    return stack_pools(build_pools(index, questions, gold_ids, pool))


def cv_select_l2(pools: Sequence[QuestionPool], groups: Sequence[str], l2_grid: Sequence[float],
                 folds: int = 3, k: int = 5, seed: int = 0, **fit_kw) -> tuple[float, dict[float, float]]:
    """Pick the L2 strength by grouped k-fold CV on *training* pools only.

    ``groups`` (one per pool, e.g. the gold document) never straddle a fold, so the score is the
    recall@``k`` of the reranker on documents it was not fit on. Ties go to the stronger penalty.
    Returns (best l2, {l2: mean CV recall@k}).
    """
    import random

    if len(pools) != len(groups):
        raise ValueError("one group label per pool is required")
    names = sorted(set(groups))
    random.Random(seed).shuffle(names)
    fold_of = {g: i % folds for i, g in enumerate(names)}
    scores: dict[float, float] = {}
    for l2 in l2_grid:
        hits, total = 0, 0
        for f in range(folds):
            tr = [p for p, g in zip(pools, groups, strict=True) if fold_of[g] != f]
            te = [p for p, g in zip(pools, groups, strict=True) if fold_of[g] == f]
            if not tr or not te:
                continue
            model = LogisticReranker().fit(*stack_pools(tr), l2=l2, **fit_kw)
            for p in te:
                top = np.argsort(-model.predict(p.X), kind="stable")[:k]
                hits += bool(p.y[top].sum() > 0)
                total += 1
        scores[l2] = hits / total if total else 0.0
    best = max(l2_grid, key=lambda v: (scores[v], v))
    return best, scores
