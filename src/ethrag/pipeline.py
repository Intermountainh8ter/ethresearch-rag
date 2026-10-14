from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .ingest import Chunk, load_corpus
from .llm import LLM
from .rerank import LogisticReranker
from .retrieval import HybridIndex, content_tokens, tokenize
from .verify import VerificationResult, claim_tokens, sentences, verify_answer

SYSTEM = (
    "Answer using ONLY the provided context chunks. Every sentence must end with a citation in the "
    "form [doc#CCC] naming the chunk id that supports it (for example [eip-1559#003]). Copy "
    "constants, numbers and identifiers exactly from the context. If the context does not contain "
    "the answer, reply exactly: INSUFFICIENT_CONTEXT"
)
REFUSAL = "I can't give a verified answer from the retrieved spec text."


@dataclass
class Answer:
    question: str
    text: str
    verified: bool
    retrieved: list[str]
    issues: list[str] = field(default_factory=list)
    raw: str = ""
    verification: VerificationResult | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def provenance(self) -> dict[str, list[str]]:
        return self.verification.provenance() if self.verification else {}

    def to_dict(self) -> dict:
        return {
            "question": self.question, "verified": self.verified, "answer": self.text,
            "retrieved": self.retrieved, "issues": self.issues, "warnings": self.warnings,
            "provenance": self.provenance, "raw": self.raw,
            "verification": self.verification.to_dict() if self.verification else None,
        }


class RagPipeline:
    def __init__(self, chunks: Sequence[Chunk], llm: LLM | None = None, k: int = 5,
                 index: HybridIndex | None = None, reranker: LogisticReranker | None = None,
                 min_coverage: float = 0.5, min_relevance: float = 0.25,
                 check_negation: bool = True, expand_graph: bool = False):
        self.chunks = {c.id: c for c in chunks}
        self.index = index or HybridIndex(list(chunks))
        self.llm, self.k, self.reranker = llm, k, reranker
        self.min_coverage, self.min_relevance = min_coverage, min_relevance
        self.check_negation, self.expand_graph = check_negation, expand_graph

    @classmethod
    def from_directory(cls, directory: str | Path, llm: LLM | None = None, **kw) -> RagPipeline:
        return cls(load_corpus(directory), llm, **kw)

    @classmethod
    def from_index(cls, path: str | Path, llm: LLM | None = None, **kw) -> RagPipeline:
        from .store import load_index

        index = load_index(path)
        return cls(index.chunks, llm, index=index, **kw)

    @property
    def graph(self):
        return self.index.graph

    def related(self, doc: str, depth: int = 1) -> list[tuple[str, int]]:
        return self.index.graph.related(doc, depth)

    def search(self, question: str, k: int | None = None) -> list[tuple[Chunk, float]]:
        """Ranked (chunk, score) pairs using the configured retriever (reranker or graph expansion)."""
        k = self.k if k is None else k
        if self.reranker is not None:
            return self.reranker.rerank(self.index, question, k)
        return self.index.search(question, k, expand_graph=self.expand_graph)

    def retrieve(self, question: str, k: int | None = None) -> list[Chunk]:
        return [c for c, _ in self.search(question, k)]

    def verify(self, question: str, raw: str, ids: Sequence[str]) -> VerificationResult:
        return verify_answer(raw, self.chunks, ids, question=question,
                             min_coverage=self.min_coverage, min_relevance=self.min_relevance,
                             check_negation=self.check_negation)

    def ask(self, question: str, k: int | None = None) -> Answer:
        ctx = self.retrieve(question, k)
        ids = [c.id for c in ctx]
        raw = self._generate(question, ctx)
        if raw.strip() == "INSUFFICIENT_CONTEXT":
            return Answer(question, REFUSAL, False, ids, ["model reported insufficient context"], raw)
        v = self.verify(question, raw, ids)
        if v.ok:
            return Answer(question, raw, True, ids, [], raw, v, v.warnings)
        return Answer(question, f"{REFUSAL} Unverified draft withheld.", False, ids, v.issues, raw,
                      v, v.warnings)

    def _generate(self, question: str, ctx: Sequence[Chunk]) -> str:
        if self.llm is None:
            return extractive_answer(question, ctx)
        context = "\n\n".join(f"[{c.id}]\n{c.text}" for c in ctx)
        return self.llm.complete(SYSTEM, f"Context:\n{context}\n\nQuestion: {question}")


def extractive_answer(question: str, ctx: Sequence[Chunk]) -> str:
    """No-LLM baseline: the single best-matching prose sentence from the retrieved chunks, cited.

    Scoring favours sentences that share content words with the question, with a bonus for
    sharing an identifier or number, and a mild length penalty. Fenced code is skipped.
    """
    q = set(content_tokens(question)) or set(tokenize(question))
    q_claims = claim_tokens(question)
    best, best_score = None, 0.0
    for c in ctx:
        prose = re.sub(r"```.*?```", " ", c.body, flags=re.DOTALL)
        prose = re.sub(r"^\s*\|.*$", " ", prose, flags=re.MULTILINE)  # drop table rows
        for s in sentences(prose):
            toks = set(tokenize(s))
            if len(toks) < 3:
                continue
            overlap = len(q & toks)
            if overlap == 0:
                continue
            bonus = 0.5 * len(q_claims & claim_tokens(s))
            score = (overlap + bonus) / (1 + 0.02 * len(toks))
            if score > best_score:
                best, best_score = (s, c.id), score
    if best is None:
        return "INSUFFICIENT_CONTEXT"
    s, cid = best
    return f"{' '.join(s.split()).rstrip('.')} [{cid}]."
