from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .ingest import Chunk, load_corpus
from .llm import LLM
from .retrieval import HybridIndex, tokenize
from .verify import VerificationResult, sentences, verify_answer

SYSTEM = (
    "Answer using ONLY the provided context chunks. Every sentence must end with a citation in the "
    "form [eip-NNNN#CCC] naming the chunk id that supports it. Copy constants, numbers and "
    "identifiers exactly from the context. If the context does not contain the answer, reply "
    "exactly: INSUFFICIENT_CONTEXT"
)
REFUSAL = "I can't give a verified answer from the retrieved EIP text."


@dataclass
class Answer:
    question: str
    text: str
    verified: bool
    retrieved: list[str]
    issues: list[str] = field(default_factory=list)
    raw: str = ""
    verification: VerificationResult | None = None


class RagPipeline:
    def __init__(self, chunks: Sequence[Chunk], llm: LLM | None = None, k: int = 5):
        self.chunks = {c.id: c for c in chunks}
        self.index = HybridIndex(chunks)
        self.llm, self.k = llm, k

    @classmethod
    def from_directory(cls, directory: str | Path, llm: LLM | None = None, **kw) -> RagPipeline:
        return cls(load_corpus(directory), llm, **kw)

    def retrieve(self, question: str) -> list[Chunk]:
        return [c for c, _ in self.index.search(question, self.k)]

    def ask(self, question: str) -> Answer:
        ctx = self.retrieve(question)
        ids = [c.id for c in ctx]
        raw = self._generate(question, ctx)
        if raw.strip() == "INSUFFICIENT_CONTEXT":
            return Answer(question, REFUSAL, False, ids, ["model reported insufficient context"], raw)
        v = verify_answer(raw, self.chunks, ids)
        if v.ok:
            return Answer(question, raw, True, ids, [], raw, v)
        return Answer(question, f"{REFUSAL} Unverified draft withheld.", False, ids, v.issues, raw, v)

    def _generate(self, question: str, ctx: Sequence[Chunk]) -> str:
        if self.llm is None:
            return extractive_answer(question, ctx)
        context = "\n\n".join(f"[{c.id}]\n{c.text}" for c in ctx)
        return self.llm.complete(SYSTEM, f"Context:\n{context}\n\nQuestion: {question}")


def extractive_answer(question: str, ctx: Sequence[Chunk]) -> str:
    """No-LLM baseline: the single best-matching sentence from the retrieved chunks, cited."""
    q = set(tokenize(question))
    best, best_score = None, 0.0
    for c in ctx:
        for s in sentences(re.sub(r"```.*?```", " ", c.text, flags=re.DOTALL)):
            toks = set(tokenize(s))
            score = len(q & toks) / (1 + 0.02 * len(toks))
            if score > best_score:
                best, best_score = (s, c.id), score
    if best is None:
        return "INSUFFICIENT_CONTEXT"
    s, cid = best
    return f"{' '.join(s.split()).rstrip('.')} [{cid}]."
