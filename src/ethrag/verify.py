"""Deterministic grounding check for cited answers.

An answer passes only if (1) every sentence carries at least one citation, (2) every citation
points at a chunk that was actually retrieved, and (3) every *checkable claim token* in the
sentence (numbers, hex values, ``2**n`` expressions, ALL_CAPS identifiers) appears as a token in
the cited chunks. This catches the failure mode that matters most for specs: a fluent answer with
the wrong constant. It does not verify paraphrased prose; see README limitations.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from .ingest import Chunk

CITATION = re.compile(r"\[(eip-\d+#\d{3})\]")
_NUM = re.compile(r"0x[0-9a-fA-F]+|\d[\d,]*(?:\.\d+)?(?:\s?\*\*\s?\d+)?")
_IDENT = re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b|\b[A-Z]{3,}\b")
_STOP_CAPS = {"EIP", "ETH", "EVM", "THE", "AND", "NOT", "MUST", "SHALL", "SHOULD", "MAY", "RLP", "ERC"}


def _norm(tok: str) -> str:
    return re.sub(r"[\s,]", "", tok).lower().rstrip(".")


def claim_tokens(text: str) -> set[str]:
    text = CITATION.sub("", text)
    toks = {_norm(m) for m in _NUM.findall(text)}
    toks |= {_norm(m) for m in _IDENT.findall(text) if m not in _STOP_CAPS}
    return {t for t in toks if t}


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z`])|\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


@dataclass
class VerificationResult:
    ok: bool
    issues: list[str] = field(default_factory=list)
    checked_tokens: int = 0
    supported_tokens: int = 0

    @property
    def support_ratio(self) -> float:
        return 1.0 if self.checked_tokens == 0 else self.supported_tokens / self.checked_tokens


def verify_answer(answer: str, chunks: Mapping[str, Chunk], retrieved_ids: Sequence[str]) -> VerificationResult:
    res = VerificationResult(ok=True)
    retrieved = set(retrieved_ids)
    token_cache: dict[str, set[str]] = {}

    def chunk_tokens(cid: str) -> set[str]:
        if cid not in token_cache:
            token_cache[cid] = claim_tokens(chunks[cid].text)
        return token_cache[cid]

    for s in sentences(answer):
        cites = CITATION.findall(s)
        if not cites:
            res.ok = False
            res.issues.append(f"uncited sentence: {s[:80]!r}")
            continue
        valid = []
        for c in cites:
            if c not in chunks or c not in retrieved:
                res.ok = False
                res.issues.append(f"citation {c} was not retrieved")
            else:
                valid.append(c)
        support: set[str] = set().union(*(chunk_tokens(c) for c in valid)) if valid else set()
        for tok in sorted(claim_tokens(s)):
            res.checked_tokens += 1
            if tok in support:
                res.supported_tokens += 1
            else:
                res.ok = False
                res.issues.append(f"unsupported claim {tok!r} not found in cited chunks {cites}")
    return res
