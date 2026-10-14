"""Deterministic grounding check for cited answers, with per-claim provenance.

An answer passes only if

1. every sentence carries at least one citation ``[doc#NNN]``;
2. every citation points at a chunk that was actually retrieved;
3. every *checkable claim token* in the sentence (numbers, hex values, ``2**n`` expressions,
   ALL_CAPS identifiers, numbers with units such as ``32 ETH``) appears as a token in the cited
   chunks, either verbatim or as a recorded equivalent (``0x48`` = ``72``, ``2**12`` = ``4096``,
   ``32 ETH`` = ``32000000000`` gwei);
4. (optional) a negated sentence ("does NOT burn") has a negated counterpart in the cited text;
5. (optional) enough of the sentence's content words appear in the cited chunks
   (``min_coverage``), so a paraphrase with no checkable token is not waved through;
6. (optional) the answer is about the question: it shares key terms with it or cites a document
   the question names (``min_relevance``).

Each check records provenance (``VerificationResult.claims``) naming the chunk that supported
each token. Substring matches never count (``8`` is not found inside ``58``). What the verifier
cannot catch is documented in docs/EVALUATION.md and in ``tests/test_verify.py``.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from .graph import mentioned_numbers
from .ingest import Chunk
from .retrieval import STOPWORDS, tokenize

CITATION = re.compile(r"\[([a-z][a-z0-9-]*#\d{3,})\]")
_NUM = re.compile(
    r"0x[0-9a-fA-F]+"
    r"|\d[\d,_]*(?:\.\d+)?(?:\s?(?:\*\*|\^)\s?\d+)?(?:[eE][+-]?\d+)?"
)
_IDENT = re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b|\b[A-Z]{3,}\b")
_STOP_CAPS = {"EIP", "ETH", "EVM", "THE", "AND", "NOT", "MUST", "SHALL", "SHOULD", "MAY", "RLP",
              "ERC", "JSON", "RPC", "API", "URL", "NOTE", "YES", "FOR"}
_UNIT = re.compile(r"(\d[\d,_]*)\s*(ETH|ether|Gwei|gwei|wei|[kMG])\b")
_UNIT_SCALE = {"eth": 10**18, "ether": 10**18, "gwei": 10**9, "wei": 1, "k": 10**3, "m": 10**6,
               "g": 10**9}
_NEGATION = re.compile(r"\b(?:not|never|cannot|no longer|neither|nor)\b|\w+n't\b", re.IGNORECASE)
_WORD_STEM = re.compile(r"(ies|ied|ing|ed|es|s)$")


def _norm(tok: str) -> str:
    """Normalise a *numeric* token: drop separators (``1,000`` / ``1_000``), lower-case, canonical hex."""
    t = re.sub(r"[\s,_]", "", tok).lower().rstrip(".")
    t = t.replace("^", "**")
    if t.startswith("0x") and len(t) > 2:
        return "0x" + (t[2:].lstrip("0") or "0")
    return t


def _norm_ident(tok: str) -> str:
    """Normalise an identifier: case-folded but underscores kept (GAS_PER_BLOB != GASPERBLOB)."""
    return tok.lower()


def number_variants(tok: str) -> set[str]:
    """Equivalent spellings of a normalised numeric token (hex <-> decimal, 2**n <-> value)."""
    out = {tok}
    try:
        if tok.startswith("0x"):
            out.add(str(int(tok, 16)))
        elif "**" in tok:
            base, exp = tok.split("**", 1)
            if base.isdigit() and exp.isdigit() and int(exp) <= 512:
                out.add(str(int(base) ** int(exp)))
        elif tok.isdigit():
            v = int(tok)
            if v >= 16:
                out.add(_norm(hex(v)))
            if v > 1 and v & (v - 1) == 0:
                out.add(f"2**{v.bit_length() - 1}")
    except ValueError:
        pass
    return out


def unit_variants(text: str) -> dict[str, set[str]]:
    """``{"32": {"32000000000", "32000000000000000000"}}`` for ``32 ETH`` style phrases."""
    out: dict[str, set[str]] = {}
    for num, unit in _UNIT.findall(text):
        base = _norm(num)
        if not base.isdigit():
            continue
        scale = _UNIT_SCALE[unit.lower()]
        v = int(base) * scale
        alts = {str(v)}
        if unit.lower() in ("eth", "ether"):
            alts.add(str(int(base) * 10**9))  # gwei spelling
        out.setdefault(base, set()).update(alts)
    return out


def claim_tokens(text: str) -> set[str]:
    text = CITATION.sub("", text)
    toks = {_norm(m) for m in _NUM.findall(text)}
    toks |= {_norm_ident(m) for m in _IDENT.findall(text) if m not in _STOP_CAPS}
    return {t for t in toks if t}


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z`\[(\"])|\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def stem(tok: str) -> str:
    return _WORD_STEM.sub("", tok) if len(tok) > 4 else tok


def content_words(text: str) -> set[str]:
    text = CITATION.sub("", text)
    return {stem(t) for t in tokenize(text) if t not in STOPWORDS and len(t) > 1}


def has_negation(text: str) -> bool:
    return bool(_NEGATION.search(CITATION.sub("", text)))


def best_source_sentence(sentence: str, chunk_texts: Sequence[str]) -> tuple[str, float]:
    """Source sentence with the largest content-word recall of ``sentence``."""
    target = content_words(sentence)
    best, best_score = "", 0.0
    if not target:
        return best, best_score
    for text in chunk_texts:
        for s in sentences(re.sub(r"```.*?```", " ", text, flags=re.DOTALL)):
            score = len(target & content_words(s)) / len(target)
            if score > best_score:
                best, best_score = s, score
    return best, best_score


def question_relevance(question: str, answer: str) -> float:
    """Fraction of the question's content words that the answer repeats (stemmed)."""
    q = content_words(question)
    if not q:
        return 1.0
    return len(q & content_words(answer)) / len(q)


@dataclass
class ClaimCheck:
    sentence: int
    token: str
    supported: bool
    chunk: str = ""
    kind: str = ""  # exact | hex-dec | pow2 | unit


@dataclass
class VerificationResult:
    ok: bool
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checked_tokens: int = 0
    supported_tokens: int = 0
    claims: list[ClaimCheck] = field(default_factory=list)
    coverage: list[float] = field(default_factory=list)
    relevance: float | None = None

    @property
    def support_ratio(self) -> float:
        return 1.0 if self.checked_tokens == 0 else self.supported_tokens / self.checked_tokens

    @property
    def min_coverage(self) -> float:
        return min(self.coverage) if self.coverage else 1.0

    def provenance(self) -> dict[str, list[str]]:
        """``{chunk_id: [tokens it supported]}``."""
        out: dict[str, list[str]] = {}
        for c in self.claims:
            if c.supported:
                out.setdefault(c.chunk, []).append(c.token)
        return out

    def to_dict(self) -> dict:
        return {
            "ok": self.ok, "issues": self.issues, "warnings": self.warnings,
            "checked_tokens": self.checked_tokens, "supported_tokens": self.supported_tokens,
            "support_ratio": self.support_ratio, "coverage": self.coverage,
            "relevance": self.relevance,
            "claims": [c.__dict__ for c in self.claims],
        }


def _find_support(tok: str, cited: Sequence[str], chunk_tokens: Mapping[str, set[str]],
                  units: Mapping[str, set[str]]) -> tuple[str, str]:
    """(chunk id, kind) of the first cited chunk supporting ``tok``; ("", "") if none."""
    for cid in cited:
        if tok in chunk_tokens[cid]:
            return cid, "exact"
    variants = number_variants(tok) - {tok}
    for cid in cited:
        for v in variants:
            if v in chunk_tokens[cid]:
                return cid, "hex-dec" if (v.startswith("0x") or tok.startswith("0x")) else "pow2"
    for cid in cited:
        for v in units.get(tok, ()):
            if v in chunk_tokens[cid]:
                return cid, "unit"
    return "", ""


def verify_answer(answer: str, chunks: Mapping[str, Chunk], retrieved_ids: Sequence[str],
                  question: str | None = None, min_coverage: float = 0.0,
                  min_relevance: float = 0.0, check_negation: bool = True) -> VerificationResult:
    res = VerificationResult(ok=True)
    retrieved = set(retrieved_ids)
    token_cache: dict[str, set[str]] = {}
    words_cache: dict[str, set[str]] = {}

    def chunk_tokens(cid: str) -> set[str]:
        if cid not in token_cache:
            token_cache[cid] = claim_tokens(chunks[cid].text)
        return token_cache[cid]

    def chunk_words(cid: str) -> set[str]:
        if cid not in words_cache:
            words_cache[cid] = content_words(chunks[cid].text)
        return words_cache[cid]

    sents = sentences(answer)
    for si, s in enumerate(sents):
        cites = CITATION.findall(s)
        if not cites:
            res.ok = False
            res.issues.append(f"uncited sentence: {s[:80]!r}")
            res.coverage.append(0.0)
            continue
        valid = []
        for c in cites:
            if c not in chunks or c not in retrieved:
                res.ok = False
                res.issues.append(f"citation {c} was not retrieved")
            elif c not in valid:
                valid.append(c)
        tokens_by_chunk = {c: chunk_tokens(c) for c in valid}
        units = unit_variants(CITATION.sub("", s))
        for tok in sorted(claim_tokens(s)):
            res.checked_tokens += 1
            cid, kind = _find_support(tok, valid, tokens_by_chunk, units)
            res.claims.append(ClaimCheck(si, tok, bool(cid), cid, kind))
            if cid:
                res.supported_tokens += 1
            else:
                res.ok = False
                res.issues.append(f"unsupported claim {tok!r} not found in cited chunks {cites}")
        words = content_words(s)
        support_words: set[str] = set().union(*(chunk_words(c) for c in valid)) if valid else set()
        cov = len(words & support_words) / len(words) if words else 1.0
        res.coverage.append(cov)
        if cov < min_coverage:
            res.ok = False
            res.issues.append(f"sentence {si} coverage {cov:.2f} < {min_coverage:.2f}: {s[:60]!r}")
        if check_negation and valid:
            src, score = best_source_sentence(s, [chunks[c].text for c in valid])
            if score >= 0.5:
                if has_negation(s) and not has_negation(src):
                    res.ok = False
                    res.issues.append(f"negation in answer sentence {si} has no negated source: {src[:60]!r}")
                elif has_negation(src) and not has_negation(s):
                    res.warnings.append(f"source sentence for {si} is negated but the answer is not: {src[:60]!r}")
    if question is not None:
        rel = question_relevance(question, answer)
        cited_docs = {c.split("#")[0] for c in CITATION.findall(answer)}
        named = {f"eip-{n}" for n in mentioned_numbers(question)} | {f"erc-{n}" for n in mentioned_numbers(question)}
        res.relevance = rel
        if rel < min_relevance and not (cited_docs & named):
            res.ok = False
            res.issues.append(f"answer relevance {rel:.2f} < {min_relevance:.2f}: shares too few terms with the question")
    return res
