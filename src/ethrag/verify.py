"""Deterministic grounding check for cited answers, with per-claim provenance.

An answer passes only if

1. every sentence carries at least one citation ``[doc#NNN]``;
2. every citation points at a chunk that was actually retrieved;
3. every *checkable claim token* in the sentence (numbers, hex values, ``2**n`` expressions,
   ALL_CAPS identifiers, numbers with units such as ``32 ETH``) appears as a token in the cited
   chunks, either verbatim or as a recorded equivalent (``0x48`` = ``72``, ``2**12`` = ``4096``,
   ``32 ETH`` = ``32000000000`` gwei, ``1,000,000`` = ``1_000_000``);
4. *binding*: when the sentence names an identifier, each number is found next to an identifier
   of that sentence inside a cited chunk, so ``X is 16`` is rejected when the chunk says
   ``X = 8`` and merely contains a 16 somewhere else;
5. *units and ranges*: a number written with a unit (``32 gwei``) must not contradict the unit
   the cited chunk gives for that number (``32 ETH``), and a range (``1-16``, ``0 to 0x7f``)
   must not contradict a range of the cited chunk that shares one endpoint;
6. (optional) a negated sentence ("does NOT burn") has a negated counterpart in the cited text;
7. (optional) enough of the sentence's content words appear in the cited chunks
   (``min_coverage``), so a paraphrase with no checkable token is not waved through;
8. (optional) the answer is about the question: it shares key terms with it (EIP/ERC reference
   numbers do not count) and repeats the question's identifiers (``min_relevance``).

Each check records provenance (``VerificationResult.claims``) naming the chunk and the source
line that supported each token. Substring matches never count (``8`` is not found inside
``58``). What the verifier cannot catch is documented in docs/EVALUATION.md and demonstrated by
``tests/test_verify.py::test_blind_spot_*``.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from .graph import mentioned_numbers
from .ingest import Chunk
from .retrieval import STOPWORDS, tokenize

CITATION = re.compile(r"\[([a-z][a-z0-9-]*#\d{3,})\]")
_NUMERIC = r"(?:\d{1,3}(?:,\d{3})+|\d+(?:_\d+)*)(?:\.\d+)?"
_NUM = re.compile(rf"0x[0-9a-fA-F]+|{_NUMERIC}(?:\s?(?:\*\*|\^)\s?\d+)?(?:[eE][+-]?\d+)?")
_IDENT = re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b|\b[A-Z]{3,}\b")
_STOP_CAPS = {"EIP", "ETH", "EVM", "THE", "AND", "NOT", "MUST", "SHALL", "SHOULD", "MAY", "RLP",
              "ERC", "JSON", "RPC", "API", "URL", "NOTE", "YES", "FOR"}
_NEGATION = re.compile(
    r"\b(?:not|never|cannot|no longer|neither|nor|without|disallow(?:s|ed)?|forbid(?:s|den)?|"
    r"prohibit(?:s|ed)?|unable)\b|\w+n't\b", re.IGNORECASE)
_WORD_STEM = re.compile(r"(ies|ied|ing|ed|es|s)$")

# unit word -> (dimension, size in the dimension's base unit). Numbers with a unit are compared
# in base units, so 32 ETH == 32000000000 gwei, 1 KiB == 1024 bytes, 2 minutes == 120 seconds.
UNITS: dict[str, tuple[str, int]] = {
    "wei": ("value", 1), "gwei": ("value", 10**9), "eth": ("value", 10**18), "ether": ("value", 10**18),
    "byte": ("data", 1), "bytes": ("data", 1), "kb": ("data", 1000), "kib": ("data", 1024),
    "mb": ("data", 10**6), "mib": ("data", 2**20), "gb": ("data", 10**9), "gib": ("data", 2**30),
    "second": ("time", 1), "seconds": ("time", 1), "minute": ("time", 60), "minutes": ("time", 60),
    "hour": ("time", 3600), "hours": ("time", 3600), "day": ("time", 86400), "days": ("time", 86400),
    "gas": ("gas", 1), "block": ("block", 1), "blocks": ("block", 1), "slot": ("slot", 1),
    "slots": ("slot", 1), "epoch": ("epoch", 1), "epochs": ("epoch", 1), "blob": ("blob", 1),
    "blobs": ("blob", 1), "bit": ("bit", 1), "bits": ("bit", 1),
}
_UNIT_CLAIM = re.compile(
    rf"(?<![\w.])(0x[0-9a-fA-F]+|{_NUMERIC}(?:\s?(?:\*\*|\^)\s?\d+)?)\s*("
    + "|".join(sorted(UNITS, key=len, reverse=True)) + r")\b", re.IGNORECASE)
_RNUM = rf"0x[0-9a-fA-F]+|{_NUMERIC}"
_RANGE = re.compile(
    rf"(?<![\w./^*-])({_RNUM})(?!\.\d|[\w^*/])\s*(?:-|–|—|\.\.\.?|\bto\b|\bthrough\b)\s*({_RNUM})(?![\w./^*-])"
    rf"|\bbetween\s+({_RNUM})\s+and\s+({_RNUM})\b", re.IGNORECASE)


_SCAN = re.compile(rf"(?P<ref>\b(?:EIP|ERC)[-\s]?\d{{1,5}}\b)|(?P<id>{_IDENT.pattern})|(?P<num>{_NUM.pattern})"
                   r"|(?P<word>[A-Za-z_]+)", re.IGNORECASE)
_BIND_WINDOW = 4  # an identifier up to this many items before a number "owns" the number


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


def int_value(tok: str) -> int | None:
    """Integer value of a normalised numeric token (hex, decimal, ``a**b``, ``NeM``), else None."""
    try:
        if tok.startswith("0x"):
            return int(tok, 16)
        if "**" in tok:
            base, exp = tok.split("**", 1)
            return int(base) ** int(exp) if base.isdigit() and exp.isdigit() and int(exp) <= 512 else None
        m = re.fullmatch(r"(\d+)e(\d{1,3})", tok)
        if m:
            return int(m.group(1)) * 10 ** int(m.group(2))
        return int(tok) if tok.isdigit() else None
    except ValueError:  # pragma: no cover - guarded above, kept as a safety net
        return None


def number_variants(tok: str) -> set[str]:
    """Equivalent spellings of a normalised numeric token (hex <-> decimal, 2**n <-> value)."""
    out = {tok}
    v = int_value(tok)
    if v is None:
        return out
    out.add(str(v))
    if v >= 16:
        out.add(_norm(hex(v)))
    if v > 1 and v & (v - 1) == 0:
        out.add(f"2**{v.bit_length() - 1}")
    return out


@dataclass(frozen=True)
class UnitClaim:
    literal: str  # normalised number as written ("32")
    unit: str  # lower-case unit word ("eth")
    dimension: str  # "value" | "data" | "time" | ...
    base: int  # value in the dimension's base unit
    text: str  # the phrase as written ("32 ETH")


def unit_claims(text: str) -> list[UnitClaim]:
    """Integer quantities written with a known unit (``32 ETH``, ``12 seconds``, ``1 KiB``)."""
    out = []
    for m in _UNIT_CLAIM.finditer(CITATION.sub("", text)):
        lit = _norm(m.group(1))
        v = int_value(lit)
        if v is None:
            continue
        unit = m.group(2).lower()
        dim, scale = UNITS[unit]
        out.append(UnitClaim(lit, unit, dim, v * scale, m.group(0)))
    return out


def unit_variants(text: str) -> dict[str, set[str]]:
    """``{"32": {"32000000000", "32000000000000000000"}}`` for ``32 ETH`` style phrases.

    The alternatives are the same quantity spelled in the dimension's base unit (and, for
    values, in gwei), which is how specs often write it.
    """
    out: dict[str, set[str]] = {}
    for c in unit_claims(text):
        alts = {str(c.base)}
        if c.dimension == "value" and c.base % 10**9 == 0:
            alts.add(str(c.base // 10**9))
        out.setdefault(c.literal, set()).update(alts - {c.literal})
    return out


def range_claims(text: str) -> list[tuple[str, str]]:
    """Ranges written as ``A-B``, ``A to B``, ``A..B`` or ``between A and B`` (normalised ends)."""
    out = []
    for m in _RANGE.finditer(CITATION.sub("", text)):
        lo, hi = (m.group(1), m.group(2)) if m.group(1) is not None else (m.group(3), m.group(4))
        out.append((_norm(lo), _norm(hi)))
    return out


def _split_claims(text: str) -> tuple[set[str], set[str]]:
    """(numeric tokens, identifier tokens) of ``text`` with citations removed."""
    text = CITATION.sub("", text)
    nums = {_norm(m) for m in _NUM.findall(text)}
    idents = {_norm_ident(m) for m in _IDENT.findall(text) if m not in _STOP_CAPS}
    return {t for t in nums if t}, {t for t in idents if t}


def bound_pairs(text: str) -> dict[str, set[str]]:
    """``{number: {identifiers just before it}}`` - the (name, value) pairs a sentence asserts.

    ``BLOB_TX_TYPE is 0x03`` yields ``{"0x3": {"blob_tx_type"}}``. A number with no identifier
    in the preceding few words ("access list transactions are type 0x01") is not listed, so
    sentences that mix several subjects are not forced to bind every number to one name.
    """
    items = []
    for m in _SCAN.finditer(CITATION.sub("", text)):
        kind, tok = m.lastgroup, m.group()
        if kind == "id" and (tok in _STOP_CAPS or not _IDENT.fullmatch(tok)):
            kind = "word"  # IGNORECASE lets "the"/"api" through the identifier branch: keep real ones only
        items.append((kind, tok))
    out: dict[str, set[str]] = {}
    for j, (kind, tok) in enumerate(items):
        if kind != "num":
            continue
        for prev_kind, prev in reversed(items[max(0, j - _BIND_WINDOW): j]):
            if prev_kind == "id":
                out.setdefault(_norm(tok), set()).add(_norm_ident(prev))
                break
    return out


def claim_tokens(text: str) -> set[str]:
    nums, idents = _split_claims(text)
    return nums | idents


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


def _best_match(target: set[str], pieces: Sequence[str]) -> tuple[str, float]:
    best, best_score = "", 0.0
    for piece in pieces:
        score = len(target & content_words(piece)) / len(target)
        if score > best_score:
            best, best_score = piece, score
    return best, best_score


def best_source_sentence(sentence: str, chunk_texts: Sequence[str]) -> tuple[str, float]:
    """Source sentence with the largest content-word recall of ``sentence``.

    Prose sentences are tried first; code lines and table rows are only used when no prose
    sentence covers half of the words (a negated claim about ``NAME = 0x01`` has no prose).
    """
    target = content_words(sentence)
    if not target:
        return "", 0.0
    prose, structured = [], []
    for text in chunk_texts:
        prose += sentences(re.sub(r"```.*?```", " ", text, flags=re.DOTALL))
        for block in re.findall(r"```.*?```", text, flags=re.DOTALL):
            structured += [ln.strip() for ln in block.splitlines() if ln.strip() and not ln.startswith("```")]
    structured += [p for p in prose if p.lstrip().startswith("|")]
    best, score = _best_match(target, prose)
    if score < 0.5:
        alt, alt_score = _best_match(target, structured)
        if alt_score > score:
            return alt, alt_score
    return best, score


# Words that ask *how* to answer, not *what about*: "What does ERC-3009 propose?" has no topic
# beyond the document it names.
_META_WORDS = (
    "propose", "proposes", "proposed", "describe", "describes", "described", "introduce", "introduces",
    "introduced", "define", "defines", "defined", "change", "changes", "feature", "purpose", "explain",
    "explains", "mean", "means", "meaning", "work", "works", "happen", "happens", "value", "values",
    "number", "set", "sets", "used", "use", "uses", "using", "specify", "specifies", "specified",
    "following", "exact", "exactly", "kind",
)
QUESTION_META = frozenset(stem(w) for w in _META_WORDS)


def question_terms(question: str) -> set[str]:
    """Topical words of the question: no EIP/ERC reference numbers and no "what/propose/describe" words."""
    refs = {str(n) for n in mentioned_numbers(question)} | {"eip", "erc"}
    return {w for w in content_words(question) if w not in refs and w not in QUESTION_META}


def answer_novelty(question: str, answer: str) -> float:
    """Share of the answer's words and numbers that the question did not already contain."""
    a_words, q_words = content_words(answer), content_words(question)
    a_nums, q_nums = _split_claims(answer)[0], _split_claims(question)[0]
    total = len(a_words) + len(a_nums)
    return (len(a_words - q_words) + len(a_nums - q_nums)) / total if total else 1.0


def question_relevance(question: str, answer: str) -> float:
    """Fraction of the question's topical words that the answer repeats (stemmed)."""
    q = question_terms(question)
    if not q:
        return 1.0
    return len(q & content_words(answer)) / len(q)


def question_identifiers(question: str) -> set[str]:
    return _split_claims(question)[1]


@dataclass
class ClaimCheck:
    sentence: int
    token: str
    supported: bool
    chunk: str = ""
    kind: str = ""  # exact | hex-dec | pow2 | unit
    snippet: str = ""  # the source line that carries the token
    bound: bool | None = None  # None = not checked (no identifier in the sentence)


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

    def explain(self) -> str:
        """Human-readable per-claim table: verdict, token, supporting chunk and source line."""
        lines = [f"verdict: {'VERIFIED' if self.ok else 'WITHHELD'}"]
        for c in self.claims:
            mark = "ok  " if c.supported else "FAIL"
            where = f"{c.chunk} ({c.kind})" if c.supported else "not found in the cited chunks"
            bind = "" if c.bound is None else (" bound" if c.bound else " UNBOUND")
            snip = f"  | {c.snippet}" if c.snippet else ""
            lines.append(f"  [{mark}] s{c.sentence} {c.token:<28} <- {where}{bind}{snip}")
        lines += [f"  issue: {i}" for i in self.issues] + [f"  warning: {w}" for w in self.warnings]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok, "issues": self.issues, "warnings": self.warnings,
            "checked_tokens": self.checked_tokens, "supported_tokens": self.supported_tokens,
            "support_ratio": self.support_ratio, "coverage": self.coverage,
            "relevance": self.relevance,
            "claims": [c.__dict__ for c in self.claims],
        }


class _ChunkView:
    """Lazily computed token sets, content words and per-line tokens of the cited chunks."""

    def __init__(self, chunks: Mapping[str, Chunk]):
        self.chunks = chunks
        self._tokens: dict[str, set[str]] = {}
        self._words: dict[str, set[str]] = {}
        self._lines: dict[str, list[tuple[str, set[str], set[str]]]] = {}
        self._units: dict[str, list[UnitClaim]] = {}
        self._ranges: dict[str, list[tuple[str, str]]] = {}

    def tokens(self, cid: str) -> set[str]:
        if cid not in self._tokens:
            self._tokens[cid] = claim_tokens(self.chunks[cid].text)
        return self._tokens[cid]

    def words(self, cid: str) -> set[str]:
        if cid not in self._words:
            self._words[cid] = content_words(self.chunks[cid].text)
        return self._words[cid]

    def lines(self, cid: str) -> list[tuple[str, set[str], set[str]]]:
        """Non-empty lines as (text, numeric tokens, identifier tokens)."""
        if cid not in self._lines:
            self._lines[cid] = [(ln, *_split_claims(ln)) for ln in self.chunks[cid].text.splitlines()
                                if ln.strip()]
        return self._lines[cid]

    def units(self, cid: str) -> list[UnitClaim]:
        if cid not in self._units:
            self._units[cid] = unit_claims(self.chunks[cid].text)
        return self._units[cid]

    def ranges(self, cid: str) -> list[tuple[str, str]]:
        if cid not in self._ranges:
            self._ranges[cid] = range_claims(self.chunks[cid].text)
        return self._ranges[cid]

    def snippet(self, cid: str, spellings: frozenset | set, phrase: str = "") -> str:
        for ln, nums, ids in self.lines(cid):
            if (phrase and phrase in ln) or (nums | ids) & spellings:
                return " ".join(ln.split())[:160]
        return ""

    def bound(self, cid: str, spellings: set[str], idents: set[str]) -> bool:
        """Does a spelling of the number sit on a line that names one of ``idents``?

        Same line wins. A line that carries *no* identifier at all ("Opcode: `0xfa`.") may borrow
        the identifier of the line just before or after it; a line that names another identifier
        (the next row of a constants table) may not, which is what stops ``X = 8`` / ``Y = 16``
        from vouching for "X is 16".
        """
        lines = self.lines(cid)
        for i, (_, nums, ids) in enumerate(lines):
            if not nums & spellings:
                continue
            if ids & idents:
                return True
            if not ids:
                near = [lines[j][2] for j in (i - 1, i + 1) if 0 <= j < len(lines)]
                if any(n & idents for n in near):
                    return True
        return False


@dataclass(frozen=True)
class _Support:
    chunk: str = ""
    kind: str = ""
    spelled: frozenset = frozenset()  # tokens of the chunk that carry the claim
    phrase: str = ""  # text of the chunk that carries it, when it is a unit phrase

    def __bool__(self) -> bool:
        return bool(self.chunk)


def _find_support(tok: str, cited: Sequence[str], view: _ChunkView,
                  claims: Sequence[UnitClaim]) -> _Support:
    """First cited chunk that supports ``tok``: verbatim, as an equivalent spelling, or as the
    same quantity in another unit (``1024 bytes`` vs ``1 KiB``, ``32 ETH`` vs ``32000000000``)."""
    for cid in cited:
        if tok in view.tokens(cid):
            return _Support(cid, "exact", frozenset({tok}))
    variants = number_variants(tok) - {tok}
    for cid in cited:
        hit = {v for v in variants if v in view.tokens(cid)}
        if hit:
            kind = "hex-dec" if (tok.startswith("0x") or any(v.startswith("0x") for v in hit)) else "pow2"
            return _Support(cid, kind, frozenset(hit))
    mine = [c for c in claims if c.literal == tok]
    for claim in mine:
        for cid in cited:
            same = [u for u in view.units(cid) if u.dimension == claim.dimension and u.base == claim.base]
            if same:
                return _Support(cid, "unit", frozenset(u.literal for u in same), same[0].text)
    for claim in mine:
        alts = {str(claim.base)}
        if claim.dimension == "value" and claim.base % 10**9 == 0:
            alts.add(str(claim.base // 10**9))  # gwei, how consensus specs write balances
        for cid in cited:
            hit = {v for v in alts - {tok} if v in view.tokens(cid)}
            if hit:
                return _Support(cid, "unit", frozenset(hit))
    return _Support()


def _unit_issues(sentence: str, cited: Sequence[str], view: _ChunkView) -> list[str]:
    issues = []
    for claim in unit_claims(sentence):
        pool = [u for cid in cited for u in view.units(cid)]
        if any(u.dimension == claim.dimension and u.base == claim.base for u in pool):
            continue  # same quantity, possibly written in another unit
        clash = next((u for u in pool if u.literal == claim.literal), None)
        if clash is not None:
            issues.append(f"unit mismatch: answer says {claim.text!r} but the cited chunk says {clash.text!r}")
    return issues


def _same_number(a: str, b: str) -> bool:
    va, vb = int_value(a), int_value(b)
    return a == b or (va is not None and va == vb)


def _range_issues(sentence: str, cited: Sequence[str], view: _ChunkView) -> list[str]:
    issues = []
    pool = [r for cid in cited for r in view.ranges(cid)]
    for lo, hi in range_claims(sentence):
        lo_v, hi_v = int_value(lo), int_value(hi)
        if lo_v is not None and hi_v is not None and lo_v > hi_v:
            issues.append(f"malformed range: {lo} to {hi} runs backwards")
            continue
        if any(_same_number(lo, clo) and _same_number(hi, chi) for clo, chi in pool):
            continue
        close = next(((clo, chi) for clo, chi in pool if _same_number(lo, clo) or _same_number(hi, chi)), None)
        if close is not None:
            issues.append(f"range mismatch: answer says {lo} to {hi} but the cited chunk says "
                          f"{close[0]} to {close[1]}")
    return issues


def verify_answer(answer: str, chunks: Mapping[str, Chunk], retrieved_ids: Sequence[str],
                  question: str | None = None, min_coverage: float = 0.0,
                  min_relevance: float = 0.0, check_negation: bool = True,
                  check_units: bool = True, check_binding: bool = True,
                  strict_negation: bool = False, min_novelty: float = 0.0) -> VerificationResult:
    res = VerificationResult(ok=True)
    retrieved = set(retrieved_ids)
    view = _ChunkView(chunks)

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
        body = CITATION.sub("", s)
        claims = unit_claims(body)
        nums, idents = _split_claims(s)
        owners = bound_pairs(body) if check_binding else {}
        for tok in sorted(nums | idents):
            res.checked_tokens += 1
            sup = _find_support(tok, valid, view, claims)
            check = ClaimCheck(si, tok, bool(sup), sup.chunk, sup.kind)
            res.claims.append(check)
            if not sup:
                res.ok = False
                res.issues.append(f"unsupported claim {tok!r} not found in cited chunks {cites}")
                continue
            res.supported_tokens += 1
            check.snippet = view.snippet(sup.chunk, sup.spelled, sup.phrase)
            if tok in owners:
                spelled = set(sup.spelled) | {tok}
                check.bound = all(any(view.bound(c, spelled, {owner}) for c in valid) for owner in owners[tok])
                if not check.bound:
                    res.ok = False
                    res.issues.append(
                        f"unbound claim {tok!r}: found in the cited chunks but never next to "
                        f"{sorted(owners[tok])}")
        if valid and check_units:
            for issue in _unit_issues(body, valid, view) + _range_issues(body, valid, view):
                res.ok = False
                res.issues.append(issue)
        words = content_words(s)
        support_words: set[str] = set().union(*(view.words(c) for c in valid)) if valid else set()
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
                    msg = f"source sentence for {si} is negated but the answer is not: {src[:60]!r}"
                    if strict_negation and score >= 0.7:
                        res.ok = False
                        res.issues.append(f"negation dropped in answer sentence {si}: {src[:60]!r}")
                    else:
                        res.warnings.append(msg)
    if question is not None:
        _check_relevance(res, question, answer, min_relevance, min_novelty)
    return res


def _check_relevance(res: VerificationResult, question: str, answer: str, min_relevance: float,
                     min_novelty: float) -> None:
    rel = question_relevance(question, answer)
    res.relevance = rel
    if rel < min_relevance:
        res.ok = False
        res.issues.append(f"answer relevance {rel:.2f} < {min_relevance:.2f}: shares too few terms with the question")
    if min_novelty > 0:
        nov = answer_novelty(question, answer)
        if nov < min_novelty:
            res.ok = False
            res.issues.append(f"answer novelty {nov:.2f} < {min_novelty:.2f}: it mostly restates the question")
    if min_relevance > 0:
        # A question about an UPPER_SNAKE parameter wants that parameter named AND a value given.
        wanted = {t for t in question_identifiers(question) if "_" in t}
        if wanted:
            if not wanted & claim_tokens(answer):
                res.ok = False
                res.issues.append(f"answer does not mention {sorted(wanted)} from the question")
            elif not _split_claims(answer)[0] - _split_claims(question)[0]:
                res.ok = False
                res.issues.append(f"answer states no value for {sorted(wanted)}: it only restates the question")
    named = {f"{kind}-{n}" for n in mentioned_numbers(question) for kind in ("eip", "erc")}
    cited_docs = {c.split("#")[0] for c in CITATION.findall(answer)}
    if named and cited_docs and not named & cited_docs:
        msg = f"the question names {sorted(named)} but the answer cites {sorted(cited_docs)}"
        if min_relevance > 0 and not question_terms(question):
            # nothing but the document reference says what the question is about
            res.ok = False
            res.issues.append(msg)
        else:
            res.warnings.append(msg)
