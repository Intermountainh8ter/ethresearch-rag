"""Evaluation-set generation and validation (offline, deterministic).

Candidates are mined from the *real chunk text* of the bundled corpus, never from model memory:

* ``mine_constants`` finds ``NAME = value`` definitions in markdown table rows, code lines and
  short prose ("`CHAIN_ID`: 1") whose value is a plain integer, hex literal or ``2**n``;
* ``perturb`` produces plausible-but-different values in the same notation for the *wrong*
  answers (``8 -> 16``, ``0x03 -> 0x04``, ``2**17 -> 2**18``);
* ``validate_question`` re-checks every record with the production verifier: the gold needle
  must occur in a chunk of the gold document, the correct answer must be supported by the
  cited gold chunk(s), and the wrong answer must NOT be supported;
* ``assign_splits`` stores a seeded train/test split in each record. Single-hop questions are
  split *by gold document* so a reranker fit on train never sees a test document.

``evals/build_questions.py`` ties these together; ``tests/test_questions.py`` re-validates the
committed ``evals/questions.jsonl`` on every run.
"""
from __future__ import annotations

import random
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from .graph import CitationGraph
from .ingest import Chunk
from .verify import claim_tokens, verify_answer

NAME_RE = re.compile(r"^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+$")
_HEX = r"0x[0-9a-fA-F]{1,64}"
_DEC = r"\d[\d,_]*"
_POW = r"2\*\*\d+"
VALUE_RE = re.compile(rf"^(?:{_HEX}|{_DEC}|{_POW})$")
_WRAPPED = re.compile(rf"^[A-Za-z][A-Za-z0-9]*\(\s*({_HEX}|{_DEC}|{_POW})\s*\)(?:\s*\(=\s*[\d,_]+\))?$")

_ROW = re.compile(r"^\|\s*`?([A-Z][A-Z0-9_]+)`?\s*\|\s*([^|\n]+?)\s*(?=\|)", re.MULTILINE)
_CODE = re.compile(
    rf"^[ \t]*([A-Z][A-Z0-9_]+)(?:[ \t]*:[ \t]*[A-Za-z0-9_\[\], ]+)?[ \t]*=[ \t]*({_HEX}|{_DEC}|2\*\*\d+)"
    r"[ \t]*(?:#.*)?$", re.MULTILINE)
_PROSE = re.compile(rf"`([A-Z][A-Z0-9_]+)`[ \t]*(?:=|:|is|of)[ \t]*`?({_HEX}|{_DEC})`?(?![\w.])")

TYPES = ("constant", "concept", "multihop")


@dataclass(frozen=True)
class Constant:
    doc: str
    chunk_id: str
    name: str
    value: str  # normalised to ``2**n`` / hex / decimal as written in the source
    needle: str  # verbatim substring of the chunk text containing both name and value
    kind: str  # row | code | prose


def doc_label(chunk: Chunk) -> str:
    """"EIP-1559" / "ERC-721" for numbered documents, otherwise the document title."""
    if chunk.number is not None:
        return f"{chunk.doc.split('-')[0].upper()}-{chunk.number}"
    if chunk.doc.startswith("cl-"):
        return f"the consensus spec '{chunk.title}'"
    if chunk.doc.startswith("fb-"):
        return f"the Flashbots doc '{chunk.title}'"
    return chunk.title


def _clean_value(cell: str) -> str | None:
    cell = cell.replace("`", "").replace("\\*", "*").strip()
    m = _WRAPPED.match(cell)
    if m:
        cell = m.group(1)
    cell = re.sub(r"\s+", "", cell) if "**" in cell else cell
    return cell if VALUE_RE.match(cell) else None


def _small(value: str) -> bool:
    return value.isdigit() and int(value) < 2


def mine_constants(chunks: Iterable[Chunk]) -> list[Constant]:
    """Unambiguous ``NAME = value`` definitions (one value per (doc, NAME)) in chunk text."""
    found: list[Constant] = []
    for c in chunks:
        text = c.text
        for m in _ROW.finditer(text):
            value = _clean_value(m.group(2))
            if value and NAME_RE.match(m.group(1)) and not _small(value):
                start = m.start(1) - (1 if m.start(1) > 0 and text[m.start(1) - 1] == "`" else 0)
                found.append(Constant(c.doc, c.id, m.group(1), value, text[start:m.end(2)], "row"))
        for m in _CODE.finditer(text):
            value = _clean_value(m.group(2))
            if value and NAME_RE.match(m.group(1)) and not _small(value):
                found.append(Constant(c.doc, c.id, m.group(1), value, m.group(0).strip(), "code"))
        for m in _PROSE.finditer(text):
            value = _clean_value(m.group(2))
            if value and NAME_RE.match(m.group(1)) and not _small(value):
                found.append(Constant(c.doc, c.id, m.group(1), value, m.group(0), "prose"))
    values: dict[tuple[str, str], set[str]] = defaultdict(set)
    for k in found:
        values[(k.doc, k.name)].add(_canon(k.value))
    seen: set[tuple[str, str]] = set()
    out: list[Constant] = []
    for k in found:
        key = (k.doc, k.name)
        if len(values[key]) != 1 or key in seen:
            continue  # ambiguous inside the document, or already taken
        seen.add(key)
        out.append(k)
    return out


def _canon(value: str) -> str:
    return re.sub(r"[\s,_\\]", "", value).lower()


# ----------------------------------------------------------------------------- perturbations
def perturb(value: str) -> list[str]:
    """Plausible wrong values in the same notation, most plausible first (never equal to input)."""
    v = value.strip()
    out: list[str] = []
    m = re.fullmatch(r"2\*\*(\d+)", v)
    if m:
        n = int(m.group(1))
        out = [f"2**{k}" for k in (n + 1, n - 1, n + 2, n - 2, n + 3) if k >= 0 and k != n]
    elif re.fullmatch(_HEX, v):
        digits = v[2:]
        n = int(digits, 16)
        width = len(digits)
        fmt = "X" if digits.upper() == digits and not digits.isdigit() else "x"
        cands = [n + 1, n - 1, n + 2, n ^ 0x10, n * 2, n + 3] if n < 2**32 else [n ^ 1, n ^ 0x10, n ^ 0x100]
        out = [f"0x{format(x, f'0{width}{fmt}')}" for x in cands
               if x >= 0 and x != n and len(format(x, 'x')) <= width]
    elif re.fullmatch(_DEC, v):
        sep = "," if "," in v else ("_" if "_" in v else "")
        n = int(re.sub(r"[,_]", "", v))
        for x in (n * 2, n // 2, n + 1, n - 1, n * 4, n // 4, n * 10, n // 10, n + 2, n + 10):
            if x >= 0 and x != n:
                out.append(f"{x:,}".replace(",", sep) if sep else str(x))
    seen: set[str] = set()
    uniq = []
    for x in out:
        if _canon(x) != _canon(v) and x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def supported(answer: str, cites: Sequence[str], by_id: dict[str, Chunk]) -> bool:
    """Would the token-level verifier accept ``answer`` when it cites ``cites``?"""
    refs = " ".join(f"[{c}]" for c in cites)
    return verify_answer(f"{answer} {refs}.", by_id, list(cites), min_coverage=0.0,
                         min_relevance=0.0, check_negation=False).ok


def replace_last(text: str, old: str, new: str) -> str:
    i = text.rfind(old)
    return text if i < 0 else text[:i] + new + text[i + len(old):]


def pick_wrong(answer: str, old: str, cites: Sequence[str], by_id: dict[str, Chunk]) -> str | None:
    """First perturbation of ``old`` (the last value inside ``answer``) the verifier rejects."""
    for cand in perturb(old):
        wrong = replace_last(answer, old, cand)
        if wrong != answer and not supported(wrong, cites, by_id):
            return wrong
    return None


_TOKEN_SPAN = re.compile(rf"{_HEX}|2\*\*\d+|{_DEC}")


_REF_BEFORE = re.compile(r"(?:EIP|ERC)[-\s]?$", re.IGNORECASE)


def value_spans(text: str) -> list[tuple[int, int]]:
    """Spans of numeric / hex / 2**n literals in ``text`` (EIP/ERC reference numbers excluded)."""
    spans = []
    for m in _TOKEN_SPAN.finditer(text):
        if _REF_BEFORE.search(text[:m.start()][-5:]) or text[m.start() - 1:m.start()].isalpha():
            continue  # "EIP-1193" is a reference, "uint256"/"bytes4" part of a word
        end = m.end()
        while end > m.start() and text[end - 1] in ",_":
            end -= 1
        spans.append((m.start(), end))
    return spans


def pick_wrong_at(answer: str, span: tuple[int, int], cites: Sequence[str],
                  by_id: dict[str, Chunk]) -> str | None:
    """Replace the literal at ``span`` by the first perturbation the verifier rejects."""
    a, b = span
    for cand in perturb(answer[a:b]):
        wrong = answer[:a] + cand + answer[b:]
        if not supported(wrong, cites, by_id):
            return wrong
    return None


def make_wrong(answer: str, cites: Sequence[str], by_id: dict[str, Chunk],
               prefer_last: bool = True) -> str | None:
    """Wrong variant of ``answer``: perturb one literal, trying later ones first."""
    spans = value_spans(answer)
    for span in (reversed(spans) if prefer_last else spans):
        wrong = pick_wrong_at(answer, span, cites, by_id)
        if wrong:
            return wrong
    return None


def clean_sentence(s: str) -> str:
    """Plain-text version of a markdown sentence (links, emphasis, bullets, code ticks removed)."""
    s = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", s)
    s = re.sub(r"^[\-\*\+]\s+", "", s.strip())
    s = s.replace("`", "").replace("*", "")
    return re.sub(r"\s+", " ", s).strip().rstrip(".:")


# ------------------------------------------------------------------------------ question text
TEMPLATES_DOC = (
    "What is the value of {name} in {label}?",
    "{label} defines {name}. What value does it have?",
    "In {label}, what is {name} set to?",
    "What does {label} set the constant {name} to?",
)
TEMPLATES_NAME_ONLY = (
    "What value does the constant {name} take?",
    "What is {name} defined as?",
)
TEMPLATES_WORDS = (
    "In {label}, what value does the {words} parameter have?",
    "What is the {words} in {label}?",
)


def constant_question(k: Constant, label: str, name_is_unique: bool, rng: random.Random) -> tuple[str, str]:
    """(question text, tier) for a mined constant; tier is exact-id | name-only | words."""
    roll = rng.random()
    if name_is_unique and roll < 0.25:
        return rng.choice(TEMPLATES_NAME_ONLY).format(name=k.name), "name-only"
    if roll < 0.6:
        return rng.choice(TEMPLATES_WORDS).format(label=label, words=k.name.lower().replace("_", " ")), "words"
    return rng.choice(TEMPLATES_DOC).format(label=label, name=k.name), "exact-id"


# ------------------------------------------------------------------------------- validation
def gold_groups(q: dict, by_doc: dict[str, list[Chunk]]) -> list[list[Chunk]]:
    """For each (doc, needle) gold pair, the chunks of that doc containing the needle."""
    return [[c for c in by_doc.get(doc, []) if needle.lower() in c.text.lower()] for doc, needle in q["gold"]]


def validate_question(q: dict, by_doc: dict[str, list[Chunk]], by_id: dict[str, Chunk]) -> list[str]:
    """Problems with one normalised question record (empty list = valid)."""
    problems: list[str] = []
    if q.get("type") not in TYPES:
        problems.append(f"bad type {q.get('type')!r}")
    if not q.get("q", "").strip().endswith("?"):
        problems.append("question text must end with '?'")
    groups = gold_groups(q, by_doc)
    for (doc, needle), g in zip(q["gold"], groups, strict=True):
        if not g:
            problems.append(f"gold needle {needle!r} not found in any chunk of {doc}")
    if problems:
        return problems
    answer, wrong = q.get("answer", ""), q.get("wrong", "")
    if answer:
        cites = best_cites(answer, groups, by_id)
        if cites is None:
            problems.append("correct answer is not supported by the gold chunk(s)")
        else:
            if wrong:
                if wrong == answer:
                    problems.append("wrong answer equals the correct answer")
                elif supported(wrong, cites, by_id):
                    problems.append("wrong answer is supported by the gold chunk(s)")
            tokens = claim_tokens(answer)
            if q["type"] != "concept" and tokens and not tokens - claim_tokens(q["q"]):
                problems.append("question already contains every claim token of the answer")
    elif wrong:
        problems.append("wrong answer without a correct answer")
    return problems


def best_cites(answer: str, groups: list[list[Chunk]], by_id: dict[str, Chunk]) -> list[str] | None:
    """Gold chunks to cite so the verifier accepts ``answer``; ``None`` if no choice works."""
    if len(groups) == 1:
        for c in groups[0]:
            if supported(answer, [c.id], by_id):
                return [c.id]
        return None
    cites = [c.id for g in groups for c in g[:4]]
    return cites if supported(answer, cites, by_id) else None


def split_index(chunks: Iterable[Chunk]) -> tuple[dict[str, list[Chunk]], dict[str, Chunk]]:
    by_doc: dict[str, list[Chunk]] = defaultdict(list)
    by_id: dict[str, Chunk] = {}
    for c in chunks:
        by_doc[c.doc].append(c)
        by_id[c.id] = c
    return dict(by_doc), by_id


# ------------------------------------------------------------------------------------ splits
def assign_splits(questions: list[dict], seed: int = 20240613, test_fraction: float = 0.3) -> list[dict]:
    """Write ``split`` ("train" | "test") into every record, deterministically.

    Single-hop questions are assigned by *gold document* (greedy over a seeded shuffle) so a
    document never straddles the split; multi-hop questions are assigned individually because
    they span two documents. Returns the same list for chaining.
    """
    rng = random.Random(seed)
    singles = [q for q in questions if q["type"] != "multihop"]
    multi = [q for q in questions if q["type"] == "multihop"]
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for q in singles:
        by_doc[q["gold_doc"]].append(q)
    docs = sorted(by_doc)
    rng.shuffle(docs)
    target = round(len(singles) * test_fraction)
    taken = 0
    for d in docs:
        side = "test" if taken < target else "train"
        for q in by_doc[d]:
            q["split"] = side
        if side == "test":
            taken += len(by_doc[d])
    ordered = sorted(multi, key=lambda q: q["id"])
    rng.shuffle(ordered)
    n_test = round(len(ordered) * test_fraction)
    for i, q in enumerate(ordered):
        q["split"] = "test" if i < n_test else "train"
    return questions


# -------------------------------------------------------------------------------- multi-hop
def multihop_pairs(consts: Sequence[Constant], graph: CitationGraph, rng: random.Random,
                   limit: int, max_uses: int = 2) -> list[tuple[Constant, Constant, str]]:
    """Pairs of constants from two different documents linked in the citation graph.

    ``relation`` is "requires" when A requires B, "mentions" when A's text names B. A document
    appears in at most ``max_uses`` pairs and a constant in at most one.
    """
    by_doc: dict[str, list[Constant]] = defaultdict(list)
    for k in consts:
        by_doc[k.doc].append(k)
    edges: list[tuple[str, str, str]] = []
    for a in sorted(by_doc):
        edges += [(a, b, "requires") for b in sorted(graph.requires.get(a, ())) if b in by_doc]
        edges += [(a, b, "mentions") for b in sorted(graph.mentions.get(a, ()))
                  if b in by_doc and b not in graph.requires.get(a, ())]
    rng.shuffle(edges)
    out: list[tuple[Constant, Constant, str]] = []
    uses: dict[str, int] = defaultdict(int)
    taken: set[tuple[str, str]] = set()
    for a, b, rel in edges:
        if uses[a] >= max_uses or uses[b] >= max_uses:
            continue
        fresh_a = [k for k in by_doc[a] if (k.doc, k.name) not in taken]
        fresh_b = [k for k in by_doc[b] if (k.doc, k.name) not in taken]
        if not fresh_a or not fresh_b:
            continue
        ka, kb = rng.choice(fresh_a), rng.choice(fresh_b)
        if ka.name == kb.name:
            continue
        out.append((ka, kb, rel))
        taken.update({(ka.doc, ka.name), (kb.doc, kb.name)})
        uses[a] += 1
        uses[b] += 1
        if len(out) >= limit:
            break
    return out


MULTI_TEMPLATES = (
    "What are the values of {na} in {la} and {nb} in {lb}?",
    "{la} defines {na} and {lb} defines {nb}. What are their values?",
    "Give {na} from {la} together with {nb} from {lb}. What are they?",
)
MULTI_TEMPLATES_ID = (
    "{la} defines {na}, and a related document defines {nb}. What are their values?",
    "What are {na} in {la} and the constant {nb}?",
)


def multihop_question(a: Constant, b: Constant, la: str, lb: str, rng: random.Random,
                      b_is_unique: bool = False) -> tuple[str, str]:
    """(question, tier): tier "both-named" or "second-by-id" (second document only via its constant)."""
    if b_is_unique and rng.random() < 0.5:
        return rng.choice(MULTI_TEMPLATES_ID).format(na=a.name, nb=b.name, la=la), "second-by-id"
    return rng.choice(MULTI_TEMPLATES).format(na=a.name, nb=b.name, la=la, lb=lb), "both-named"


def stratified_take(items: Sequence[Constant], limit: int, per_doc: int, rng: random.Random,
                    group_of: Callable[[Constant], str]) -> list[Constant]:
    """Take up to ``limit`` items, at most ``per_doc`` per document, round-robin over groups."""
    groups: dict[str, list[Constant]] = defaultdict(list)
    for k in items:
        groups[group_of(k)].append(k)
    for g in groups.values():
        rng.shuffle(g)
    per: dict[str, int] = defaultdict(int)
    out: list[Constant] = []
    names = sorted(groups)
    progress = True
    while len(out) < limit and progress:
        progress = False
        for name in names:
            g = groups[name]
            while g:
                k = g.pop()
                if per[k.doc] < per_doc:
                    per[k.doc] += 1
                    out.append(k)
                    progress = True
                    break
            if len(out) >= limit:
                break
    return out


# ----------------------------------------------------------------------------------- purpose
SUMMARY_SECTIONS = ("abstract", "simple summary", "summary")
TEMPLATES_PURPOSE = (
    "What does {label} propose?",
    "What is {label} ({title}) about?",
    "What change or feature does {label} describe?",
)


def first_summary_sentence(doc_chunks: Sequence[Chunk]) -> tuple[Chunk, str] | None:
    """(chunk, sentence) for the first plain-prose sentence of an Abstract/Summary section."""
    from .verify import sentences

    for c in doc_chunks:
        if c.section.strip().lower() not in SUMMARY_SECTIONS or c.kind != "prose":
            continue
        for s in sentences(c.body):
            if 8 <= len(s.split()) <= 45 and not s.lstrip().startswith(("#", "|", "<", "!", ">")):
                return c, s
    return None
