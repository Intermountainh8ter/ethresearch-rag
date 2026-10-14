"""Mining of ``NAME = value`` constant definitions from chunk text.

Used by the offline extractive answerer (``pipeline.extractive_answer``) and by the evaluation
set builder (``evalgen``). Only unambiguous definitions are returned: the name is an
UPPER_SNAKE identifier, the value a plain integer, hex literal or ``2**n`` (optionally wrapped
in a type constructor such as ``uint64(2**5)``), and a (document, name) pair that has more than
one value inside a document is dropped.
"""
from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from .ingest import Chunk

NAME_RE = re.compile(r"^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+$")
HEX = r"0x[0-9a-fA-F]{1,64}"
DEC = r"\d[\d,_]*"
POW = r"2\*\*\d+"
VALUE_RE = re.compile(rf"^(?:{HEX}|{DEC}|{POW})$")
_WRAPPED = re.compile(rf"^[A-Za-z][A-Za-z0-9]*\(\s*({HEX}|{DEC}|{POW})\s*\)(?:\s*\(=\s*[\d,_]+\))?$")

_ROW = re.compile(r"^\|\s*`?([A-Z][A-Z0-9_]+)`?\s*\|\s*([^|\n]+?)\s*(?=\|)", re.MULTILINE)
_CODE = re.compile(
    rf"^[ \t]*([A-Z][A-Z0-9_]+)(?:[ \t]*:[ \t]*[A-Za-z0-9_\[\], ]+)?[ \t]*=[ \t]*({HEX}|{DEC}|2\*\*\d+)"
    r"[ \t]*(?:#.*)?$", re.MULTILINE)
_PROSE = re.compile(rf"`([A-Z][A-Z0-9_]+)`[ \t]*(?:=|:|is|of)[ \t]*`?({HEX}|{DEC})`?(?![\w.])")


@dataclass(frozen=True)
class Constant:
    doc: str
    chunk_id: str
    name: str
    value: str  # normalised to ``2**n`` / hex / decimal as written in the source
    needle: str  # verbatim substring of the chunk text containing both name and value
    kind: str  # row | code | prose


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
        values[(k.doc, k.name)].add(canon(k.value))
    seen: set[tuple[str, str]] = set()
    out: list[Constant] = []
    for k in found:
        key = (k.doc, k.name)
        if len(values[key]) != 1 or key in seen:
            continue  # ambiguous inside the document, or already taken
        seen.add(key)
        out.append(k)
    return out


def canon(value: str) -> str:
    return re.sub(r"[\s,_\\]", "", value).lower()
