"""EIP markdown -> section-aware, overlapping chunks with stable ids."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    id: str  # e.g. "eip-1559#007"
    doc: str  # e.g. "eip-1559"
    title: str
    section: str
    text: str


def parse_front_matter(raw: str) -> tuple[dict[str, str], str]:
    m = re.match(r"^---\n(.*?)\n---\n", raw, re.DOTALL)
    if not m:
        return {}, raw
    meta = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta, raw[m.end():]


def split_sections(body: str) -> list[tuple[str, str]]:
    """Split on markdown headings, ignoring '#' lines inside fenced code blocks."""
    sections: list[tuple[str, list[str]]] = [("Preamble", [])]
    in_code = False
    for line in body.splitlines():
        if line.lstrip().startswith("```"):
            in_code = not in_code
        m = re.match(r"^(#{1,4})\s+(.*)", line)
        if m and not in_code:
            sections.append((m.group(2).strip(), []))
        else:
            sections[-1][1].append(line)
    return [(h, "\n".join(ls).strip()) for h, ls in sections if "\n".join(ls).strip()]


def window(text: str, max_words: int, overlap: int) -> list[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text]
    out, step = [], max_words - overlap
    for start in range(0, len(words), step):
        out.append(" ".join(words[start:start + max_words]))
        if start + max_words >= len(words):
            break
    return out


def chunk_document(doc_id: str, raw: str, max_words: int = 180, overlap: int = 30) -> list[Chunk]:
    meta, body = parse_front_matter(raw)
    title = meta.get("title", doc_id)
    chunks: list[Chunk] = []
    for heading, text in split_sections(body):
        for piece in window(text, max_words, overlap):
            cid = f"{doc_id}#{len(chunks):03d}"
            # Prefix gives retrievers the doc title and section as context.
            chunks.append(Chunk(cid, doc_id, title, heading, f"{doc_id.upper()} {title} / {heading}\n{piece}"))
    return chunks


def load_corpus(directory: str | Path, **kw) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(Path(directory).glob("*.md")):
        chunks.extend(chunk_document(path.stem, path.read_text(encoding="utf-8"), **kw))
    return chunks
