"""Markdown specs -> metadata-carrying, structure-aware chunks with stable ids.

Chunking keeps three kinds of block atomic: prose paragraphs, fenced code blocks and markdown
tables. Blocks are greedily packed into chunks of at most ``max_words`` words; an oversized
paragraph or code block is windowed, and an oversized table is split by rows with its header
repeated on every piece so a row never loses its column names. Front-matter metadata (status,
category, type, requires, created) rides along on every chunk so retrieval priors and the
citation graph never need the source files again.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

_HEADING = re.compile(r"^(#{1,6})\s+(.*)")
_FENCE = re.compile(r"^\s*(```|~~~)")
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
# MDX module lines only: `import X from "y";` / `export default ...` / `export const ...`. Prose that
# merely starts with the word "import"/"export" (hard-wrapped paragraphs) must survive.
_MDX_LINE = re.compile(
    r"^\s*(import\s+[\w{}\s,*]+\s+from\s+['\"][^'\"]+['\"];?"
    r"|import\s+['\"][^'\"]+['\"];?"
    r"|export\s+(?:default|const|let|var|function|async|\{).*)\s*$")
# A table-of-contents bullet that only links to an in-page anchor: "  - [Types](#types)".
_TOC_LINE = re.compile(r"^\s*[-*+]\s+\[[^\]]*\]\(#[^)]*\)\s*$")
_TAG_LINE = re.compile(r"^\s*</?[A-Za-z][A-Za-z0-9]*(\s[^>]*)?/?>\s*$")
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_DOC_ID = re.compile(r"^[a-z][a-z0-9-]*$")


@dataclass(frozen=True)
class Chunk:
    id: str  # e.g. "eip-1559#007"
    doc: str  # e.g. "eip-1559"
    title: str
    section: str
    text: str
    status: str = ""
    category: str = ""
    type: str = ""
    requires: tuple[int, ...] = ()
    created: str = ""
    kind: str = "prose"  # prose | code | table
    number: int | None = None

    @property
    def body(self) -> str:
        """Chunk text without the ``DOC Title / Section`` context line."""
        nl = self.text.find("\n")
        return self.text[nl + 1:] if nl >= 0 else self.text


@dataclass
class DocMeta:
    id: str
    title: str
    number: int | None = None
    status: str = ""
    category: str = ""
    type: str = ""
    requires: tuple[int, ...] = ()
    created: str = ""
    chunks: int = 0
    extra: dict[str, str] = field(default_factory=dict)


def parse_front_matter(raw: str) -> tuple[dict[str, str], str]:
    """Return (front-matter mapping with lower-cased keys, body)."""
    m = re.match(r"^---\r?\n(.*?)\r?\n---[ \t]*\r?\n?", raw, re.DOTALL)
    if not m:
        return {}, raw
    meta: dict[str, str] = {}
    for line in m.group(1).splitlines():
        if ":" in line and not line.startswith((" ", "\t")):
            k, v = line.split(":", 1)
            meta[k.strip().lower()] = v.strip().strip('"').strip("'")
    return meta, raw[m.end():]


def parse_requires(value: str | None) -> tuple[int, ...]:
    return tuple(sorted({int(x) for x in re.findall(r"\d+", value or "")}))


def doc_number(doc_id: str, meta: dict[str, str]) -> int | None:
    for key in ("eip", "erc"):
        if meta.get(key, "").strip().isdigit():
            return int(meta[key])
    m = re.match(r"^(?:eip|erc)-(\d+)$", doc_id)
    return int(m.group(1)) if m else None


def clean_markdown(body: str) -> str:
    """Drop MDX import/export lines, bare JSX/HTML tag lines, in-page TOC bullets and HTML comments."""
    body = _HTML_COMMENT.sub("", body)
    out, in_code = [], False
    for line in body.splitlines():
        if _FENCE.match(line):
            in_code = not in_code
            out.append(line)
            continue
        if not in_code and (_MDX_LINE.match(line) or _TAG_LINE.match(line)
                            or _TOC_LINE.match(line)):
            continue
        out.append(line)
    return "\n".join(out)


def split_sections(body: str) -> list[tuple[str, str]]:
    """Split on markdown headings, ignoring '#' lines inside fenced code blocks."""
    sections: list[tuple[str, list[str]]] = [("Preamble", [])]
    in_code = False
    for line in body.splitlines():
        if _FENCE.match(line):
            in_code = not in_code
        m = _HEADING.match(line)
        if m and not in_code:
            sections.append((m.group(2).strip().strip("#").strip(), []))
        else:
            sections[-1][1].append(line)
    return [(h, "\n".join(ls).strip()) for h, ls in sections if "\n".join(ls).strip()]


def split_blocks(text: str) -> list[tuple[str, str]]:
    """Split section text into atomic (kind, text) blocks: prose | code | table."""
    blocks: list[tuple[str, str]] = []
    lines = text.splitlines()
    i, n = 0, len(lines)
    para: list[str] = []

    def flush() -> None:
        if para and "\n".join(para).strip():
            blocks.append(("prose", "\n".join(para).strip()))
        para.clear()

    while i < n:
        line = lines[i]
        if _FENCE.match(line):
            flush()
            fence = [line]
            i += 1
            while i < n and not _FENCE.match(lines[i]):
                fence.append(lines[i])
                i += 1
            if i < n:
                fence.append(lines[i])
                i += 1
            blocks.append(("code", "\n".join(fence)))
            continue
        if _TABLE_ROW.match(line) and i + 1 < n and _TABLE_SEP.match(lines[i + 1]):
            flush()
            rows = [line, lines[i + 1]]
            i += 2
            while i < n and _TABLE_ROW.match(lines[i]):
                rows.append(lines[i])
                i += 1
            blocks.append(("table", "\n".join(rows)))
            continue
        if not line.strip():
            flush()
        else:
            para.append(line)
        i += 1
    flush()
    return blocks


def window(text: str, max_words: int, overlap: int) -> list[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text]
    out, step = [], max(1, max_words - overlap)
    for start in range(0, len(words), step):
        out.append(" ".join(words[start:start + max_words]))
        if start + max_words >= len(words):
            break
    return out


def window_lines(text: str, max_words: int, overlap_lines: int = 2) -> list[str]:
    """Window a code block by lines (never splitting a line), keeping the fence on each piece."""
    lines = text.splitlines()
    if sum(len(ln.split()) for ln in lines) <= max_words:
        return [text]
    open_fence = lines[0] if _FENCE.match(lines[0]) else ""
    close_fence = lines[-1] if len(lines) > 1 and _FENCE.match(lines[-1]) else ""
    inner = lines[1 if open_fence else 0: len(lines) - (1 if close_fence else 0)]
    # A single enormous line (e.g. hex bytecode) cannot be split by lines: window it by words.
    inner = [piece for ln in inner for piece in
             (window(ln, max_words, 0) if len(ln.split()) > max_words else [ln])]
    cap = max(1, max_words // 4)  # overlap lines may never crowd out the new content
    pieces, cur, count = [], [], 0
    for ln in inner:
        w = max(1, len(ln.split()))
        if cur and count + w > max_words:
            pieces.append(cur)
            keep: list[str] = []
            for prev in reversed(cur[-overlap_lines:] if overlap_lines else []):
                if sum(max(1, len(x.split())) for x in keep) + max(1, len(prev.split())) > cap:
                    break
                keep.insert(0, prev)
            cur = keep
            count = sum(max(1, len(x.split())) for x in cur)
        cur.append(ln)
        count += w
    if cur:
        pieces.append(cur)
    return ["\n".join([open_fence, *p, close_fence] if open_fence else p) for p in pieces]


def split_table(text: str, max_words: int) -> list[str]:
    """Split a long markdown table by rows, repeating the header + separator on each piece."""
    rows = text.splitlines()
    header, body = rows[:2], rows[2:]
    hw = sum(len(r.split()) for r in header)
    if hw + sum(len(r.split()) for r in body) <= max_words:
        return [text]
    pieces, cur, count = [], [], hw
    for r in body:
        w = len(r.split())
        if w > max_words:  # one giant row: emit it on its own, windowed by words
            if cur:
                pieces.append("\n".join(header + cur))
                cur, count = [], hw
            pieces.extend("\n".join(header + [p]) for p in window(r, max_words, 0))
            continue
        if cur and count + w > max_words:
            pieces.append("\n".join(header + cur))
            cur, count = [], hw
        cur.append(r)
        count += w
    if cur:
        pieces.append("\n".join(header + cur))
    return pieces


def pack_blocks(blocks: list[tuple[str, str]], max_words: int, overlap: int) -> list[tuple[str, str]]:
    """Greedily pack atomic blocks into (kind, text) chunks of at most ``max_words`` words."""
    units: list[tuple[str, str]] = []
    for kind, text in blocks:
        if kind == "code":
            units.extend(("code", p) for p in window_lines(text, max_words))
        elif kind == "table":
            units.extend(("table", p) for p in split_table(text, max_words))
        else:
            units.extend(("prose", p) for p in window(text, max_words, overlap))
    chunks: list[tuple[str, str]] = []
    cur: list[tuple[str, str]] = []
    count = 0
    for kind, text in units:
        w = len(text.split())
        if cur and count + w > max_words:
            chunks.append(_merge(cur))
            tail = cur[-1]
            # Carry a short trailing block forward as overlap, but never when it was the whole
            # chunk (that would emit a chunk that is a strict subset of the next one).
            carry = len(cur) > 1 and len(tail[1].split()) <= overlap
            cur = [tail] if carry else []
            count = sum(len(t.split()) for _, t in cur)
        cur.append((kind, text))
        count += w
    if cur:
        chunks.append(_merge(cur))
    return chunks


def _merge(units: list[tuple[str, str]]) -> tuple[str, str]:
    kinds = {k for k, _ in units}
    kind = "table" if "table" in kinds else ("code" if kinds == {"code"} else "prose")
    return kind, "\n\n".join(t for _, t in units)


def first_heading(body: str) -> str:
    m = re.search(r"^#\s+(.+)$", body, re.MULTILINE)
    return m.group(1).strip() if m else ""


def chunk_document(doc_id: str, raw: str, max_words: int = 180, overlap: int = 30) -> list[Chunk]:
    if not _DOC_ID.match(doc_id):
        raise ValueError(f"doc id {doc_id!r} must match {_DOC_ID.pattern}")
    meta, body = parse_front_matter(raw)
    body = clean_markdown(body)
    title = meta.get("title") or first_heading(body) or doc_id
    number = doc_number(doc_id, meta)
    common = {
        "status": meta.get("status", ""), "category": meta.get("category", ""),
        "type": meta.get("type", ""), "requires": parse_requires(meta.get("requires")),
        "created": meta.get("created", ""), "number": number,
    }
    chunks: list[Chunk] = []
    for heading, text in split_sections(body):
        for kind, piece in pack_blocks(split_blocks(text), max_words, overlap):
            cid = f"{doc_id}#{len(chunks):03d}"
            # Prefix gives retrievers the doc title and section as context.
            chunks.append(Chunk(cid, doc_id, title, heading,
                                f"{doc_id.upper()} {title} / {heading}\n{piece}", kind=kind, **common))
    return chunks


def iter_documents(directory: str | Path) -> list[Path]:
    return sorted(p for p in Path(directory).rglob("*.md") if p.is_file())


def load_corpus(directory: str | Path, **kw) -> list[Chunk]:
    chunks: list[Chunk] = []
    seen: set[str] = set()
    for path in iter_documents(directory):
        if path.stem in seen:
            raise ValueError(f"duplicate document id {path.stem!r} under {directory}")
        seen.add(path.stem)
        chunks.extend(chunk_document(path.stem, path.read_text(encoding="utf-8"), **kw))
    return chunks


def doc_metadata(chunks: list[Chunk]) -> dict[str, DocMeta]:
    """One ``DocMeta`` per document, derived from the chunks (no file access needed)."""
    docs: dict[str, DocMeta] = {}
    for c in chunks:
        d = docs.get(c.doc)
        if d is None:
            d = docs[c.doc] = DocMeta(c.doc, c.title, c.number, c.status, c.category, c.type,
                                      c.requires, c.created)
        d.chunks += 1
    return docs
