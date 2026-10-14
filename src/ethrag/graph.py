"""Networkless citation graph between documents.

Edges come from two places, both available offline:

* ``requires`` front matter (EIP-1559 requires 2718, 2930) - strong, directed edges;
* inline mentions of ``EIP-N`` / ``ERC-N`` in chunk text - weak, directed edges.

``related()`` walks the graph (both directions) to a bounded depth and is what the retriever's
graph-expansion option uses to pull in chunks from dependencies of the top-ranked documents.
"""
from __future__ import annotations

import re
from collections import Counter, deque
from collections.abc import Iterable, Sequence

from .ingest import Chunk, DocMeta, doc_metadata

_MENTION = re.compile(r"\b(?:EIP|ERC)[-\s]?(\d{1,5})\b", re.IGNORECASE)


def mentioned_numbers(text: str) -> list[int]:
    """EIP/ERC numbers named in ``text`` (``EIP-1559``, ``eip 1559``, ``ERC20``), in order."""
    return [int(n) for n in _MENTION.findall(text)]


class CitationGraph:
    def __init__(self, chunks: Sequence[Chunk], mention_min_count: int = 1):
        self.docs: dict[str, DocMeta] = doc_metadata(list(chunks))
        self.by_number: dict[int, str] = {}
        # eip-N wins over erc-N when both exist (the EIPs copy is the one core specs cite).
        for doc_id, meta in sorted(self.docs.items(), key=lambda kv: kv[0].startswith("erc-")):
            if meta.number is not None:
                self.by_number.setdefault(meta.number, doc_id)
        self.requires: dict[str, set[str]] = {d: set() for d in self.docs}
        self.required_by: dict[str, set[str]] = {d: set() for d in self.docs}
        self.mentions: dict[str, Counter[str]] = {d: Counter() for d in self.docs}
        self.mentioned_by: dict[str, set[str]] = {d: set() for d in self.docs}
        for doc_id, meta in self.docs.items():
            for n in meta.requires:
                target = self.by_number.get(n)
                if target and target != doc_id:
                    self.requires[doc_id].add(target)
                    self.required_by[target].add(doc_id)
        for c in chunks:
            for n in mentioned_numbers(c.body):
                target = self.by_number.get(n)
                if target and target != c.doc:
                    self.mentions[c.doc][target] += 1
        for src, counter in self.mentions.items():
            for target, count in counter.items():
                if count >= mention_min_count:
                    self.mentioned_by[target].add(src)

    def resolve(self, number: int) -> str | None:
        return self.by_number.get(number)

    def neighbours(self, doc: str, kinds: Iterable[str] = ("requires", "required_by")) -> set[str]:
        out: set[str] = set()
        for kind in kinds:
            if kind == "requires":
                out |= self.requires.get(doc, set())
            elif kind == "required_by":
                out |= self.required_by.get(doc, set())
            elif kind == "mentions":
                out |= set(self.mentions.get(doc, Counter()))
            elif kind == "mentioned_by":
                out |= self.mentioned_by.get(doc, set())
            else:
                raise ValueError(f"unknown edge kind {kind!r}")
        return out

    def related(self, doc: str, depth: int = 1,
                kinds: Iterable[str] = ("requires", "required_by")) -> list[tuple[str, int]]:
        """Documents reachable from ``doc`` within ``depth`` hops, as (doc_id, distance)."""
        if doc not in self.docs:
            return []
        kinds = tuple(kinds)
        seen = {doc: 0}
        q: deque[str] = deque([doc])
        while q:
            cur = q.popleft()
            if seen[cur] >= depth:
                continue
            for nxt in sorted(self.neighbours(cur, kinds)):
                if nxt not in seen:
                    seen[nxt] = seen[cur] + 1
                    q.append(nxt)
        del seen[doc]
        return sorted(seen.items(), key=lambda kv: (kv[1], kv[0]))

    def edges(self) -> list[tuple[str, str, str]]:
        """All (source, target, kind) edges, deterministic order."""
        out = [(s, t, "requires") for s in sorted(self.requires) for t in sorted(self.requires[s])]
        out += [(s, t, "mentions") for s in sorted(self.mentions) for t in sorted(self.mentions[s])]
        return out

    def stats(self) -> dict[str, int]:
        return {
            "docs": len(self.docs),
            "requires_edges": sum(len(v) for v in self.requires.values()),
            "mention_edges": sum(len(v) for v in self.mentions.values()),
            "isolated": sum(1 for d in self.docs
                            if not self.neighbours(d, ("requires", "required_by", "mentions", "mentioned_by"))),
        }
