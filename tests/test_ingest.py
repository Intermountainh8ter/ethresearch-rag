from ethrag.ingest import chunk_document, parse_front_matter, split_sections, window

DOC = """---
eip: 9999
title: Test Proposal
---
Intro text before any heading.
## Specification
```
# not a heading
X = 1
```
Real content here.
## Rationale
Because.
"""


def test_front_matter():
    meta, body = parse_front_matter(DOC)
    assert meta["title"] == "Test Proposal" and body.startswith("Intro")


def test_headings_inside_code_fences_are_ignored():
    names = [h for h, _ in split_sections(parse_front_matter(DOC)[1])]
    assert names == ["Preamble", "Specification", "Rationale"]


def test_window_overlap_and_coverage():
    text = " ".join(str(i) for i in range(100))
    parts = window(text, 30, 10)
    assert parts[0].split()[-10:] == parts[1].split()[:10]
    assert set(" ".join(parts).split()) == set(text.split())


def test_chunk_ids_are_stable_and_prefixed():
    chunks = chunk_document("eip-9999", DOC)
    assert [c.id for c in chunks] == ["eip-9999#000", "eip-9999#001", "eip-9999#002"]
    assert chunks[1].text.startswith("EIP-9999 Test Proposal / Specification")
