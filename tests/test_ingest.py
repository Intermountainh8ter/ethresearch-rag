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


def test_toc_bullets_dropped_but_intro_prose_under_title_heading_kept():
    raw = ("---\ntitle: T\n---\n# T\n- [Intro](#intro)\n  - [Sub](#sub)\n\n"
           "Real intro prose lives under the title heading.\n## Intro\nx\n")
    chunks = chunk_document("eip-9", raw)
    assert "Real intro prose" in chunks[0].text and "(#intro)" not in chunks[0].text


def test_prose_starting_with_import_or_export_survives_but_mdx_lines_do_not():
    raw = ("---\ntitle: T\n---\nWe import data from the registry\nand export the state.\n"
           "import Foo from \"./foo\";\nexport default Layout;\n")
    text = chunk_document("eip-9", raw)[0].text
    assert "import data from the registry" in text and "export the state" in text
    assert "./foo" not in text and "Layout" not in text


def test_long_code_lines_do_not_blow_up_chunk_size():
    line = " ".join(f"w{i}" for i in range(180))
    raw = "---\ntitle: T\n---\n```\n" + "\n".join([line] * 6) + "\n```\n"
    for c in chunk_document("eip-9", raw, max_words=180):
        assert len(c.body.split()) <= 180 + 180 // 4 + 2  # fence tokens + capped overlap
