import pytest

from ethrag.ingest import (
    Chunk,
    chunk_document,
    clean_markdown,
    doc_metadata,
    doc_number,
    first_heading,
    iter_documents,
    load_corpus,
    pack_blocks,
    parse_front_matter,
    parse_requires,
    split_blocks,
    split_sections,
    split_table,
    window,
    window_lines,
)

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


def test_front_matter_absent_and_quoted_values():
    assert parse_front_matter("no front matter") == ({}, "no front matter")
    meta, _ = parse_front_matter('---\ntitle: "Quoted: yes"\n  nested: skipped\nStatus: Draft\n---\nx')
    assert meta == {"title": "Quoted: yes", "status": "Draft"}


def test_parse_requires_and_doc_number():
    assert parse_requires("2718, 2930, 2718") == (2718, 2930)
    assert parse_requires("") == () and parse_requires(None) == ()
    assert doc_number("whatever", {"eip": "1559"}) == 1559
    assert doc_number("whatever", {"erc": "721"}) == 721
    assert doc_number("erc-20", {}) == 20
    assert doc_number("cl-phase0-beacon-chain", {}) is None


def test_headings_inside_code_fences_are_ignored():
    names = [h for h, _ in split_sections(parse_front_matter(DOC)[1])]
    assert names == ["Preamble", "Specification", "Rationale"]


def test_window_overlap_and_coverage():
    text = " ".join(str(i) for i in range(100))
    parts = window(text, 30, 10)
    assert parts[0].split()[-10:] == parts[1].split()[:10]
    assert set(" ".join(parts).split()) == set(text.split())
    assert window("short text", 30, 10) == ["short text"]


def test_chunk_ids_are_stable_and_prefixed():
    chunks = chunk_document("eip-9999", DOC)
    assert [c.id for c in chunks] == ["eip-9999#000", "eip-9999#001", "eip-9999#002"]
    assert chunks[1].text.startswith("EIP-9999 Test Proposal / Specification")
    assert chunks[1].body.startswith("```")
    assert chunks[0].number == 9999 and chunks[0].title == "Test Proposal"


def test_chunk_carries_front_matter_metadata():
    raw = ("---\neip: 1559\ntitle: Fee\nstatus: Final\ncategory: Core\ntype: Standards Track\n"
           "created: 2019-04-13\nrequires: 2718, 2930\n---\n## Specification\nText.\n")
    c = chunk_document("eip-1559", raw)[0]
    assert (c.status, c.category, c.type, c.created) == ("Final", "Core", "Standards Track", "2019-04-13")
    assert c.requires == (2718, 2930) and c.number == 1559 and c.doc == "eip-1559"


def test_invalid_doc_id_rejected():
    with pytest.raises(ValueError):
        chunk_document("Bad Id", DOC)


def test_title_falls_back_to_first_heading_then_doc_id():
    assert chunk_document("cl-x", "# Phase 0 -- Beacon\n\nBody text.")[0].title == "Phase 0 -- Beacon"
    assert chunk_document("cl-y", "just body text")[0].title == "cl-y"
    assert first_heading("no heading here") == ""


def test_tables_stay_atomic_and_are_typed():
    raw = "---\ntitle: T\n---\n## Spec\nIntro.\n\n| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n"
    chunks = chunk_document("eip-9", raw)
    assert [c.kind for c in chunks] == ["table"]
    assert "| 3 | 4 |" in chunks[0].text and "Intro." in chunks[0].text


def test_long_table_is_split_by_rows_with_header_repeated():
    header = "| Name | Value |\n|------|-------|"
    rows = [f"| ROW_{i} | {i} |" for i in range(60)]
    pieces = split_table("\n".join([header, *rows]), max_words=40)
    assert len(pieces) > 1
    for p in pieces:
        assert p.startswith(header)
    covered = [line for p in pieces for line in p.splitlines()[2:]]
    assert covered == rows  # no row lost, none duplicated


def test_giant_table_row_is_windowed_on_its_own():
    big = "| " + " ".join(f"w{i}" for i in range(120)) + " | x |"
    pieces = split_table("| A | B |\n|---|---|\n| s | t |\n" + big, max_words=40)
    assert len(pieces) >= 3 and all(p.startswith("| A | B |") for p in pieces)


def test_code_blocks_stay_whole_when_small_and_keep_fences_when_split():
    small = "```py\nx = 1\n```"
    assert window_lines(small, 50) == [small]
    lines = [f"line_{i} = {i}" for i in range(80)]
    pieces = window_lines("```py\n" + "\n".join(lines) + "\n```", max_words=30)
    assert len(pieces) > 1
    assert all(p.startswith("```py") and p.rstrip().endswith("```") for p in pieces)
    assert {ln for p in pieces for ln in p.splitlines() if ln.startswith("line_")} == set(lines)


def test_code_without_closing_fence_and_giant_single_line():
    piece = window_lines("```\n" + " ".join(f"0x{i:02x}" for i in range(400)), 50)
    assert len(piece) > 1 and all(len(p.split()) <= 70 for p in piece)


def test_long_code_lines_do_not_blow_up_chunk_size():
    line = " ".join(f"w{i}" for i in range(180))
    raw = "---\ntitle: T\n---\n```\n" + "\n".join([line] * 6) + "\n```\n"
    for c in chunk_document("eip-9", raw, max_words=180):
        assert len(c.body.split()) <= 180 + 180 // 4 + 2  # fence tokens + capped overlap


def test_split_blocks_recognises_prose_code_table():
    blocks = split_blocks("para one\nstill one\n\n```\ncode\n```\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\nlast")
    assert [k for k, _ in blocks] == ["prose", "code", "table", "prose"]
    assert blocks[0][1] == "para one\nstill one"


def test_pack_blocks_merges_small_blocks_and_never_emits_subset_chunks():
    blocks = [("prose", " ".join(["a"] * 20)), ("code", "```\nx\n```"), ("prose", " ".join(["b"] * 25))]
    packed = pack_blocks(blocks, max_words=30, overlap=5)
    texts = [t for _, t in packed]
    assert all(len(t.split()) <= 35 for t in texts)
    for i, t in enumerate(texts):
        assert not any(i != j and t in u for j, u in enumerate(texts))


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


def test_clean_markdown_strips_comments_tags_but_not_code():
    raw = "a <!-- hidden -->b\n<Callout>\ntext\n</Callout>\n```\n<Callout>\nimport x from 'y';\n```\n"
    out = clean_markdown(raw)
    assert "hidden" not in out and "<Callout>\ntext" not in out.replace("```\n<Callout>", "")
    assert "```\n<Callout>\nimport x from 'y';\n```" in out


def test_load_corpus_walks_recursively_and_rejects_duplicate_ids(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "eip-1.md").write_text("---\ntitle: One\n---\nBody one.\n")
    (tmp_path / "b" / "eip-2.md").write_text("---\ntitle: Two\n---\nBody two.\n")
    (tmp_path / "b" / "notes.txt").write_text("ignored")
    assert [p.stem for p in iter_documents(tmp_path)] == ["eip-1", "eip-2"]
    assert {c.doc for c in load_corpus(tmp_path)} == {"eip-1", "eip-2"}
    (tmp_path / "b" / "eip-1.md").write_text("---\ntitle: Dup\n---\nBody.\n")
    with pytest.raises(ValueError, match="duplicate document id"):
        load_corpus(tmp_path)


def test_doc_metadata_counts_chunks(mini_chunks):
    docs = doc_metadata(mini_chunks)
    assert docs["eip-1559"].requires == (2718, 2930) and docs["eip-1559"].status == "Final"
    assert docs["eip-1559"].chunks == sum(c.doc == "eip-1559" for c in mini_chunks)
    assert docs["cl-phase0-beacon-chain"].number is None


def test_chunk_is_immutable_and_body_without_context_line():
    c = Chunk("eip-1#000", "eip-1", "T", "S", "EIP-1 T / S\nbody here")
    assert c.body == "body here"
    assert Chunk("eip-1#001", "eip-1", "T", "S", "single line").body == "single line"
    with pytest.raises(AttributeError):
        c.id = "x"  # type: ignore[misc]


def test_real_corpus_has_all_three_chunk_kinds(corpus):
    assert {c.kind for c in corpus} == {"prose", "code", "table"}


def test_spec_and_docs_without_front_matter_get_manifest_style_labels():
    cl = chunk_document("cl-phase0-beacon-chain", "# Phase 0\n\nSpec body text here.")[0]
    fb = chunk_document("fb-auction-overview", "---\ntitle: Overview\n---\nDocs body text here.")[0]
    assert (cl.status, cl.category, cl.type) == ("Spec", "Consensus", "Specification")
    assert (fb.status, fb.category, fb.type) == ("Docs", "MEV", "Documentation")
    eip = chunk_document("eip-9", "---\ntitle: T\nstatus: Draft\n---\nBody text here.")[0]
    assert eip.status == "Draft" and eip.category == "" and chunk_document("eip-8", "plain body text")[0].status == ""
    explicit = chunk_document("cl-x", "---\ntitle: T\nstatus: Final\n---\nBody text here.")[0]
    assert explicit.status == "Final"  # front matter always wins


def test_stats_style_status_counts_have_no_unknown_for_spec_docs(corpus):
    from ethrag.ingest import doc_metadata

    meta = doc_metadata(corpus)
    assert all(m.status for d, m in meta.items() if d.startswith(("cl-", "fb-")))
