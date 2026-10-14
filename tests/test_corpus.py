"""Integrity of the bundled corpus: manifest hashes, no empty/duplicate/foreign files, size cap."""
from __future__ import annotations

import hashlib
import json
import re

import pytest

MIN_BYTES = 300
MAX_TOTAL_BYTES = 40 * 1024 * 1024


@pytest.fixture(scope="module")
def manifest(data_dir):
    return json.loads((data_dir / "MANIFEST.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def md_files(data_dir):
    return sorted(p for p in data_dir.rglob("*") if p.is_file() and p.suffix == ".md")


def test_only_markdown_plus_license_and_manifest(data_dir):
    extra = [p.name for p in data_dir.rglob("*")
             if p.is_file() and p.suffix != ".md" and p.name not in ("LICENSE", "MANIFEST.json")]
    assert extra == []


def test_manifest_covers_every_file_exactly(manifest, md_files, data_dir):
    rel = {str(p.relative_to(data_dir)) for p in md_files}
    listed = {e["path"] for e in manifest["entries"]}
    assert rel == listed
    assert manifest["documents"] == len(manifest["entries"]) == len(md_files)


def test_sha256_and_bytes_match_manifest(manifest, data_dir):
    for e in manifest["entries"]:
        raw = (data_dir / e["path"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == e["sha256"], e["path"]
        assert len(raw) == e["bytes"], e["path"]
    assert sum(e["bytes"] for e in manifest["entries"]) == manifest["bytes"]


def test_no_empty_tiny_or_html_files(md_files):
    for p in md_files:
        raw = p.read_bytes()
        assert len(raw) >= MIN_BYTES, f"{p.name} is only {len(raw)} bytes"
        assert b"<!DOCTYPE html" not in raw[:400] and b"<html" not in raw[:400], p.name


def test_no_duplicate_content_or_ids(md_files):
    digests, ids = {}, {}
    for p in md_files:
        digests.setdefault(hashlib.sha256(p.read_bytes()).hexdigest(), []).append(p.name)
        ids.setdefault(p.stem, []).append(p.name)
    assert [v for v in digests.values() if len(v) > 1] == []
    assert [v for v in ids.values() if len(v) > 1] == []


def test_eip_and_erc_files_are_what_their_names_claim(data_dir):
    for p in sorted((data_dir / "eips").glob("*.md")):
        m = re.fullmatch(r"(eip|erc)-(\d+)", p.stem)
        assert m, f"non-EIP file in data/eips: {p.name}"
        fm = re.match(r"^---\n(.*?)\n---", p.read_text(encoding="utf-8"), re.DOTALL)
        assert fm, f"{p.name} has no front matter"
        number = re.search(r"^(?:eip|erc):\s*(\d+)\s*$", fm.group(1), re.MULTILINE)
        assert number and number.group(1) == m.group(2), p.name
        assert not re.search(r"^status:\s*Moved\s*$", fm.group(1), re.MULTILINE | re.IGNORECASE)


def test_other_sources_use_their_prefixes(data_dir):
    assert all(p.name.startswith("cl-") for p in (data_dir / "consensus-specs").glob("*.md"))
    assert all(p.name.startswith("fb-") for p in (data_dir / "flashbots").glob("*.md"))


def test_total_size_under_cap(md_files):
    assert sum(p.stat().st_size for p in md_files) < MAX_TOTAL_BYTES


def test_license_notice_present(data_dir):
    text = (data_dir / "LICENSE").read_text(encoding="utf-8")
    assert "CC0" in text and "MIT" in text


def test_chunks_cover_every_document_and_stay_bounded(corpus, manifest):
    assert {c.doc for c in corpus} == {e["id"] for e in manifest["entries"]}
    assert len({c.id for c in corpus}) == len(corpus)
    assert max(len(c.body.split()) for c in corpus) <= 220
