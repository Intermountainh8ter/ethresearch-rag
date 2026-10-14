"""scripts/fetch_corpus.py with the network replaced by canned responses."""
import importlib.util
import json
import sys

import pytest


@pytest.fixture()
def fc(root):
    spec = importlib.util.spec_from_file_location("fetch_corpus_script", root / "scripts" / "fetch_corpus.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # @dataclass looks its module up here
    spec.loader.exec_module(mod)
    yield mod
    sys.modules.pop(spec.name, None)


def eip(n, title="T", status="Final", body="Body text. " * 40, typ="Standards Track", cat="Core", requires=""):
    req = f"requires: {requires}\n" if requires else ""
    return (f"---\neip: {n}\ntitle: {title}\nstatus: {status}\ntype: {typ}\ncategory: {cat}\n"
            f"created: 2020-01-01\n{req}---\n\n## Abstract\n{body}\n").encode()


def fake_network(fc, monkeypatch, table):
    seen = []

    def fetch(url):
        seen.append(url)
        for suffix, (code, body) in table.items():
            if url.endswith(suffix):
                return code, body
        return 404, b""

    monkeypatch.setattr(fc, "fetch", fetch)
    return seen


def test_dry_run_lists_urls_for_every_source(fc, capsys):
    assert fc.main(["--dry-run", "--numbers", "1559"]) == 0
    urls = capsys.readouterr().out.split()
    assert any(u.endswith("/ethereum/EIPs/master/EIPS/eip-1559.md") for u in urls)
    assert any(u.endswith("/ethereum/ERCs/master/ERCS/erc-1559.md") for u in urls)
    assert any("consensus-specs/master/specs/phase0/beacon-chain.md" in u for u in urls)
    assert any("flashbots-docs/main/docs/" in u for u in urls)
    capsys.readouterr()
    assert fc.main(["--dry-run", "--no-specs", "--numbers", "1"]) == 0
    assert len(capsys.readouterr().out.split()) == 2  # one EIP url, one ERC url


def test_keeps_ok_bodies_skips_404_html_tiny_and_moved(fc, monkeypatch, tmp_path, capsys):
    fake_network(fc, monkeypatch, {
        "EIPS/eip-1.md": (200, eip(1, "Guidelines", "Living", typ="Meta", cat="")),
        "EIPS/eip-2.md": (200, b"<!DOCTYPE html><html>" + b"x" * 400),
        "EIPS/eip-3.md": (200, b"tiny"),
        "EIPS/eip-4.md": (200, eip(4, status="Moved")),
        "EIPS/eip-5.md": (500, b""),
        "specs/phase0/beacon-chain.md": (200, b"# Phase 0 -- The Beacon Chain\n\n" + b"Spec text here. " * 40),
    })
    assert fc.main(["--out", str(tmp_path), "--numbers", "1,2,3,4,5,6"]) == 0
    out = capsys.readouterr().out
    assert "kept 2 docs" in out and "'404':" in out and "'other': 1" in out and "'rejected': 3" in out
    man = json.loads((tmp_path / "MANIFEST.json").read_text())
    assert {e["id"] for e in man["entries"]} == {"eip-1", "cl-phase0-beacon-chain"}
    e1 = next(e for e in man["entries"] if e["id"] == "eip-1")
    assert (e1["title"], e1["status"], e1["type"], e1["number"]) == ("Guidelines", "Living", "Meta", 1)
    cl = next(e for e in man["entries"] if e["id"] == "cl-phase0-beacon-chain")
    assert (cl["status"], cl["category"], cl["title"]) == ("Spec", "Consensus", "Phase 0 -- The Beacon Chain")
    assert (tmp_path / "eips" / "eip-1.md").is_file() and not (tmp_path / "eips" / "eip-4.md").exists()
    assert man["documents"] == 2 and man["bytes"] == sum(e["bytes"] for e in man["entries"])


def test_the_ercs_repo_wins_over_a_stale_eips_copy_and_sha256_is_recorded(fc, monkeypatch, tmp_path):
    import hashlib

    erc = eip(20, "Token Standard", cat="ERC", requires="165")
    fake_network(fc, monkeypatch, {"EIPS/eip-20.md": (200, eip(20, "Old copy")), "ERCS/erc-20.md": (200, erc)})
    assert fc.main(["--out", str(tmp_path), "--numbers", "20", "--no-specs"]) == 0
    man = json.loads((tmp_path / "MANIFEST.json").read_text())
    assert [e["id"] for e in man["entries"]] == ["erc-20"]
    assert man["entries"][0]["sha256"] == hashlib.sha256(erc).hexdigest() and man["entries"][0]["requires"] == [165]
    assert not (tmp_path / "eips" / "eip-20.md").exists()


def test_a_partial_run_merges_and_never_deletes_other_documents(fc, monkeypatch, tmp_path):
    fake_network(fc, monkeypatch, {"EIPS/eip-1.md": (200, eip(1, "One")), "EIPS/eip-2.md": (200, eip(2, "Two"))})
    assert fc.main(["--out", str(tmp_path), "--numbers", "1,2", "--no-specs"]) == 0
    fake_network(fc, monkeypatch, {"EIPS/eip-3.md": (200, eip(3, "Three")), "EIPS/eip-2.md": (200, eip(2, "Two v2"))})
    assert fc.main(["--out", str(tmp_path), "--numbers", "3,2", "--no-specs"]) == 0
    man = json.loads((tmp_path / "MANIFEST.json").read_text())
    assert {e["id"] for e in man["entries"]} == {"eip-1", "eip-2", "eip-3"}
    assert next(e for e in man["entries"] if e["id"] == "eip-2")["title"] == "Two v2"
    assert (tmp_path / "eips" / "eip-1.md").is_file() and man["documents"] == 3


def test_a_full_run_prunes_files_that_left_the_corpus(fc, monkeypatch, tmp_path):
    monkeypatch.setattr(fc, "KNOWN_NUMBERS", [1])
    monkeypatch.setattr(fc, "SPEC_FORKS", [])
    monkeypatch.setattr(fc, "FLASHBOTS_PAGES", [])
    stale = tmp_path / "eips"
    stale.mkdir(parents=True)
    (stale / "eip-999.md").write_text("stale")
    fake_network(fc, monkeypatch, {"EIPS/eip-1.md": (200, eip(1))})
    assert fc.main(["--out", str(tmp_path)]) == 0
    assert not (stale / "eip-999.md").exists() and (stale / "eip-1.md").is_file()


def test_transport_errors_are_retried_then_reported_as_status_zero(fc, monkeypatch, capsys):
    import urllib.error

    calls = []

    def boom(url, timeout):
        calls.append(url)
        raise urllib.error.URLError("no route")

    monkeypatch.setattr(fc.urllib.request, "urlopen", boom)
    monkeypatch.setattr(fc.time, "sleep", lambda s: None)
    assert fc.fetch("https://example.invalid/x") == (0, b"")
    assert len(calls) == fc.RETRIES + 1 and "no route" in capsys.readouterr().err


def test_http_errors_return_their_status_without_retrying(fc, monkeypatch):
    import urllib.error

    def nf(url, timeout):
        raise urllib.error.HTTPError(url, 404, "nf", {}, None)

    monkeypatch.setattr(fc.urllib.request, "urlopen", nf)
    assert fc.fetch("https://example.invalid/x") == (404, b"")


def test_curated_number_list_is_sorted_unique_and_in_range(fc):
    assert fc.KNOWN_NUMBERS == sorted(set(fc.KNOWN_NUMBERS)) and fc.KNOWN_NUMBERS[0] == 1
    assert 1559 in fc.KNOWN_NUMBERS and 4844 in fc.KNOWN_NUMBERS
