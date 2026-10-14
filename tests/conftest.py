"""Shared fixtures. The real corpus (901 docs, ~17k chunks) is loaded once per test session."""
from __future__ import annotations

from pathlib import Path

import pytest

from ethrag import HybridIndex, load_corpus

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return DATA


@pytest.fixture(scope="session")
def corpus():
    return load_corpus(DATA)


@pytest.fixture(scope="session")
def corpus_index(corpus):
    """A HybridIndex over the full corpus (built once; ~5 s)."""
    return HybridIndex(corpus)
