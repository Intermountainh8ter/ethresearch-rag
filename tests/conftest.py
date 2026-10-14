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


# --------------------------------------------------------------------------------- mini corpus
MINI_DOCS = {
    "eips/eip-2718.md": """---
eip: 2718
title: Typed Transaction Envelope
status: Final
type: Standards Track
category: Core
created: 2020-06-13
---

## Abstract

`TransactionType || TransactionPayload` is a new envelope for typed transactions.

## Specification

The transaction type is a single byte in the range 0x00 to 0x7f. The type 0x7f is reserved.

| Type | Meaning |
|------|---------|
| 0x00 | legacy |
| 0x01 | access list |
""",
    "eips/eip-2929.md": """---
eip: 2929
title: Gas cost increases for state access opcodes
status: Final
type: Standards Track
category: Core
created: 2020-09-01
---

## Specification

| Constant | Value |
|----------|-------|
| `COLD_SLOAD_COST` | 2100 |
| `COLD_ACCOUNT_ACCESS_COST` | 2600 |
| `WARM_STORAGE_READ_COST` | 100 |

Reading a cold storage slot costs COLD_SLOAD_COST gas.
""",
    "eips/eip-2930.md": """---
eip: 2930
title: Optional access lists
status: Final
type: Standards Track
category: Core
created: 2020-08-29
requires: 2718, 2929
---

## Specification

A new transaction type 0x01 carries an access list, see EIP-2718 for the envelope.

The cost is ACCESS_LIST_ADDRESS_COST = 2400 per address and ACCESS_LIST_STORAGE_KEY_COST = 1900 per key.
""",
    "eips/eip-1559.md": """---
eip: 1559
title: Fee market change for ETH 1.0 chain
status: Final
type: Standards Track
category: Core
created: 2019-04-13
requires: 2718, 2930
---

## Abstract

A base fee per gas that is burned and adjusts with block fullness.

## Specification

```python
INITIAL_BASE_FEE = 1000000000
BASE_FEE_MAX_CHANGE_DENOMINATOR = 8
ELASTICITY_MULTIPLIER = 2
```

The base fee is not paid to the miner; it is burned. Transactions use type 0x02 as defined in EIP-2718.

## Rationale

The elasticity multiplier lets blocks be twice the target size.
""",
    "eips/eip-4844.md": """---
eip: 4844
title: Shard Blob Transactions
status: Final
type: Standards Track
category: Core
created: 2022-02-25
requires: 1559, 2718, 2930
---

## Specification

| Parameter | Value |
|-----------|-------|
| `BLOB_TX_TYPE` | `0x03` |
| `GAS_PER_BLOB` | `2**17` |
| `MAX_BLOB_GAS_PER_BLOCK` | `786432` |

Blob transactions carry KZG commitments and versioned hashes. They build on EIP-1559.
""",
    "eips/erc-20.md": """---
eip: 20
title: Token Standard
status: Final
type: Standards Track
category: ERC
created: 2015-11-19
---

## Specification

Every token implements `totalSupply`, `balanceOf` and `transfer`. Decimals is OPTIONAL.
""",
    "consensus-specs/cl-phase0-beacon-chain.md": """# Phase 0 -- The Beacon Chain

- [Introduction](#introduction)
- [Constants](#constants)

## Introduction

This document represents the specification for Phase 0: the beacon chain.

## Constants

| Name | Value |
| - | - |
| `SLOTS_PER_EPOCH` | `uint64(2**5)` (= 32) |
| `SECONDS_PER_SLOT` | `uint64(12)` |
""",
}


@pytest.fixture(scope="session")
def mini_dir(tmp_path_factory) -> Path:
    base = tmp_path_factory.mktemp("mini-data")
    for rel, text in MINI_DOCS.items():
        path = base / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return base


@pytest.fixture(scope="session")
def mini_chunks(mini_dir):
    return load_corpus(mini_dir)


@pytest.fixture(scope="session")
def mini_index(mini_chunks):
    return HybridIndex(mini_chunks)


@pytest.fixture()
def mini_pipeline(mini_chunks, mini_index):
    from ethrag import RagPipeline

    return RagPipeline(mini_chunks, None, index=mini_index, k=5)


class RepeatLLM:
    """Always answers with the same reply (unlike ScriptedLLM it never runs out)."""

    def __init__(self, reply: str):
        self.reply, self.calls = reply, []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.reply


@pytest.fixture(scope="session")
def repeat_llm():
    return RepeatLLM


@pytest.fixture(scope="session")
def cid(mini_chunks):
    """cid(doc, needle) -> id of the first chunk of ``doc`` containing ``needle``."""
    def find(doc: str, needle: str) -> str:
        return next(c.id for c in mini_chunks if c.doc == doc and needle in c.text)
    return find
