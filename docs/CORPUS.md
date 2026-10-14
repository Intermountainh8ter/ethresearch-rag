# Corpus

`data/` holds **901 documents, 11.3 MB** (cap: 40 MB), indexed as **17,387 chunks**. Every file
is an unmodified copy of upstream markdown fetched from `raw.githubusercontent.com` by
`scripts/fetch_corpus.py`; `data/MANIFEST.json` records for each one its source URL, id, number,
title, status, category, type, created date, `requires`, byte size and sha256.

| source | repo | files | licence | ids |
|---|---|---|---|---|
| EIPs | [ethereum/EIPs](https://github.com/ethereum/EIPs) | 379 | CC0 1.0 | `data/eips/eip-N.md` |
| ERCs | [ethereum/ERCs](https://github.com/ethereum/ERCs) | 458 | CC0 1.0 | `data/eips/erc-N.md` |
| consensus specs | [ethereum/consensus-specs](https://github.com/ethereum/consensus-specs) | 38 | CC0 1.0 | `data/consensus-specs/cl-<fork>-<file>.md` |
| Flashbots docs | [flashbots/flashbots-docs](https://github.com/flashbots/flashbots-docs) | 26 | MIT (notice in `data/LICENSE`) | `data/flashbots/fb-<path>.md` |

The consensus specs cover phase0, altair, bellatrix, capella, deneb, electra and fulu
(`beacon-chain`, `fork-choice`, `validator`, `p2p-interface`, `fork`, light-client sync protocol
and `das-core` where they exist). The Flashbots pages are auction, protect, mev-boost and
mev-share documentation, converted from MDX by dropping import/export lines and bare JSX tags
at chunking time (the files on disk stay raw).

Status mix of the EIP/ERC documents (front matter): Stagnant 317, Final 241, Draft 171, Review 51,
Withdrawn 32, Last Call 23, Living 2; consensus specs are labelled `Spec`, Flashbots pages `Docs`.
**The corpus is not filtered by status**: a verified answer can be faithful to a Stagnant or
Withdrawn proposal. The status travels with every chunk.

## This is a curated subset, not the whole repos

`scripts/fetch_corpus.py` tries a curated list of EIP/ERC numbers (an explicit list of well-known
numbers plus the range 7521-7900) in *both* repos and keeps what exists. That is about 2,200
candidate URLs; 990 returned 404 (most numbers do not exist), 306 were rejected and 901 documents
were kept. The highest number is 7954. Core protocol EIPs, the ERC token and wallet standards and
the 2024-2025 Pectra/Fusaka drafts are well represented; many Draft ERCs outside the list are
absent. A question about a document that is not in `data/MANIFEST.json` cannot be answered.

Rules applied by the fetcher:

* only HTTP 200 bodies of at least 300 bytes whose first 400 bytes are not an HTML page;
* documents whose front matter says `status: Moved` are dropped;
* a number present in both repos is taken from the ERCs repo (the canonical home of ERCs);
* a full run removes files that are no longer part of the corpus; a partial run (`--numbers`,
  `--no-specs`) only adds or replaces files and merges into the existing manifest;
* the standard library only, 8 worker threads, 2 retries on transport errors.

Rebuild or extend:

```
python scripts/fetch_corpus.py                      # full curated run, rewrites data/ + MANIFEST.json
python scripts/fetch_corpus.py --numbers 7600,7623  # add or refresh specific numbers (merges)
python scripts/fetch_corpus.py --dry-run | head     # list the candidate URLs
ethrag build-index --data data --out .index         # then re-index
```

Upstream moves: a re-fetch pulls the *current* `master`, so hashes and the evaluation questions
can change. `tests/test_corpus.py` and `tests/test_questions.py` catch both, and
`python evals/build_questions.py` regenerates the questions from the new text.

## Chunking

`ethrag.ingest` turns a document into chunks with stable ids `doc#NNN`:

* front matter becomes metadata (`status`, `category`, `type`, `requires`, `created`, `number`)
  on every chunk; documents without front matter get `Spec`/`Docs` labels from their id prefix;
* the text is split on headings (a `#` inside a code fence is not a heading), then into **atomic
  blocks**: prose paragraphs, fenced code blocks and markdown tables;
* blocks are packed greedily into chunks of at most 180 words (30 words of overlap for prose);
  an oversized code block is windowed by lines and keeps its fences; an oversized table is split
  by rows with the **header repeated on every piece**, so a row never loses its column names;
* in-page table-of-contents bullets, HTML comments, MDX `import`/`export` lines and bare JSX
  tags are dropped; prose that merely starts with the word "import" survives;
* every chunk text starts with a context line `EIP-1559 <title> / <section>`, which is what
  BM25 title boosting and the verifier's provenance quote.

Chunk statistics: 14,711 prose, 2,150 code and 526 table chunks; median 69 words, 95th percentile
178, maximum 210 (EIPs 6,000, ERCs 10,007, consensus specs 1,151, Flashbots 229 chunks).

## Citation graph

`ethrag.graph.CitationGraph` has directed `requires` edges from front matter (997 edges) and
weaker `mentions` edges from inline `EIP-N` / `ERC-N` text (1,775 edges); 136 documents have no
edge of either kind. `ethrag related eip-1559 --depth 2` walks it in both directions.

## Integrity checks (`tests/test_corpus.py`)

The manifest covers every file and nothing else; sha256 and byte counts match; no file is empty,
tiny or HTML; no duplicate content or id; every `eips/` file is named `eip-N` or `erc-N` and its
front matter number agrees; total size is under 40 MB; every document produces at least one
chunk; chunk size is bounded.
