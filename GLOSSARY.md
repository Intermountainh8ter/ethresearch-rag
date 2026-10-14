# Glossary

Every abbreviation and piece of jargon used in this repository, in plain words.

## Search and retrieval

**BM25** — "Best Match 25", the standard word-counting search algorithm. It
ranks a document by how often the query's words appear in it, discounting words
that appear everywhere. Old, cheap, and in this repo's measurements very hard
to beat.

**chunk** — A document cut into a passage small enough to retrieve and quote.
This corpus holds 17,387 of them.

**corpus** — The whole collection of documents being searched.

**dense retrieval** — Search by meaning rather than by words: each passage is
turned into a list of numbers, and the closest numbers win.

**embedder / embedding** — The model that turns text into that list of numbers,
and the numbers themselves.

**hashed embedder** — A cheap stand-in that produces those numbers by hashing
word features instead of running a neural network. This repo's default, and in
the measured results it beats a real neural embedder on its own.

**hybrid** — Combining word-counting and dense retrieval into one ranking.

**fusion weight (w) / boost** — The dial controlling how much of the hybrid
score comes from dense versus BM25, and an extra bump for exact EIP-number
matches. Both are fitted on the training split only.

**graph layer** — An addition that follows links between documents to pull in
related ones. Measured here as slightly *worse* than plain hybrid.

**reranker** — A second pass that re-orders the top results using a more
expensive model. The only component in this repo that clearly earns its place.

**RRF — Reciprocal Rank Fusion** — A standard way to merge two ranked lists
using each item's position rather than its score.

**sparse retrieval** — Search by words, i.e. BM25.

## Scoring metrics

**docset@k** — Whether the full *set* of documents needed for a multi-part
question all appeared in the top k. Harder than R@k.

**MRR@10 — Mean Reciprocal Rank** — Averages 1/(position of the first correct
result), looking only at the top 10. Rewards putting the right answer first.

**multi-hop** — A question needing two or more documents combined to answer.

**nDCG@10 — normalised Discounted Cumulative Gain** — A ranking score over the
top 10 that gives more credit the higher a correct result appears. Unlike R@k it
notices *order*, which is why it moved in this repo's embedder comparison when
R@5 did not.

**R@k — Recall at k** — The share of questions where a correct document appeared
anywhere in the top k results. R@5 is this repo's headline measure.

**tier** — A difficulty band the test questions are grouped into.

## Statistics

**bootstrap** — Estimating uncertainty by resampling the data thousands of times
and watching how much the answer moves.

**CI** — Two meanings:
1. **Confidence interval** — a range of plausible values, here always 95%. If it
   spans zero, the difference is "not distinguishable".
2. **Continuous Integration** — the automated checks on every change.

**gate** — A pass/fail threshold the evaluation must clear, e.g. test R@5 ≥ 0.80.

**not distinguishable / better / worse** — The three verdicts this repo reports
for a comparison, based on whether the confidence interval excludes zero.

**train / test split** — Two separate sets of questions. Weights are fitted on
train; test is scored once and never tuned against.

## The subject matter

**EIP — Ethereum Improvement Proposal** — The numbered specification documents
that make up this corpus.

**ERC — Ethereum Request for Comment** — A category of EIP covering application
standards such as token formats.

## Machine learning

**AI / LLM — Large Language Model** — The kind of AI that predicts text.

**RAG — Retrieval-Augmented Generation** — Answering by retrieving documents
first and writing from them, so claims can be traced to a source.

**citation verification** — Checking that a cited chunk really supports the
sentence it is attached to, rather than merely existing.

**token** — The chunk of text a model reads and writes.

## Files, formats and systems

**API — Application Programming Interface** — A network interface for sending
requests to a service.

**CLI — Command-Line Interface** — The terminal commands this tool provides.

**HTTP** — The protocol web requests use.

**JSON** — A structured record format.

**MB** — Megabytes.

**MCP — Model Context Protocol** — A standard way to expose tools to an AI
assistant. This repo ships an MCP server.

**MIT** — The permissive open-source licence this repo uses.

**POST** — The HTTP request type used to send data to a server.
