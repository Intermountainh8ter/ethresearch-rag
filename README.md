# ethrag

Retrieval-augmented Q&A over Ethereum EIPs where **every answer is machine-checked before it is
shown**. Generation is untrusted; a deterministic verifier decides whether a draft is released.

```
question ─► hybrid retrieval (BM25 + dense, RRF) ─► generator (LLM or extractive baseline)
                                                       └─► verifier ─► verified answer | withheld
```

## Why the verifier exists

Spec questions are dominated by constants (`BASE_FEE_MAX_CHANGE_DENOMINATOR = 8`, opcode numbers,
type bytes). A fluent answer with the wrong constant is worse than no answer. `verify_answer`
requires that every sentence has a citation, every citation points at a chunk that was actually
retrieved, and every number / hex value / `2**n` / `ALL_CAPS` identifier in a sentence is a token
of the chunks it cites. Substring matches don't count (`8` is not found inside `58`).

## Results (bundled corpus: 23 EIPs, 497 chunks; 26 hand-written questions)

```
python evals/run_evals.py --show-misses
```

| system | R@1 | R@3 | R@5 | MRR@10 |
|---|---|---|---|---|
| BM25 | 0.577 | 0.808 | **0.885** | 0.698 |
| dense (hashed tf-idf, char trigrams) | 0.654 | 0.731 | 0.731 | 0.706 |
| hybrid (RRF) | 0.654 | 0.731 | 0.769 | 0.728 |

Verifier (answer cites the gold chunk): correct statements accepted 17/17; wrong constants rejected
15/15; uncited sentence rejected; fabricated citation rejected.

Read this honestly:
- **Hybrid does not beat BM25 at R@5 here.** The "dense" side is a dependency-free hashed baseline,
  not a neural embedder, and it drags the fusion down on type-byte questions. I did not tune fusion
  weights on 26 questions, since that would overfit. The `Embedder` protocol exists so a real model
  (`SentenceTransformerEmbedder`, not exercised in CI) can be swapped in and re-measured.
- 26 questions is a smoke-test sized set; differences of a few points are noise.
- The verifier tests were built from synthetic "correct vs. wrong-constant" pairs that I wrote.
  A wrong value that also appears as a token in the cited chunk would pass (e.g. a chunk that
  mentions both `0xEF` and `0xFE`). Paraphrased claims with no checkable token are not verified.
- Passing verification means *supported by the cited text*, not *relevant to the question*. The
  no-LLM extractive baseline picks sentences by keyword overlap and often verifies yet answers a
  different question. Answer-quality evals need an LLM judge or human labels and are not included.

## Corpus

`data/eips/*.md` are unmodified copies from [ethereum/EIPs](https://github.com/ethereum/EIPs)
(CC0 1.0). Add more files and re-run the evals.

## Usage

```
pip install -e ".[dev,llm]"
pytest -q
ethrag "What opcode number does BASEFEE use?"          # extractive baseline, no API key
ANTHROPIC_API_KEY=... ethrag --llm "How many field elements are in a blob?"
```
Exit code is 2 when the answer could not be verified. `ETHRAG_MODEL` overrides the model id.
The Anthropic path is covered only by a scripted stand-in in tests.
