# Evaluation

Everything below is produced by two offline, deterministic scripts and can be re-run in about a
minute on a laptop:

```
python evals/run_evals.py          --min-test-recall5 0.80      # retrieval  -> evals/results.json
python evals/run_verifier_eval.py                               # verifier   -> evals/verifier_results.json
python evals/build_questions.py --check                         # is evals/questions.jsonl up to date?
```

The numbers pasted here are copied from those runs. If a table disagrees with a fresh run, the
run wins and the table is a bug.

## 1. The question set

`evals/questions.jsonl` has **504 questions** over the 901-document corpus
([CORPUS.md](CORPUS.md)): 317 `constant`, 133 `concept`, 54 `multihop`, spanning 228 gold
documents. Each record carries the question, a gold document and a `must_contain` needle, a
reference `answer`, usually a `wrong` answer, a `tier`, an `origin` and its `split`.

It is **synthetic**. Nobody asked these questions; they were generated from the corpus text and
then checked by program. That makes the gold labels cheap and reproducible and it also means the
set is biased toward questions whose answer is a literal in a table, a code block or a sentence
(see section 7).

| origin | what | count |
|---|---|---|
| seed | hand-written (with an AI assistant) constant, concept and multi-hop questions about well-known EIPs, kept only if they validate | 231 |
| mined | constants found by `ethrag.constants.mine_constants` in table rows, code lines and short prose of the corpus itself, asked in three tiers: exact identifier + document (`exact-id`), identifier only (`name-only`), identifier spelled out as words + document (`words`) | 173 |
| mined | multi-hop pairs: two constants from two documents linked by `requires` or by an inline `EIP-N` mention; `both-named` or `second-by-id` (the second document is reachable only through its constant's name) | 30 |
| mined | "purpose" questions (`What does ERC-N propose?`) whose answer is the first sentence of the Abstract | 70 |

Mined questions are templated, so they share phrasing; the seeds are the only questions written
the way a person would phrase them. Treat the `seed-*` rows of the per-tier table as the closest
thing here to real usage.

### Gold labels

A **gold chunk** of a question is any chunk of the gold document whose text contains the
`must_contain` needle (case-insensitive). A multi-hop question has one needle per gold document
and its gold chunk set is the union. R@k asks whether *any* gold chunk is in the top k.

### Programmatic validation (`ethrag.evalgen.validate_question`)

`tests/test_questions.py` re-runs this on every test run and `build_questions.py --check`
fails if the committed file is stale. A question is kept only if

1. every needle occurs in a chunk of its gold document;
2. the reference `answer`, citing the best gold chunk(s), is accepted by the token-level verifier;
3. the `wrong` answer, citing the same chunks, is **not** accepted;
4. for constant and multi-hop questions the question text does not already contain every claim
   token of the answer;
5. the question ends in `?`; ids are unique.

Seeds whose own `wrong` answer fails (3) are repaired with a new perturbation (`8 -> 16`,
`0x03 -> 0x04`, `2**17 -> 2**18`, keeping the notation) or dropped. Stage (3) is why the `wrong`
class is not informative about the verifier (section 6).

## 2. Train / test split

The split is **stored in the file** (`"split": "train" | "test"`), written by
`ethrag.evalgen.assign_splits` with seed 20240613 and target 30% test.

* Single-hop questions are split **by gold document**: a document never appears on both sides, so
  a reranker cannot memorise "EIP-1559 chunks are good".
* Multi-hop questions span two documents and are assigned individually, so their documents may
  also appear in train singles.

Result: **353 train / 151 test**; the test split has 93 constant, 42 concept and 16 multi-hop
questions (38 multi-hop in train). 16 multi-hop questions is small: doc-set recall there moves
by 6 points per question.

## 3. What was tuned on which split

| knob | how it was chosen | split |
|---|---|---|
| fusion weight `w` (BM25 share of the RRF credit) and EIP-number `boost` | grid, 11 weights x 5 boosts; selection restricted to `0 < w < 1` because a "hybrid" must use both rankers; ties prefer a smaller boost | train |
| reranker weights | logistic regression on the fused top-50 candidates of train questions; L2 strength by 3-fold cross-validation **grouped by gold document** inside train | train |
| BM25 query stop-word removal | one experiment: R@5 0.781 -> 0.818 on train | train (see the note below) |
| verifier `min_coverage`, `min_relevance`, `min_novelty`, strict negation | grid of 16 settings, objective TPR minus pooled FPR | train |

Chosen: `w* = 0.5`, `boost* = 4.0`; reranker L2 = 0.01; verifier `min_coverage 0.7`,
`min_relevance 0.1`, `min_novelty 0` (off), non-strict negation. These are the shipped defaults.

**Leakage controls**

* the reranker's 20 features (`ethrag.rerank.FEATURE_NAMES`) contain no document id, no question id
  and no text of the gold answer; they are fixed in code before any fit;
* document-level split, so doc priors cannot leak;
* the corpus and index never see the questions; the gold text is of course *in* the corpus, as in
  any retrieval benchmark;
* no model sits in the loop; generation of questions, validation and scoring are plain Python.

**What is not controlled, honestly**

* The architecture, the feature list, the verifier rules (binding window, meta words in the
  relevance check, code lines as negation sources, all-owners binding) were developed while
  looking at output from **both** splits, including failing TEST cases. Only the numeric knobs
  above are train-only. TEST numbers are therefore optimistic estimates, not a blind hold-out.
* The BM25 stop-word experiment printed test numbers too (R@5 0.788 -> 0.788, nDCG@10 +0.014).
  The decision rests on the train gain but I saw the test result.
* Templated questions share phrasing across the split.
* The "purpose" questions always point at an Abstract chunk, and mined constants at table rows or
  code lines. The reranker learns exactly those priors (largest weights after the lexical ones:
  `sec_abstract`, `kind_table`). Part of its gain is that regularity of the generator, not
  retrieval skill. The sensitivity table below removes the purpose tier.

## 4. Retrieval results

Metrics: **R@k** fraction of questions with a gold chunk in the top k; **MRR@10**; **nDCG@10**
with binary relevance against the whole gold chunk set; **docset@k** (multi-hop only) fraction
whose top k covers *every* gold document. Systems:

* `bm25` - field-boosted BM25, title and section terms folded in, query stop words dropped;
* `dense(hashed)` - hashed word + character-trigram tf-idf, cosine (a dependency-free stand-in
  for a neural embedder, **not** a semantic model);
* `hybrid` - weighted RRF of the two plus the EIP-number boost, `w* = 0.5`, `boost* = 4.0`;
* `hybrid+rerank` - the logistic reranker over the fused top 50;
* `hybrid+graph` - budget-neutral: top-(k-e) hybrid hits plus e documents linked by `requires`
  edges, e = 2 for k >= 5.

**TEST (n = 151)**

| system | R@1 | R@3 | R@5 | R@10 | MRR@10 | nDCG@10 |
|---|---|---|---|---|---|---|
| bm25 | 0.424 | 0.702 | 0.788 | 0.901 | 0.579 | 0.622 |
| dense(hashed) | 0.325 | 0.483 | 0.609 | 0.742 | 0.444 | 0.467 |
| hybrid | 0.437 | 0.715 | 0.815 | 0.940 | 0.594 | 0.649 |
| hybrid+rerank | 0.695 | 0.894 | 0.934 | 0.987 | 0.801 | 0.814 |
| hybrid+graph | 0.437 | 0.715 | 0.815 | 0.901 | 0.590 | 0.635 |

**TRAIN (n = 353)** - what the tuning saw

| system | R@1 | R@3 | R@5 | R@10 | MRR@10 | nDCG@10 |
|---|---|---|---|---|---|---|
| bm25 | 0.518 | 0.725 | 0.819 | 0.918 | 0.640 | 0.676 |
| dense(hashed) | 0.442 | 0.586 | 0.660 | 0.756 | 0.533 | 0.543 |
| hybrid | 0.513 | 0.737 | 0.847 | 0.946 | 0.648 | 0.684 |
| hybrid+rerank | 0.717 | 0.895 | 0.958 | 0.980 | 0.814 | 0.832 |
| hybrid+graph | 0.513 | 0.737 | 0.847 | 0.901 | 0.643 | 0.670 |

**Does hybrid beat BM25? Not convincingly.** On TEST the hybrid is 2.6 points ahead of raw BM25 at
R@5, with a paired-bootstrap 95% interval of [-0.020, +0.073]: not distinguishable. The ablation
isolates why it looks ahead at all: BM25 given the *same* EIP-number boost is just as good.

| system | R@1 | R@3 | R@5 | R@10 | MRR@10 | nDCG@10 |
|---|---|---|---|---|---|---|
| bm25+eip-boost | 0.497 | 0.735 | 0.821 | 0.927 | 0.642 | 0.675 |
| dense+eip-boost | 0.450 | 0.656 | 0.768 | 0.901 | 0.581 | 0.622 |
| hybrid(rrf w=0.5, no boost) | 0.351 | 0.636 | 0.689 | 0.821 | 0.496 | 0.540 |
| hybrid(rrf w=0.5, boost=1) | 0.437 | 0.715 | 0.815 | 0.934 | 0.593 | 0.649 |
| hybrid(w*, restrict to named EIP) | 0.437 | 0.715 | 0.815 | 0.940 | 0.594 | 0.649 |

`bm25+eip-boost` is level with the hybrid at R@5 (0.821 vs 0.815) and ahead at nDCG@10 (0.675 vs
0.649; interval of the difference [-0.052, +0.002]). The grid search on train also scored w = 1.0
(pure BM25 + boost) at least as high as any fused cell; only the rule "a hybrid must fuse"
selected w = 0.5. The dense side here is a hashed lexical baseline; a real embedding model is
measured once with `--embedder neural` (`BAAI/bge-small-en-v1.5`; see README and `results/2026-10-14/`): no gain in R@5 on test. `SentenceTransformerEmbedder` stays optional and is not exercised in CI.

Paired bootstrap, 2000 resamples, TEST:

| comparison | R@5 difference | 95% interval | verdict |
|---|---|---|---|
| hybrid - bm25 | +0.026 | [-0.020, +0.073] | not distinguishable |
| hybrid - bm25+eip-boost | -0.007 | [-0.046, +0.033] | not distinguishable |
| hybrid+rerank - hybrid | +0.119 | [+0.066, +0.172] | better |
| hybrid+graph - hybrid | +0.000 | [+0.000, +0.000] | no effect on R@5; nDCG@10 -0.014, interval [-0.023, -0.006]: worse |

**The reranker is the one clear win** (R@5 0.815 -> 0.934, nDCG@10 0.649 -> 0.814). Sensitivity:
without the synthetic purpose tier (n = 131) it is still ahead, by less.

| system | R@1 | R@3 | R@5 | R@10 | MRR@10 | nDCG@10 |
|---|---|---|---|---|---|---|
| bm25 | 0.481 | 0.786 | 0.847 | 0.931 | 0.639 | 0.670 |
| dense(hashed) | 0.366 | 0.534 | 0.664 | 0.794 | 0.490 | 0.508 |
| hybrid | 0.504 | 0.779 | 0.878 | 0.977 | 0.659 | 0.703 |
| hybrid+rerank | 0.710 | 0.885 | 0.931 | 0.985 | 0.808 | 0.813 |
| hybrid+graph | 0.504 | 0.779 | 0.878 | 0.931 | 0.654 | 0.687 |

**Graph expansion did not help** on this set: R@5 is unchanged and a slot spent on a neighbour
is a slot not spent on a lower-ranked hit. Most multi-hop questions here name both documents, so
there was little for the graph to add; that says little about questions that name only one.

**Multi-hop doc-set recall** (every gold document in the top k):

| system | n | docset@5 | docset@10 | R@10 |
|---|---|---|---|---|
| train: bm25 | 38 | 0.605 | 0.816 | 1.000 |
| train: dense(hashed) | 38 | 0.553 | 0.737 | 0.921 |
| train: hybrid | 38 | 0.579 | 0.684 | 0.947 |
| train: hybrid+rerank | 38 | 0.737 | 0.974 | 0.974 |
| train: hybrid+graph | 38 | 0.579 | 0.763 | 0.921 |
| test: bm25 | 16 | 0.500 | 0.750 | 0.938 |
| test: dense(hashed) | 16 | 0.562 | 0.688 | 0.812 |
| test: hybrid | 16 | 0.625 | 0.688 | 1.000 |
| test: hybrid+rerank | 16 | 0.812 | 0.938 | 1.000 |
| test: hybrid+graph | 16 | 0.625 | 0.625 | 0.938 |

**By tier, TEST R@5** (the `seed-*` rows are the hand-written questions):

| system | n | bm25 | hybrid | hybrid+rerank |
|---|---|---|---|---|
| both-named | 6 | 1.000 | 1.000 | 1.000 |
| exact-id | 12 | 1.000 | 1.000 | 1.000 |
| name-only | 6 | 0.833 | 0.833 | 0.833 |
| purpose | 20 | 0.400 | 0.400 | 0.950 |
| second-by-id | 2 | 1.000 | 1.000 | 1.000 |
| seed-concept | 22 | 0.591 | 0.682 | 0.773 |
| seed-constant | 55 | 0.873 | 0.891 | 0.964 |
| seed-multihop | 8 | 0.875 | 0.875 | 0.875 |
| words | 20 | 0.900 | 0.950 | 1.000 |

Reading it: identifier lookups are nearly solved by BM25; the hard cases are concept questions
phrased without the identifier (`seed-concept`, `purpose`). `name-only` (n = 6) and `second-by-id`
(n = 2) are too small to conclude anything.

## 5. Verifier results

`ethrag.verifyeval` builds a family of **cases** from every question that has a reference answer,
all citing the gold chunk(s) unless stated, and counts how often the verifier accepts each class.
TPR is the accept rate of `correct`; the FPR of a negative class is its accept rate; the pooled
FPR is their mean.

| class | what it is | should be |
|---|---|---|
| `correct` | the reference answer | accepted |
| `wrong` | a plausible different value that is *not* in the chunk | rejected |
| `swap` | the value replaced by **another literal from the same cited chunk** | rejected |
| `negated` | the correct answer with `is` -> `is not` | rejected |
| `vague` | a fluent sentence with no checkable token ("as described above") | rejected |
| `echo` | the question restated as a sentence, no value given | rejected |
| `wrong_citation` | the correct answer citing another chunk of the same document | rejected |
| `wrong_q` | a correct, properly cited answer to a *different* question | rejected |
| `uncited` / `fake_citation` | no citation / a chunk id that was never retrieved | rejected |

**TEST, shipped defaults, with Wilson intervals**

| class | accepted / n | rate | 95% Wilson |
|---|---|---|---|
| correct | 147 / 151 | 0.974 | [0.934, 0.990] |
| wrong | 0 / 117 | 0.000 | [0.000, 0.032] |
| swap | 19 / 103 | 0.184 | [0.121, 0.270] |
| negated | 2 / 113 | 0.018 | [0.005, 0.062] |
| vague | 0 / 151 | 0.000 | [0.000, 0.025] |
| echo | 33 / 151 | 0.219 | [0.160, 0.291] |
| wrong_citation | 1 / 151 | 0.007 | [0.001, 0.037] |
| wrong_q | 6 / 151 | 0.040 | [0.018, 0.084] |
| uncited | 0 / 151 | 0.000 | [0.000, 0.025] |
| fake_citation | 0 / 151 | 0.000 | [0.000, 0.025] |

**What each check buys (TEST).** Rows are cumulative: each adds one gate to the row above.

| configuration | correct | wrong | swap | negated | vague | echo | wrong_citation | wrong_q | uncited | fake_citation | pooled FPR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tokens only | 1.000 | 0.043 | 0.961 | 1.000 | 1.000 | 0.921 | 0.073 | 1.000 | 0.000 | 0.000 | 0.555 |
| + binding, units, ranges | 1.000 | 0.000 | 0.204 | 1.000 | 1.000 | 0.874 | 0.073 | 1.000 | 0.000 | 0.000 | 0.461 |
| + content coverage 0.7 | 0.987 | 0.000 | 0.194 | 0.982 | 0.000 | 0.424 | 0.007 | 0.987 | 0.000 | 0.000 | 0.288 |
| + negation | 0.987 | 0.000 | 0.194 | 0.018 | 0.000 | 0.424 | 0.007 | 0.987 | 0.000 | 0.000 | 0.181 |
| + question relevance 0.1 (default) | 0.974 | 0.000 | 0.184 | 0.018 | 0.000 | 0.219 | 0.007 | 0.040 | 0.000 | 0.000 | 0.052 |
| + novelty 0.25 (optional) | 0.868 | 0.000 | 0.155 | 0.018 | 0.000 | 0.053 | 0.007 | 0.040 | 0.000 | 0.000 | 0.030 |

* *Tokens only* is the v0.1 verifier: it accepts almost every `swap` (0.961), every `negated`
  answer and every `wrong_q` answer.
* *Binding, units, ranges* is what stops the `swap` class (0.961 -> 0.204): a value must sit on
  the same line as the name it is claimed for, or on a line with no name of its own.
* *Content coverage* removes the `vague` paraphrases and most `wrong_citation` cases.
* *Negation* takes `negated` from 0.982 to 0.018.
* *Question relevance* is the only thing that rejects `wrong_q` (0.987 -> 0.040): an answer can
  be perfectly supported by its citation and still answer a different question.
* *Novelty* (answer must not mostly restate the question) cuts `echo` to 0.053 but costs about 11 points
  of TPR (0.868), so it is optional (`--min-novelty`) and off.

The same table on TRAIN shows the same pattern (TPR 0.972, pooled FPR 0.040):

| configuration | correct | wrong | swap | negated | vague | echo | wrong_citation | wrong_q | uncited | fake_citation | pooled FPR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tokens only | 1.000 | 0.083 | 0.920 | 1.000 | 1.000 | 0.907 | 0.065 | 1.000 | 0.000 | 0.000 | 0.553 |
| + binding, units, ranges | 1.000 | 0.000 | 0.109 | 1.000 | 1.000 | 0.892 | 0.065 | 1.000 | 0.000 | 0.000 | 0.452 |
| + content coverage 0.7 | 0.986 | 0.000 | 0.105 | 0.989 | 0.000 | 0.377 | 0.006 | 0.975 | 0.000 | 0.000 | 0.272 |
| + negation | 0.986 | 0.000 | 0.105 | 0.029 | 0.000 | 0.374 | 0.006 | 0.975 | 0.000 | 0.000 | 0.165 |
| + question relevance 0.1 (default) | 0.972 | 0.000 | 0.101 | 0.026 | 0.000 | 0.204 | 0.006 | 0.020 | 0.000 | 0.000 | 0.040 |
| + novelty 0.25 (optional) | 0.878 | 0.000 | 0.101 | 0.026 | 0.000 | 0.105 | 0.006 | 0.020 | 0.000 | 0.000 | 0.029 |

**How much to trust it**

* TPR 0.974 means 4 of 151 correct reference answers are withheld. The reference answers are
  terse and lexically close to their source. **LLM-written paraphrases will be rejected more
  often**, mainly by `min_coverage`; none was measured here (no API key in this environment).
  `--min-coverage 0.5` is the knob to loosen first.
* `wrong` is 0 by construction (validation stage 3) and carries no information about the verifier.
  `uncited` and `fake_citation` are trivial. The informative rows are `swap`, `echo`, `wrong_q`,
  `negated`, `wrong_citation`.
* Intervals are wide at this sample size (`swap` 0.184, [0.121, 0.270]).

**End to end, offline extractive answerer, TEST (n = 151).** The no-LLM baseline answers
`NAME is value` from a definition line when the question names or spells out a constant and
otherwise returns the best-matching sentence:

| | without the relevance check | with it (default) |
|---|---|---|
| answers verified | 146 / 151 | 138 / 151 |
| of those, containing the reference value or sentence | 75 | 75 |
| answers to *another* question that still verify | 147 / 147 | 8 / 147 |

By type (with the default): constant 62 of 93 contain the reference, concept 9 of 42, multi-hop
4 of 16. "Verified" is therefore **not** "correct": 63 of the 138 verified answers are grounded
text that is not the reference (for concept questions the reference match is strict: 70% of the
reference sentence's words). The extractive answerer is a baseline, not the product; the point of
this table is that the verifier alone does not make a bad generator good.

## 6. What the verifier cannot catch

"Verified" means *every checkable token and every sentence is supported by the cited chunk text*.
It does not mean true, current or responsive. Specifically:

1. **Presence is not binding when there is no name to bind to.** EIP-3541 forbids new code
   starting with `0xEF`; a chunk that also mentions `0xFE` (the INVALID opcode) supports the false
   sentence "code starting with 0xFE is rejected": every token is in the chunk, no identifier ties
   the number to a subject. `tests/test_verify.py::test_blind_spot_0xef_vs_0xfe` pins this
   behaviour. Naming the parameter makes the check bite (`BAD_PREFIX is 0xFE` is rejected when
   the chunk says `BAD_PREFIX = 0xEF`).
2. **Two values on one line cannot be told apart.** `BASE_FEE = 8, ELASTICITY = 2` supports
   "BASE_FEE is 2". 19 of 103 TEST `swap` cases get through for this reason (and `CREATE2 ... takes
   4 stack arguments` supports "CREATE2 is opcode 4").
3. **Claims without a token are only checked by word overlap.** A prose sentence with no number,
   hex or identifier is accepted when 70% of its content words occur in the cited chunks. A
   paraphrase that reverses the meaning with different words passes (`echo`: 22% on TEST).
4. **Negation is lexical.** It looks for `not/never/cannot/no longer/neither/nor/n't/without/
   disallow/forbid/prohibit/unable` in the best-matching source sentence (code lines and table
   rows count when no prose sentence matches). Reversals by antonym ("allowed" -> "forbidden" is
   caught only if `forbidden` is a negation word in the source) are not understood.
5. **Relevance is lexical.** "How large is the payload timeliness committee?" answered with
   `PTC_SIZE is 2**9` has no word in common with the question and is withheld; a related-but-wrong
   sentence that reuses the question's words is accepted. 6 of 151 `wrong_q` answers pass.
6. **The source can be wrong or stale.** The corpus includes `Draft`, `Stagnant` and `Withdrawn`
   documents (317 stagnant, 32 withdrawn). A verified answer is faithful to the cited text, not to
   mainnet. The status is in the chunk metadata and the reranker has a status prior, but nothing
   warns the user.
7. **Units and ranges are a short list.** ETH/gwei/wei, bytes/KiB/MiB..., seconds-days, gas,
   blocks, slots, epochs, blobs, bits; integer quantities only; ranges only between integers.
   A unit mismatch is reported only when the chunk gives the *same number* with another unit.
8. **Facts spread over chunks.** A sentence may cite several chunks and a token is supported if
   any of them has it; the sentence as a whole is not checked for composition.
9. **Arithmetic is not evaluated.** "2 x 8 = 16" is supported only if `16` is in the chunk.

## 7. Limits of this evaluation

* Synthetic questions, templated phrasing, 151 test questions: intervals are wide and several
  differences above are inside the noise. Where an interval is shown, read the interval.
* Gold is "chunk contains the needle". A different chunk that also answers the question counts as
  a miss, and a chunk that merely contains the needle counts as a hit.
* The constant questions measure lookup of literals. They say little about explanatory questions,
  which are exactly where lexical retrieval and a lexical verifier are weakest (`seed-concept`).
* No LLM-generated answers were evaluated. `evals/run_judge.py` exists for that (LLM-as-judge,
  needs an API key, tested only against a scripted judge); no number from it is reported here.
* Retrieval was evaluated with the hashed dense baseline only.

## 8. Reproducing and extending

* `make eval-gate` runs the two gates CI runs: TEST hybrid R@5 >= 0.80 (measured 0.815) and
  verifier TEST TPR >= 0.93, pooled FPR <= 0.08 (measured 0.974 and 0.052).
* To add questions, append to `evals/seed_questions.jsonl` and run
  `python evals/build_questions.py`. The split is recomputed deterministically from the seed over
  the whole set, so adding questions can move existing documents between train and test: re-run
  both eval scripts afterwards and commit `evals/questions.jsonl`, `evals/results.json`,
  `evals/verifier_results.json` and `evals/artifacts/reranker.json` together.
* A question that the verifier cannot tell from its `wrong` answer is dropped at build time and
  listed in the build report.
