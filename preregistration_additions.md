## 2026-10-27T00:24:32Z relevance-false-withhold proposal:834be265f4bc9c68763863f1c82a365866b14ce39e38d1a16b11579d5be6f80c

**Hypothesis.** Among TEST questions whose offline extractive answer contains the reference value, raising min_relevance from the shipped 0.1 to 0.25 (all other settings default) wrongly withholds at least 0.05 of them, the shipped default withholds a smaller share, and relevance-type issues account for the majority of the added withholds.

**Metric.** /false_withhold_rate/relevance_0_25 in results/67-relevance-false-withhold.json.

**Expected.** At least 0.05 and greater than the default-setting rate.

**Falsifier.** The false-withhold rate at min_relevance 0.25 is below 0.05, or is not greater than the default-setting rate, on the TEST split.

## 2026-10-27T00:28:02Z strictness-ladder-false-withhold proposal:4a3a78094f09b3b71296181c50ecd56cdf438ed694465b0b1b1589b1eb02cc32

**Hypothesis.** On TRAIN and TEST questions (pooled) whose offline extractive answer contains the reference value, a combined strict setting (min_coverage 0.95, min_novelty 0.3, min_relevance 0.25) wrongly withholds at least 0.05 of them, the shipped default withholds less than 0.01, and novelty-type issues account for the majority of the added withholds.

**Metric.** /false_withhold_rate/strict_combined in results/68-strictness-ladder-false-withhold.json.

**Expected.** At least 0.05 for the combined setting, below 0.01 for the shipped default.

**Falsifier.** The pooled false-withhold rate at the combined strict setting is below 0.05, or the shipped-default rate is 0.01 or higher, or novelty issues are not the majority of added withholds.

## 2026-11-11T10:48:01Z planted-passage-blind-spot proposal:84e81e4ca4cf3cf56aae425a44d08ac0f21ebc519b7f4e4053c83444f4b7caec

**Hypothesis.** On constructed constant questions where the cited chunk is replaced by a planted copy that states a common-misconception value (the stored `wrong` value) next to the correct constant name, the shipped verifier (default settings) accepts the wrong answer in at least 0.50 of cases, because it checks that claims appear in the cited chunk and not that the chunk is authoritative; the same verifier accepts a wrong value in fewer than 0.05 of cases when the wrong value is absent from the cited chunk.

**Metric.** /wrong_accept_rate/planted_agrees_with_memory in results/101-planted-passage-blind-spot.json (control: /wrong_accept_rate/planted_absent).

**Expected.** At least 0.50 for planted-agrees-with-memory, below 0.05 for the planted-absent control.

**Falsifier.** The wrong-accept rate for planted-agrees-with-memory variants is below 0.50, or the planted-absent control rate is 0.05 or higher.

## 2026-11-11T11:08:55Z numeric-substitution proposal:d16f204c06d4f0c1c7a11e29edc6f8ead4920b72d52a7c2cadb42f2986e93a11

**Hypothesis.** On constant questions whose gold chunk contains at least one other numeric token, an answer that states a different in-chunk number for the constant is accepted by the default verifier at a rate below 0.05, while the unaltered control false-withhold rate stays below 0.05; a false-accept rate of 0.05 or more would show a name-binding blind spot.

**Metric.** /numeric_substitution/false_accept_rate in results/110-numeric-substitution.json.

**Expected.** Below 0.05 (name-adjacency binding rejects in-chunk swaps).

**Falsifier.** A Wilson 95% lower bound for the in-chunk swap false-accept rate at or above 0.05, or a point estimate at or above 0.05.

## 2026-11-13T03:25:36Z leave-one-check-out proposal:bdfce43c9051baf4ca98216689d685bf01a0617b6e614c7490a0327ed2317ad3

**Hypothesis.** On the stored answerable questions (train+test pooled, offline extractive drafts that contain the reference value), disabling any single verifier check (unsupported claim, citation, uncited, unit, range, binding, negation; relevance and coverage under the strict_combined setting) lowers the false-withhold rate by at most 0.02 absolute, i.e. no single check dominates the false withholds. A check is "disabled" by ignoring its issue type in the verdict of the same verification run.

**Metric.** /leave_one_out/max_false_withhold_delta in results/135-leave-one-check-out.json.

**Expected.** At most 0.02 (range 0.0 to 0.02).

**Falsifier.** Any single leave-one-check-out arm lowering the false-withhold rate by more than 0.02 absolute with its Wilson interval for the arm rate excluding the full-verifier rate.

## 2026-11-14T03:35:47Z stale-version-blind-spot proposal:86107ca7c35fb435f40fc23bb8fa6825d21d3d04f6e71f7ced1a6af80d50b157

**Hypothesis.** On constructed constant questions, the verifier releases answers citing a superseded-variant chunk (value line preserved, a status line changed to Superseded, all other integer parameters changed) at a rate of at least 0.90, because it checks only that claimed numbers and names appear in the cited chunk, not the document status. The answerable control (unaltered chunk) false-withhold rate stays below 0.10.

**Metric.** /stale_version/pass_rate in results/163-stale-version-blind-spot.json.

**Expected.** At least 0.90 (range 0.90 to 1.0).

**Falsifier.** A stale_version pass_rate below 0.90 on the constructed set.

## 2026-11-14T03:41:00Z quote-shape-blind-spot proposal:2a8dc68e3379863c5978abcd101b5966fed91c8d70057f90bb500e7e7d44620b

**Hypothesis.** On constant questions (simulated answers quoting the cited chunk, default verifier), answers whose quote joins two source spans with an ellipsis are released at a rate of at least 0.90, because the verifier checks numbers and names against the cited chunk and not quote contiguity; mid-sentence truncations that cut off the claimed value are released under 0.10.

**Metric.** /quote_shape/release_rate/ellipsis_join in results/165-quote-shape-blind-spot.json.

**Expected.** At least 0.90 (range 0.90 to 1.0).

**Falsifier.** An ellipsis_join release rate below 0.90 with at least 20 accepted cases, with the exact-quote control release rate at or above 0.90.

## 2026-11-17T14:42:06Z pooled-citation-gap proposal:7ed764b97ab49a1d500d5f3990e7ca43d12fd102dd6bffaa2dba53f9ecbf23af

**Hypothesis.** On constant questions whose must_contain value line also occurs verbatim in a chunk of a different document, an answer citing [gold chunk with the value line removed, other-document chunk with the line] is released by the default verifier at a rate of at least 0.90, because support is pooled across cited chunks; in all released cases the first-cited (asked-about document) chunk lacks the claim, i.e. a per-citation check would reject them. The unaltered gold-only control false-withhold rate stays below 0.10.

**Metric.** /pooled_citation/release_rate in results/374-pooled-citation-gap.json.

**Expected.** At least 0.90 (range 0.90 to 1.0).

**Falsifier.** A release rate below 0.90 with at least 50 questions used, with the control false-withhold rate below 0.10.

## 2026-11-18T15:05:16Z spelling-variants proposal:1dfe53219118f3bf3779e9b3874ccb83dc81452237f259b61b2c617155c95dd6

**Hypothesis.** On constant questions whose must_contain value is an integer, simulated answers that restate the value from the cited gold chunk in an equivalent spelling (decimal, hex, 2**n where a power of two, comma-grouped) are released by the default verifier at a rate of at least 0.95 per spelling, so spelling variation is not a dominant source of false withholds; the exact-spelling control release rate stays at or above 0.90.

**Metric.** /spelling_variants/min_release_rate in results/419-spelling-variants.json.

**Expected.** At least 0.95 (range 0.95 to 1.0).

**Falsifier.** Any spelling variant with at least 20 accepted questions whose release rate is below 0.95 with its Wilson upper bound below 0.95.

## 2026-11-18T15:13:48Z name-variants proposal:0c7e4575bab700ac1952b526a0575fe39d84630bbaf66d803bb320d7dbc9cff0

**Hypothesis.** On constant questions whose must_contain is NAME = <decimal integer> (simulated answers citing the unaltered gold chunk, default verifier), answers that restate NAME as space-separated lowercase words or as camelCase are each released at a rate below 0.50, because the binding check matches the identifier token and not its prose form, while the exact-name control release rate stays at or above 0.90.

**Metric.** /name_variants/release_rate/spaced in results/421-name-variants.json.

**Expected.** Below 0.50 (range 0.0 to 0.5).

**Falsifier.** A spaced-words release rate of 0.50 or more, or an exact-name control release rate below 0.90. Accepted-question count is reported with Wilson bounds; if fewer than 20 are accepted the result is reported as inconclusive, not as support.
