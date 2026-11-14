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
