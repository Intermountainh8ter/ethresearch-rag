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
