## 2026-10-27T00:24:32Z relevance-false-withhold proposal:834be265f4bc9c68763863f1c82a365866b14ce39e38d1a16b11579d5be6f80c

**Hypothesis.** Among TEST questions whose offline extractive answer contains the reference value, raising min_relevance from the shipped 0.1 to 0.25 (all other settings default) wrongly withholds at least 0.05 of them, the shipped default withholds a smaller share, and relevance-type issues account for the majority of the added withholds.

**Metric.** /false_withhold_rate/relevance_0_25 in results/67-relevance-false-withhold.json.

**Expected.** At least 0.05 and greater than the default-setting rate.

**Falsifier.** The false-withhold rate at min_relevance 0.25 is below 0.05, or is not greater than the default-setting rate, on the TEST split.
