# Judge agreement on Base model rankings

Scores are sample means, ranked in descending order at the precision stored in
the summaries and displayed to three decimal places. Spearman rho is the Pearson
correlation of the ranks, using average ranks for ties. This measures agreement
in model rankings, not agreement on individual judgments. A correlation of one
does not imply equal scores.

## Spearman rank correlation

| Benchmark | Metric | Qwen vs GLM-5.2 Spearman ρ | Metric source |
|---|---|---:|---|
| SpecGap | Missing-Constraint Discovery F1 | 1.0000 | LLM judge |
| SpecGap | Question Quality | 0.9643 | LLM judge |
| SpecGap | Location F1 | 1.0000 | Rule-based |
| SilentSwap | Localization Score | 1.0000 | Rule-based |
| SilentSwap | Location Correct | 1.0000 | LLM judge |
| SilentSwap | Code Correct | 0.9643 | LLM judge |
| FeedbackTrace | Verification Point Alignment | 0.8929 | LLM judge |
| FeedbackTrace | Evidence Location Score | 0.9643 | LLM judge |
| FeedbackTrace | Evidence Hit Rate | 1.0000 | Rule-based |

## SpecGap

### Missing-Constraint Discovery F1

| Qwen Rank | Model | Score | GLM-5.2 Rank | Model | Score |
|---:|---|---:|---:|---|---:|
| 1 | K3 | 0.485 | 1 | K3 | 0.502 |
| 2 | Sol | 0.449 | 2 | Sol | 0.462 |
| 3 | Sonnet 5 | 0.318 | 3 | Sonnet 5 | 0.338 |
| 4 | Terra | 0.268 | 4 | Terra | 0.282 |
| 5 | Luna | 0.234 | 5 | Luna | 0.265 |
| 6 | Flash | 0.085 | 6 | Flash | 0.097 |
| 7 | Pro | 0.060 | 7 | Pro | 0.089 |

The two judges produce identical model rankings but different scores.

### Question Quality

| Qwen Rank | Model | Score | GLM-5.2 Rank | Model | Score |
|---:|---|---:|---:|---|---:|
| 1 | K3 | 0.930 | 1 | K3 | 0.924 |
| 2 | Sol | 0.924 | 2 | Sol | 0.922 |
| 3 | Sonnet 5 | 0.771 | 3 | Sonnet 5 | 0.799 |
| 4 | Terra | 0.677 | 4 | Terra | 0.700 |
| 5 | Luna | 0.618 | 5 | Luna | 0.636 |
| 6 | Flash | 0.226 | 6 | Pro | 0.232 |
| 7 | Pro | 0.186 | 7 | Flash | 0.229 |

Flash and Pro exchange sixth and seventh place; the other ranks agree.

### Location F1

| Qwen Rank | Model | Score | GLM-5.2 Rank | Model | Score |
|---:|---|---:|---:|---|---:|
| 1 | Sol | 0.501 | 1 | Sol | 0.501 |
| 2 | K3 | 0.497 | 2 | K3 | 0.497 |
| 3 | Sonnet 5 | 0.368 | 3 | Sonnet 5 | 0.368 |
| 4 | Terra | 0.309 | 4 | Terra | 0.309 |
| 5 | Luna | 0.305 | 5 | Luna | 0.305 |
| 6 | Pro | 0.138 | 6 | Pro | 0.138 |
| 7 | Flash | 0.114 | 7 | Flash | 0.114 |

Scores and ranks agree exactly for this rule-based metric.

## SilentSwap

### Localization Score

| Qwen Rank | Model | Score | GLM-5.2 Rank | Model | Score |
|---:|---|---:|---:|---|---:|
| 1 | K3 | 0.555 | 1 | K3 | 0.555 |
| 2 | Sol | 0.464 | 2 | Sol | 0.464 |
| 3 | Terra | 0.417 | 3 | Terra | 0.417 |
| 4 | Luna | 0.403 | 4 | Luna | 0.403 |
| 5 | Sonnet 5 | 0.294 | 5 | Sonnet 5 | 0.294 |
| 6 | Flash | 0.262 | 6 | Flash | 0.262 |
| 7 | Pro | 0.136 | 7 | Pro | 0.136 |

Scores and ranks agree exactly for this rule-based metric.

### Location Correct

| Qwen Rank | Model | Score | GLM-5.2 Rank | Model | Score |
|---:|---|---:|---:|---|---:|
| 1 | K3 | 0.430 | 1 | K3 | 0.590 |
| 2 | Sol | 0.320 | 2 | Sol | 0.535 |
| 3 | Terra | 0.300 | 3 | Terra | 0.525 |
| 4 | Luna | 0.275 | 4 | Luna | 0.385 |
| 5 | Sonnet 5 | 0.135 | 5 | Sonnet 5 | 0.240 |
| 6 | Flash | 0.100 | 6 | Flash | 0.230 |
| 7 | Pro | 0.025 | 7 | Pro | 0.075 |

The two judges produce identical model rankings but different scores.

### Code Correct

| Qwen Rank | Model | Score | GLM-5.2 Rank | Model | Score |
|---:|---|---:|---:|---|---:|
| 1 | K3 | 0.675 | 1 | K3 | 0.705 |
| 2 | Sol | 0.605 | 2 | Terra | 0.595 |
| 3 | Terra | 0.570 | 3 | Sol | 0.590 |
| 4 | Luna | 0.290 | 4 | Luna | 0.245 |
| 5 | Sonnet 5 | 0.210 | 5 | Sonnet 5 | 0.235 |
| 6 | Flash | 0.120 | 6 | Flash | 0.130 |
| 7 | Pro | 0.010 | 7 | Pro | 0.025 |

Sol and Terra exchange second and third place; the other ranks agree.

## FeedbackTrace

### Verification Point Alignment

| Qwen Rank | Model | Score | GLM-5.2 Rank | Model | Score |
|---:|---|---:|---:|---|---:|
| 1 | Terra | 0.470 | 1 | Terra | 0.555 |
| 2 | Sol | 0.425 | 2 | Sol | 0.535 |
| 3 | Luna | 0.410 | 3 | Luna | 0.510 |
| 4 | Flash | 0.390 | 4 | K3 | 0.475 |
| 5 | K3 | 0.365 | 5 | Sonnet 5 | 0.455 |
| 6 | Sonnet 5 | 0.300 | 6 | Flash | 0.440 |
| 7 | Pro | 0.200 | 7 | Pro | 0.315 |

The top three agree; differences concern the relative positions of Flash, K3, and Sonnet 5.

### Evidence Location Score

| Qwen Rank | Model | Score | GLM-5.2 Rank | Model | Score |
|---:|---|---:|---:|---|---:|
| 1 | Terra | 0.590 | 1 | Terra | 0.585 |
| 2 | Luna | 0.545 | 2 | Luna | 0.545 |
| 3 | Sol | 0.540 | 3 | Sol | 0.525 |
| 4 | K3 | 0.525 | 4 | K3 | 0.510 |
| 5 | Sonnet 5 | 0.465 | 5 | Flash | 0.445 |
| 6 | Flash | 0.460 | 6 | Sonnet 5 | 0.435 |
| 7 | Pro | 0.345 | 7 | Pro | 0.345 |

Flash and Sonnet 5 exchange fifth and sixth place; the other ranks agree.

### Evidence Hit Rate

| Qwen Rank | Model | Score | GLM-5.2 Rank | Model | Score |
|---:|---|---:|---:|---|---:|
| 1.5 | Sol | 0.550 | 1.5 | Sol | 0.550 |
| 1.5 | K3 | 0.550 | 1.5 | K3 | 0.550 |
| 3.5 | Flash | 0.540 | 3.5 | Flash | 0.540 |
| 3.5 | Terra | 0.540 | 3.5 | Terra | 0.540 |
| 5 | Luna | 0.530 | 5 | Luna | 0.530 |
| 6 | Sonnet 5 | 0.480 | 6 | Sonnet 5 | 0.480 |
| 7 | Pro | 0.340 | 7 | Pro | 0.340 |

Scores and ranks agree exactly. Sol and K3 have average rank 1.5; Flash and Terra have average rank 3.5. This is a rule-based metric.
