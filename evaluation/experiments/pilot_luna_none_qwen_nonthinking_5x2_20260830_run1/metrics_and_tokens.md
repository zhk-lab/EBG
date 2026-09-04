# Luna 10×3 指标与 Token

AgentLoop Baseline 与 BEG 的 Prediction 均为 GPT-5.6 Luna `reasoning_effort=none`，Judge 均为 Qwen 3.7 `enable_thinking=false`。

每个 benchmark 均评测 10 个样本：`001、007、013、025、037、050、062、075、087、100`。

## SpecGap

### 平均指标变化

| 指标 | 源 Baseline | AgentLoop Baseline | AgentLoop Baseline − 源 | BEG | BEG − AgentLoop Baseline |
|---|---:|---:|---:|---:|---:|
| Semantic match F1 (LLM) | 0.1576 | 0.1438 | -0.0138 | 0.2706 | +0.1267 |
| Question quality (LLM) | 0.5000 | 0.3000 | -0.2000 | 0.7000 | +0.4000 |
| Location F1 (rule-based) | 0.2213 | 0.1438 | -0.0775 | 0.2956 | +0.1517 |

### 预测模型 Token 变化

| Token | 源 Baseline | AgentLoop Baseline | AgentLoop Baseline − 源 | BEG | BEG − AgentLoop Baseline |
|---|---:|---:|---:|---:|---:|
| 调用数 | 19 | 29 | +10 | 30 | +1 |
| 输入 | 228,402 | 499,969 | +271,567 (+118.90%) | 791,935 | +291,966 (+58.40%) |
| 输出 | 39,734 | 2,666 | -37,068 (-93.29%) | 4,090 | +1,424 (+53.41%) |
| 总计 | 268,136 | 502,635 | +234,499 (+87.46%) | 796,025 | +293,390 (+58.37%) |

## SilentSwap

### 平均指标变化

| 指标 | 源 Baseline | AgentLoop Baseline | AgentLoop Baseline − 源 | BEG | BEG − AgentLoop Baseline |
|---|---:|---:|---:|---:|---:|
| Localization score | 0.4315 | 0.4359 | +0.0044 | 0.5446 | +0.1087 |
| Location correct | 0.4000 | 0.3500 | -0.0500 | 0.4000 | +0.0500 |
| Code change correct | 0.4500 | 0.3000 | -0.1500 | 0.4500 | +0.1500 |

### 预测模型 Token 变化

| Token | 源 Baseline | AgentLoop Baseline | AgentLoop Baseline − 源 | BEG | BEG − AgentLoop Baseline |
|---|---:|---:|---:|---:|---:|
| 调用数 | 53 | 35 | -18 | 40 | +5 |
| 输入 | 1,074,817 | 801,499 | -273,318 (-25.43%) | 1,704,346 | +902,847 (+112.64%) |
| 输出 | 95,120 | 11,907 | -83,213 (-87.48%) | 9,180 | -2,727 (-22.90%) |
| 总计 | 1,169,937 | 813,406 | -356,531 (-30.47%) | 1,713,526 | +900,120 (+110.66%) |

## FeedbackTrace

AgentLoop Baseline 与 BEG 均使用只输出 `KEY` 和原始 Evidence ID 的统一协议。

### 平均指标变化

| 指标 | AgentLoop Baseline | BEG | BEG − AgentLoop Baseline |
|---|---:|---:|---:|
| Verification Point Alignment | 0.6000 | 0.7000 | +0.1000 |
| Evidence Location Score | 0.6500 | 0.6500 | 0.0000 |
| Evidence Hit Rate | 0.6000 | 0.7000 | +0.1000 |

### 预测模型 Token 变化

| Token | AgentLoop Baseline | BEG | BEG − AgentLoop Baseline |
|---|---:|---:|---:|
| 调用数 | 11 | 10 | -1 |
| 输入 | 601,078 | 584,563 | -16,515 (-2.75%) |
| 输出 | 1,252 | 1,234 | -18 (-1.44%) |
| 总计 | 602,330 | 585,797 | -16,533 (-2.74%) |

AgentLoop Baseline 的额外一次调用来自 `ft_087_long` 首次 HTTP 502 后的重试；本次 BEG 的 10 条预测均为一次成功。
