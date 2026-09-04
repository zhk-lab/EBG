# Luna 10×3 指标与 Token（含对齐式 Compact）

AgentLoop Baseline、BEG 与对齐式 Compact 的 Prediction 均为 GPT-5.6 Luna `reasoning_effort=none`，Judge 均为 Qwen 3.7 `enable_thinking=false`。

每个 benchmark 均评测 10 个样本：`001、007、013、025、037、050、062、075、087、100`。

## SpecGap

### 平均指标变化

| 指标 | AgentLoop Baseline | BEG | BEG − AgentLoop Baseline | 对齐式 Compact | 对齐式 Compact − AgentLoop Baseline |
|---|---:|---:|---:|---:|---:|
| Semantic match F1 (LLM) | 0.1438 | 0.2706 | +0.1267 | 0.2198 | +0.0760 |
| Question quality (LLM) | 0.3000 | 0.7000 | +0.4000 | 0.5667 | +0.2667 |
| Location F1 (rule-based) | 0.1438 | 0.2956 | +0.1517 | 0.2614 | +0.1176 |

### 预测模型 Token 变化

| Token | AgentLoop Baseline | BEG | BEG − AgentLoop Baseline | 对齐式 Compact | 对齐式 Compact − AgentLoop Baseline |
|---|---:|---:|---:|---:|---:|
| 调用数 | 29 | 30 | +1 | 23 | -6 |
| 输入 | 499,969 | 791,935 | +291,966 (+58.40%) | 454,450 | -45,519 (-9.10%) |
| 输出 | 2,666 | 4,090 | +1,424 (+53.41%) | 4,458 | +1,792 (+67.22%) |
| 总计 | 502,635 | 796,025 | +293,390 (+58.37%) | 458,908 | -43,727 (-8.70%) |

对齐式 Compact 的三项 SpecGap 指标均高于 AgentLoop Baseline，总 token 减少 8.70%；

## SilentSwap

### 平均指标变化

| 指标 | AgentLoop Baseline | BEG | BEG − AgentLoop Baseline | 对齐式 Compact | 对齐式 Compact − AgentLoop Baseline |
|---|---:|---:|---:|---:|---:|
| Localization score | 0.4359 | 0.5446 | +0.1087 | 0.5523 | +0.1164 |
| Location correct | 0.3500 | 0.4000 | +0.0500 | 0.4500 | +0.1000 |
| Code change correct | 0.3000 | 0.4500 | +0.1500 | 0.4500 | +0.1500 |

### 预测模型 Token 变化

| Token | AgentLoop Baseline | BEG | BEG − AgentLoop Baseline | 对齐式 Compact | 对齐式 Compact − AgentLoop Baseline |
|---|---:|---:|---:|---:|---:|
| 调用数 | 35 | 40 | +5 | 30 | -5 |
| 输入 | 801,499 | 1,704,346 | +902,847 (+112.64%) | 693,683 | -107,816 (-13.45%) |
| 输出 | 11,907 | 9,180 | -2,727 (-22.90%) | 7,941 | -3,966 (-33.31%) |
| 总计 | 813,406 | 1,713,526 | +900,120 (+110.66%) | 701,624 | -111,782 (-13.74%) |

对齐式 Compact 的 SilentSwap 三项指标达到或超过原 BEG；相比原 BEG，输入 token 减少 59.30%、总 token 减少 59.05%；相比 AgentLoop Baseline，总 token 减少 13.74%。

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
