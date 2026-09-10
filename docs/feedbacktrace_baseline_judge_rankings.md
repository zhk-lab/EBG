# FeedbackTrace baseline：两个 judge 的模型排名

统计日期：2026-09-09。覆盖 7 个被测模型，每个模型 100 题；两个 judge 均完成全部评分。Qwen 指 `qwen3.7-max-2026-06-08`，GLM 指 `glm-5.2`。

分数为各指标的样本均值，按分数降序排名。Spearman ρ 比较两个 judge 对同一组 7 个模型的排名；并列分数使用平均名次。

## Spearman 排名相关系数

| 指标 | Qwen 与 GLM-5.2 的 Spearman ρ |
|---|---:|
| 验证点对齐 | 0.8929 |
| 证据定位 | 0.9643 |
| 证据命中 | 1.0000 |

## 验证点对齐

| 名次 | Qwen | 分数 | GLM-5.2 | 分数 |
|---|---|---:|---|---:|
| 1 | Terra | 0.470 | Terra | 0.555 |
| 2 | Sol | 0.425 | Sol | 0.535 |
| 3 | Luna | 0.410 | Luna | 0.510 |
| 4 | Flash | 0.390 | K3 | 0.475 |
| 5 | K3 | 0.365 | Sonnet 5 | 0.455 |
| 6 | Sonnet 5 | 0.300 | Flash | 0.440 |
| 7 | Pro | 0.200 | Pro | 0.315 |

前三名一致；差异集中在 Flash、K3、Sonnet 5 的相对位置。

## 证据定位

| 名次 | Qwen | 分数 | GLM-5.2 | 分数 |
|---|---|---:|---|---:|
| 1 | Terra | 0.590 | Terra | 0.585 |
| 2 | Luna | 0.545 | Luna | 0.545 |
| 3 | Sol | 0.540 | Sol | 0.525 |
| 4 | K3 | 0.525 | K3 | 0.510 |
| 5 | Sonnet 5 | 0.465 | Flash | 0.445 |
| 6 | Flash | 0.460 | Sonnet 5 | 0.435 |
| 7 | Pro | 0.345 | Pro | 0.345 |

仅 Flash 与 Sonnet 5 的第五、第六名互换。

## 证据命中

两个 judge 下的分数和排名完全相同。

| 名次 | 模型 | Qwen | GLM-5.2 |
|---|---|---:|---:|
| 并列 1 | Sol、K3 | 0.550 | 0.550 |
| 并列 3 | Flash、Terra | 0.540 | 0.540 |
| 5 | Luna | 0.530 | 0.530 |
| 6 | Sonnet 5 | 0.480 | 0.480 |
| 7 | Pro | 0.340 | 0.340 |

证据命中由程序根据预测证据 ID 与 gold 证据 ID 是否存在交集计算，不依赖 judge 的主观评分。因此，该指标的 ρ＝1 不能作为两个 judge 独立判断一致的证据。

## 数据来源

读取 `experiments/feedbacktrace/<模型目录>/baseline/summary.json` 中两个 judge 的 `groups.*.score_means`，使用 `scipy.stats.spearmanr` 计算相关系数。

| 展示名称 | 模型目录 |
|---|---|
| Luna | `luna` |
| Flash | `deepseek_flash` |
| Pro | `deepseek_pro` |
| Terra | `terra` |
| Sol | `sol` |
| K3 | `kimi-k3` |
| Sonnet 5 | `claude` |
