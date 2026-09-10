# SpecGap baseline：两个 judge 的模型排名

统计日期：2026-09-09。覆盖 7 个被测模型，每个模型 100 题；两个 judge 均完成全部评分。Qwen 指 `qwen3.7-max-2026-06-08`，GLM 指 `glm-5-2`。

按各模型 baseline 的 100 条样本指标均值降序排名。这里比较的是模型排名，不是单个模型内部的样本排名。

## F1

| 排名 | Qwen 模型 | Qwen F1 | GLM 模型 | GLM F1 |
|---|---|---:|---|---:|
| 1 | Kimi K3 | 0.485291 | Kimi K3 | 0.502152 |
| 2 | Sol | 0.449259 | Sol | 0.462336 |
| 3 | Sonnet 5 | 0.318286 | Sonnet 5 | 0.338440 |
| 4 | Terra | 0.267595 | Terra | 0.281869 |
| 5 | Luna | 0.234158 | Luna | 0.265216 |
| 6 | DeepSeek Flash | 0.085076 | DeepSeek Flash | 0.096668 |
| 7 | DeepSeek Pro | 0.059875 | DeepSeek Pro | 0.088591 |

两个 judge 的模型排名完全一致，Spearman 排名相关系数为 **1.0000**，所有模型的名次差均为 0。GLM 的平均 F1 全部略高，比 Qwen 高约 **0.012～0.031**，但不影响模型排序。

## Question Quality

| 排名 | Qwen 模型 | Qwen Question Quality | GLM 模型 | GLM Question Quality |
|---|---|---:|---|---:|
| 1 | Kimi K3 | 0.929619 | Kimi K3 | 0.923738 |
| 2 | Sol | 0.924278 | Sol | 0.921702 |
| 3 | Sonnet 5 | 0.770833 | Sonnet 5 | 0.799167 |
| 4 | Terra | 0.676667 | Terra | 0.699786 |
| 5 | Luna | 0.618208 | Luna | 0.636000 |
| 6 | DeepSeek Flash | 0.225500 | DeepSeek Pro | 0.231714 |
| 7 | DeepSeek Pro | 0.185833 | DeepSeek Flash | 0.229119 |

前五名完全一致，只有 DeepSeek Flash 和 DeepSeek Pro 的第六、第七名互换。GLM 下两者仅差 **0.002595**。

## 数据来源

读取 `experiments/specgap/<模型目录>/baseline/summary.json` 中 `judges.<judge>.groups.specgap/raw.score_means` 的 `f1` 和 `question_quality`，仅使用完整 100/100 的评分结果。

| 展示名称 | 模型目录 |
|---|---|
| Kimi K3 | `kimi-k3` |
| Sol | `sol` |
| Sonnet 5 | `claude` |
| Terra | `terra` |
| Luna | `gpt5.6` |
| DeepSeek Flash | `deepseek_flash` |
| DeepSeek Pro | `deepseek_pro` |
