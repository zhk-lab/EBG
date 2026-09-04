# BEG 对齐式 Compact 实验报告

## 改动

目标是在保留有效源码的同时，让模型明确区分“任务直接相关的代码”和“沿图补充的上下文”。实现仍只位于 `.tmp/token_reduction_ablation_20260903`，尚未修改正式源代码。

展示格式改为：

```text
[DIRECT ROOT]
Doc: <明确命中该 Symbol 的最具体文档章节>
path.py::RootSymbol@10-20
10 | <Root 原始源码>

[CONTEXT via calls]
RootSymbol@15 calls OtherSymbol@30-35
30 | <邻居原始源码>
```

规则如下：

1. Root 源码标为 `[DIRECT ROOT]`；调用或数据流带来的邻居分别标为 `[CONTEXT via calls]` 和 `[CONTEXT via feeds]`。
2. 只有文档明确出现完整 Symbol（顶层函数可使用明确函数名）时才显示 `Doc:`。Evidence 词、父作用域和 `__init__` 等通用方法名可以参与 Root 排序，但不能生成文档对齐标签，避免误导。
3. Root 保留完整源码；邻居能由边的 Evidence 定位具体 Behavior 时只返回相关 Behavior，无法可靠定位时回退到完整 Symbol。
4. 源码保持原文件顺序，相同行全局去重；不向模型展示缩进 JSON、Behavior ID 或 Evidence ID。
5. Prompt 已同步解释新格式和直接证据/结构上下文的区别，没有添加 benchmark 或 gold 特例。

## 验证

- 13 个针对性单元测试全部通过，包括精确章节选择、避免通用方法名误匹配、Behavior 安全回退、源码顺序及去重。
- 原 10 个 SilentSwap 样本的 Top-10 离线检查覆盖 `47/50` 个 gold location（94.0%）和 `66/69` 个 gold 行（95.7%），与改动前一致。
- 离线展示 token 相比原 BEG 减少 79.30%。该数字只表示 read 返回内容量，不等于多轮预测的最终输入 token。

## SilentSwap 10 样本正式结果

Prediction 使用 GPT-5.6 Luna（`reasoning_effort=none`），Judge 使用 Qwen 3.7（thinking 关闭）。样本为 `001、007、013、025、037、050、062、075、087、100`，Prediction 与 Judge 均为 10/10 成功。

| 方法 | Localization | Location correct | Code change correct | 调用数 | 输入 token | 总 token |
|---|---:|---:|---:|---:|---:|---:|
| AgentLoop Baseline | 0.4359 | 0.3500 | 0.3000 | 35 | 801,499 | 813,406 |
| 原 BEG | 0.5446 | 0.4000 | 0.4500 | 40 | 1,704,346 | 1,713,526 |
| **对齐式 compact** | **0.5523** | **0.4500** | **0.4500** | **30** | **693,683** | **701,624** |

相比原 BEG，对齐式 compact 的 Localization 提高 `0.0077`，Location correct 提高 `0.05`，Code change correct 持平；输入 token 减少 **59.30%**，总 token 减少 **59.05%**。

相比 AgentLoop Baseline，三项指标分别提高 `0.1164、0.10、0.15`；输入 token 减少 **13.45%**，总 token 减少 **13.74%**。

## 结论

这组 10 样本结果支持对齐式 compact 有效：它保住了原 BEG 的准确率优势，同时将预测 token 降到 AgentLoop Baseline 以下。关键收益不是继续堆剪枝规则，而是用可靠标签把直接证据与结构邻居分开，使紧凑源码仍具有清晰的阅读语义。由于样本量只有 10 个，合入正式实现前仍应在更大规模或独立样本上复验。
