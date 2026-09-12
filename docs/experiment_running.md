# BEG 实验运行说明

实验比较规则与参数见 [BEG.md](../BEG.md)。以下操作说明由原模块六迁入。

## 模型选择

切换模型可修改 `.env` 的两个 Profile 选择项，也可给 `predict.py` / `judge.py` 传 `--model-profile <PROFILE>`。参数优先级为命令行 > 对应角色的 `BEG_` / `JUDGE_` 配置 > Profile；`--show-config` 可查看实际生效配置，不发起模型请求。

## 结果保存与续跑

| 项目 | 约定 |
|---|---|
| 结果目录 | `experiments/<benchmark>/<模型目录>/<baseline或BEG>/`；`--experiment-name` 传相对路径，如 `specgap/gpt5.6/BEG` |
| 预测结果目录 | 实验目录下 `runs/<input-id>/`；保留样本内的 attempt、请求、响应和状态 |
| Judge 结果目录 | 实验目录下 `judges/<judge-model>/<input-id>/`；保留评分尝试和补评记录 |
| 汇总 | 实验根目录唯一的 `summary.json`：`prediction` 保存预测汇总，`judges[模型名]` 保存对应评分汇总 |
| 续跑 | 同名实验按已完成结果续跑，并校验配置；更换配置使用新实验名 |

`benchmark`、`arm` 和阶段保留在 manifest 中，不再重复放入单组实验的目录路径。历史上同一实验同时运行 raw 和 graph 时，仅额外保留 arm 层以避免同名样本冲突。根目录汇总的 schema_version 为 2，两个阶段更新各自部分；后续预测汇总不会覆盖已保存的 Judge 汇总。

旧目录迁移需在该实验停止写入后执行：`python -m scripts.main.migrate_layout --experiment-name specgap/deepseek_flash/BEG`。加 `--check` 只检查路径和冲突；不指定实验名时处理 `experiments/` 下的全部实验。迁移保留原始答案、响应、分数和断点，合并旧汇总，并更新 Judge 续跑所用的预测路径。

配置入口：`scripts/model_config.py`、`agentloop/config.py`、`tracereview/config.py`；批量默认值见 `scripts/main/predict.py` 与 `scripts/main/judge.py`。

## 取证与输入细则

以下保留实验协议精简前的操作约定，方法说明见 BEG.md 附录 D。

### Repo

模型每轮选择搜索、读取或提交一个动作，最后一轮必须提交。

搜索优先匹配完整标识、路径、Symbol 与行为名称；无精确命中时，按空白拆分查询并作大小写不敏感的 AND 匹配。结果按文件聚合，超过上限时按匹配程度截取。初始目录未展示的位置仍可搜索，文件内其他作用域可通过专用入口单独读取。

目录与搜索结果仅用于导航，最终引用须由成功读取的源码覆盖。两组共用引用校验规则。BEG 的入口与呈现方式遵循附录 D，并固定以下读取规则：

- 目录同级入口按生产材料类型、可观察行为数量与稳定路径排序，各文档章节轮流进入目录预算。
- Root 优先采用明确文档对应，其次采用可唯一定位的代码证据，父作用域用于兜底；无直接命中时选择一个代表作用域。Root 在展开前固定，不因去重后的剩余预算追加入口。
- 模块级 Root 只在文档明确涉及常量、配置或注册语句时选取，并呈现命中片段。
- 邻居须通过具有 Root 行为证据支持的直接边连接；优先保留调用目标、数据来源和有文档对应的相邻作用域。端点证据命中多个 Behavior 时全部保留，无法可靠定位时返回完整作用域。
- 相同邻居与源码行在一次读取中去重，其他作用域仍可独立读取。通用方法名与父作用域的匹配不单独生成文档锚点。


### Trace

FeedbackTrace 使用单次模型调用。Raw 输入完整原始轨迹；BEG 输入全部已构建的 Task Scope，并在各 Behavior 内呈现需求、工具交互与回复。没有形成 Behavior 的孤立用户事件不进入 BEG 模型输入，因此完整 Scope 输入不等于与 Raw 完全相同的原文覆盖范围。

BEG 仅展开逐字证据片段，不生成摘要。保留原始证据标识及工具名称，不向模型输出内部字符范围；共同引用的回复只展示与各行为对应的片段。Scope 按首次出现位置排列，内部按行为顺序排列。当前任务取截止点前最后一个 Behavior 的归属，历史任务全部保留。引用边仅展示类型与目标任务，时间边不展示，不额外添加 Behavior 间关系边。

正式样本均包含一个重要的 Verification Point。模型输出该验证点、支撑 Evidence 与严重程度，保存时统一补充 `verdict: KEY`。两组共用任务定义、输出上限、预测格式及 Judge，引用须来自实际输入中的证据标识。

