# SilentSwap code_change_correct：预测问题与 judge 问题

核对日期：2026-09-10。范围：7 个模型、BEG/baseline、Qwen/GLM 的正式 100 题结果，以及有针对性的原始答案、gold 和源码复核。没有修改正式预测、评分或 prompt，也没有重跑模型。

## 结论

真实预测遗漏和 judge 误判同时存在。现有证据不能把全部下降归因于 judge，也不足以断言 BEG 图方法本身不如 raw。当前比较还包含不同的阅读提示与停止策略；需要控制这些变量才能归因。

## Judge 给出的逐个 swap 计数（不是已核实的实际正确数）

每组有 100 题、500 个 gold swap。下表为当前 judge 判为完全正确的 swap 数；尚未人工纠正 judge 误判，不能视为人工真值。

| 模型 | Qwen BEG / baseline | GLM BEG / baseline |
|---|---:|---:|
| Luna | 322 / 316 | 321 / 305 |
| DeepSeek Flash | 232 / 229 | 228 / 232 |
| DeepSeek Pro | 66 / 104 | 79 / 119 |
| Terra | 371 / 407 | 374 / 410 |
| Sol | 399 / 417 | 396 / 413 |
| Kimi K3 | 415 / 443 | 412 / 439 |
| Sonnet 5 | 248 / 266 | 260 / 285 |

该表只说明 Terra、Sol、K3、Sonnet 5 在两个 judge 的逐 swap 判断中得分更低，不能据此确认其实际正确数更少。两个 judge 使用相同 rubric，可能有共同偏差；均值、多数票或取较高者都不能直接当作真实计数。

### 已证实的局部更正与未确定的总数

依据下文逐项对照候选、gold 和源码的复核，已确认 Qwen 错判了 6 个 swap：Terra 的 ss_019/1、ss_034/1，Sol 的 ss_034/1，K3 的 ss_017/1、4、5。

| 模型 | 原 Qwen BEG 计数 | 仅应用已证实更正后的暂定计数 | 全量实际正确数 |
|---|---:|---:|---|
| Terra | 371 | 373 | 未确定 |
| Sol | 399 | 400 | 未确定 |
| Kimi K3 | 415 | 418 | 未确定 |

暂定计数保留了其余未复核的 Qwen 判断，因此既不是真实计数，也不是真实计数的下界：仍可能有误判为正确的条目。其余模型、baseline 与 GLM 同样尚未完成全量独立复核，不能暗示它们无需更正。Sonnet ss_003 的理由/布尔不一致未直接计入正确数。

要得到这张表对应的可辩护全量计数，需要对 7 个模型 × 2 组 × 100 题 × 5 个 gold swap，共 7000 个待判条目，固定代码语义标准，检查每题完整的五个候选并独立对齐；同时检查 BEFORE 文档、AFTER 源码和 gold 的一致性。分歧项以及双方一致判对/判错项都要覆盖，不能只修正有利于 BEG 的案例。对证据不足或标准有歧义的条目单列待裁定，不强行填入精确总数。

## 可核实的 judge 问题

1. **K3 BEG，ss_017：分类标签污染代码语义评分。** Qwen 承认 swap 1、4、5 的具体操作、原逻辑、新逻辑和方向均正确，仅因 `swap_type` 与 gold 不同，把 `no_material_contradiction` 判为 false。核对原始答案与 gold 后，这三处语义相符。其余两处已获认可；按代码语义 rubric，此题应从 0 改为 1。GLM 已给 1，Qwen 给 baseline 也是 1。因此这里的 BEG 劣势由误判产生。仅纠正这一题会使 K3 的 Qwen BEG 均分增加 0.010，不能单独抹平当前 0.100 的组间差距。
2. **Terra/Sol BEG，ss_034 swap 1：忽略触发条件。** 答案明确限定 `username_attribute` 匹配其他 XML 属性值、但不匹配 `AttributeName`，因此 AFTER 不再重映射用户、不设置 uid。Qwen 拿“匹配 AttributeName 时仍设置 uid”来反驳，比较了不同条件。gold 的 why_different 和源码均支持候选描述。两题当前均为 4 个 swap 全对、整题 0.5；修正此判断后各为 1。GLM 已认可该 swap。
3. **Terra BEG，ss_019 swap 1：读错异常分支。** AFTER 源码先执行符号查询，捕获 `AttributeError/KeyError` 后进入 `int(value)`，再捕获 `ValueError`。答案准确描述前一个 fallback，Qwen 却用后一个异常类型判其矛盾；GLM 判该 swap 正确。Qwen 当前整题 0.5，修正这一处后为 1。
4. **Sonnet BEG，ss_003 swap 2：理由与布尔字段矛盾。** Qwen note 明写应将 `no_material_contradiction` 标为 true，实际字段却是 false。这可以确认输出自相矛盾，但候选关于 DBAPI 的额外后果仍应单独审查，不能只按 note 自动改分。仅改这一字段也不会跨过整题得分门槛。

这里区分 prompt 的标准表达问题与 judge 未遵守已有指令的问题：现有 prompt 已要求跨字段阅读、语义匹配，以及不惩罚仅未被 gold 提及的说法；上述误判部分是执行失败，不能保证只改 prompt 就彻底解决。

## 可核实的真实预测问题

**ss_006：BEG 漏掉模板文件中的真实变化。** Terra、Sol、K3 的最终 BEG 答案均没有指出 `field.jinja2` 的 `or typename` 和 `directive.jinja2` 的 `arguments|reverse` 两处变化，而各自 baseline 都指出了。BEG 答案集中在 Python 文件；Sol 还把同一种 Fragment 行为重复列为多个 swap。原始答案、gold 与两个 judge 均支持这里存在实际遗漏，不能通过公平改写 judge prompt 把未回答的内容判对。

该样本中，Terra/Sol BEG 各仅有 1 个 swap 获认可，K3 为 0；对应 baseline 均有 4 个获认可。BEG 整题均为 0，baseline 均为 0.5。

已核对 Terra ss_006 保存的预测请求：BEG 提示包含“优先前 5 个目录文件”“更广阅读可能分散注意”“争取 2–3 轮结束”。这些策略可能限制覆盖面，属于待验证的原因，不能仅凭该样本断言图表示本身不可用。

## 评分门槛放大差异

`evaluation/silentswap/judge/judge.py` 的 `_evaluate_checks` 把每个 swap 的六个布尔判断取 AND，再计算：5 个全对得 1；4 个全对得 0.5；0–3 个全对都得 0。这与现有 prompt 一致，没有发现此处实现与规定不符。

因此一处真实遗漏或误判可能令整题下降 0.5；三处分类误判可以令 ss_017 从 1 降到 0。可额外报告完全正确 swap 比例与分项结果作为诊断指标，但不能暗中替换正式评分标准。

## 建议的验证顺序

1. 在独立版本的 judge prompt 中明确：在候选给定的同一 trigger 下比较 BEFORE/AFTER；分类标签本身不作为代码语义错误；false 必须给出候选引用和具体反证；约束理由与布尔字段一致。
2. 固定现有预测，对 BEG 与 baseline 同时做盲化复核，纳入随机样本及明确误判样本，不能只纠正 BEG 的低分项。先测一致性和误判率，再决定是否全量重评；旧结果保留。
3. 独立做预测实验：统一 BEG/raw 的任务语义要求、阅读预算和停止条件，检查模板、配置和其他非 Python 文件的可发现性与可读性，再比较覆盖率和 code_change_correct。

## 数据与复核入口

- 统计：`experiments/silentswap/<模型>/<BEG或baseline>/judges/<judge>/ss_*/result.json` 的 `llm_judge.code_change_checks`。
- 候选：相同实验的 `runs/ss_*/prediction.json`；实际发送内容保存在 `judge_input.json` 与预测 `attempt_*/requests/`。
- Gold：`evaluation/silentswap/artifacts/hidden_gold/ss_*.json`。
- AFTER 源码：`evaluation/silentswap/artifacts/visible_bundles/ss_*/repository/`。
- Judge：`prompts/judge/silentswap.txt`、`evaluation/silentswap/judge/judge.py`。核对了保存请求中的 rubric，避免仅依据可能已改动的当前文件。
