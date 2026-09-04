# BEG 语义与 Token 优化计划

## 1. 目标与验收口径

本轮同时解决两个问题：定位已经提高，但语义收益不足；当前多轮 BEG 反而比历史 Raw baseline 消耗更多 token。

最终只接受同样本、同配置的配对结果。Prediction 对齐各 benchmark 的历史 baseline：SpecGap 使用 GPT-5.6 Luna `disabled`，SilentSwap 使用 GPT-5.6 Luna `medium`；Judge 均使用 GLM-5.2 `disabled`。同一 benchmark 内的 Raw 与 BEG 共用模型配置、动作上限、纠错规则和输出 schema。

- SpecGap：主要语义指标 GLM F1 相对 Raw baseline 提高至少 `0.10` 绝对值；
- SilentSwap：主要语义指标 GLM Code Change 相对 Raw baseline 提高至少 `0.10` 绝对值；
- 两个 benchmark 的定位指标不得低于配对 Raw baseline；
- BEG 的 Prediction input + output token 总量和样本中位数都必须低于配对 Raw baseline；
- 只有配对 95% bootstrap 区间下界大于 0，才称为“显著优于”。

## 2. 已确认的事实

### 2.1 效果瓶颈

- SpecGap 的 146 条 Gold 在图中全部存在，129 条已进入模型实际读取的源码，但最终只有 20 条语义匹配，已读证据转化率仅 `15.5%`。初始目录平均展示 110 个行为 ID，40% 没有 `When/Result`；加入一跳使目录增长 52%，Gold 源码覆盖只增加 4.1 个百分点。
- SilentSwap 的 150 条 Gold 在图中全部存在，120 条被实际读取。完成样本的 29 个未读目标中，24 个已经出现在目录或 query 结果中，只是模型没有继续读取；106 个已对齐目标中有 100 个代码语义完全正确。
- 当前模块六隐藏了图中已有的原子 evidence 和文档锚点，只突出一个行为标签，再附整段 symbol 源码。因此模型获得的是“代码定位器”，而不是容易比较的完整行为语义。
- SilentSwap 中大量 file/symbol 正确但行号过宽；SpecGap 中模型平均只用 6.23/12 轮便提前结束，并经常转去报告与删除 Gold 无关的普通实现细节。
- 当前最终 Code 图仍是扁平的 `{behaviors: [B...], edges: [B→B...]}`；模块五再按 `(path, symbol)` 临时分组。这正好说明原子行为抽取有效，但最终 Repo 图的节点层级不对。
- 全部 100 个样本中，按现有 path/symbol 粗聚合后，SpecGap 的 45,449 条 B 边只剩 12,001 条 symbol 关系，SilentSwap 的 40,477 条只剩 12,791 条。这说明聚合能显著减少导航噪声；原 B 边继续保存在独立的 `behavior_edges.json` 中，不能混入模型使用的最终图。
- 不能只用 `(path, symbol)` 识别节点：SpecGap/SilentSwap 分别有 133/128 个同 path、同 symbol、但定义范围不同的情况。因此 symbol 身份还必须包含完整定义行范围。

结论：模块一至三的抽取与关系算法不需要重写，但模块四必须先把扁平 B 图装配成“symbol 节点 + 内嵌 behavior 证据 + symbol 间 calls/feeds”的最终 Code 图。随后模块五至七直接消费 symbol 图，不再各自临时拼组。完整 symbol 源码继续保留，避免结构化证据改变原意。

### 2.2 历史 token 账目

以下均为 Prediction 模型实际 API usage，不含 Judge；input 包含 provider 报告的 cached tokens，纠错或失败调用只要日志中有 usage 就计入。

| Benchmark 与集合 | Input | Output | Total | 平均 Total |
|---|---:|---:|---:|---:|
| SpecGap 历史 Raw 全 100 | 3,575,093 | 389,482 | 3,964,575 | 39,646 |
| SpecGap 历史 Raw 配对 30 | 976,042 | 111,033 | 1,087,075 | 36,236 |
| SpecGap 当前 BEG 30 | 5,543,401 | 60,693 | 5,604,094 | 186,803 |
| SilentSwap 历史 Raw 全 100 | 14,527,115 | 1,039,614 | 15,566,729 | 155,667 |
| SilentSwap 历史 Raw 配对 30 | 5,834,657 | 321,009 | 6,155,666 | 205,189 |
| SilentSwap 当前 BEG 30 | 10,468,795 | 115,622 | 10,584,417 | 352,814 |

当前 BEG/Raw 比值为：SpecGap `5.16x`，SilentSwap `1.72x`。SilentSwap 即使去掉失败尝试仍为 `1.40x`。历史 SpecGap 日志可能覆盖早期 HTTP 失败，因此其 token 是下界。当前 SilentSwap BEG 错用了 `disabled`，而历史 baseline 使用 `medium`；正式比较必须在每个 benchmark 内对齐对应 baseline 配置。

## 3. 修改方案

### 3.1 先把模块一至四收敛为 symbol-centered Code 图

这一步只调整最终组织层，不重做已经验证有效的证据抽取：

1. **模块一保持输出不变。** 继续保留生产 artifact、逐字 L3 和精确位置，并复用现有确定性 symbol 范围解析；不加入 Test、普通 Repo Document 或任务文档。为重复定义、配置块和模板块补充范围测试即可。
2. **模块二保持原子行为算法与 B ID 不变。** 条件、结果、依赖和结果锚点仍由现有 AST/确定性逻辑提取；B 只作为 symbol 内部的细粒度证据，不再作为模型的读取入口。
3. **模块三保持精确 B→B `calls/feeds` 分析不变。** `behavior_edges.json` 继续作为内部派生和审计依据，不重新从 symbol 层猜边。
4. **模块四改为唯一的聚合边界。** 先展开现有 B，再按 `(path, symbol, full_start, full_end)` 分组，排序生成 `S0001...`；同名但定义范围不同的 symbol 不合并。每个 symbol 只保存一份完整源码，内部保留全部 B 及其证据行范围。
5. 模块四把原 B 边确定性聚合为 S→S 边。`calls` 按 `from/type/to` 去重；`feeds` 按 `from/type/to/via` 去重，不能合并不同数据流。同一 S 内的 B 边只保留在 `behavior_edges.json`，不生成没有导航价值的 S 自环。模块五、模块六和模型只消费最终 S 边。

SpecGap/SilentSwap 最终图采用下面的简单外层结构。`artifact_kind` 只有 `source`、`runtime_template`、`executable`、`configuration` 四种，只供程序解析和校验，不展示给模型：

```json
{
  "input_id": "sg_001",
  "benchmark": "specgap",
  "symbols": [
    {
      "symbol_id": "S0042",
      "artifact_kind": "source",
      "path": "src/time.py",
      "symbol": "normalize",
      "lines": [10, 28],
      "source": "<完整原文>",
      "behaviors": [
        {
          "behavior_id": "B0131",
          "behavior_name": "src/time.py::normalize#raise@18",
          "evidence_lines": [[12, 18]]
        }
      ]
    }
  ],
  "edges": [
    {
      "from": "S0042",
      "type": "calls",
      "to": "S0057"
    }
  ]
}
```

必须校验：每个 B 恰好属于一个 S；B ID、名称、证据行范围和全部 L3 覆盖集合在聚合前后不变；每份 symbol 源码逐字对应其行范围；从 `behavior_edges.json` 重新聚合得到的跨 S 边必须与最终图完全一致；S ID、节点和边均确定性排序。FeedbackTrace 没有 Code symbol，继续使用现有 interaction 图，不套用这套 schema。

### 3.2 先修正模型配置和动作协议

1. Prediction 配置按 benchmark 对齐历史 baseline：SpecGap 使用 `thinking: disabled` 且不设置 reasoning effort；SilentSwap 使用 `thinking: enabled`、`reasoning_effort: "medium"`。Raw 与 BEG 必须发送完全相同的字段，运行清单保存实际请求配置。
2. 取消不稳定的原生多工具调用。模型每次只返回一个严格 JSON 动作：`query`、`read` 或 `finish`；本地校验后才执行。
3. 多动作时一个都不执行，只返回精确格式错误；网络错误与协议错误分开记录，避免把相同错误盲重试五次。
4. `finish` 先做白名单无损规范化，例如空 `qualified_name` 转为空数组、路径斜杠统一；仍不合法时只允许修正格式，不提供新证据，也不能改变为 `query/read`。

### 3.3 模块五：把密集行为目录改成紧凑 symbol 目录

1. 初始目录一行只放一个 symbol：包含 `Read ID`、path、symbol、命中的文档 section，以及带行号的确定性行为信号；不再列出同一 symbol 下几十个 B ID。

模型首轮看到的目录形如：

```text
[[SYMBOL DIRECTORY]]

[DOCUMENT-LINKED SYMBOLS]
Read ID | File | Symbol | Document section | Behavior signals
S0042 | src/patchy/cache.py | PatchingCache.store | Cache behavior | delete@25; state_write@28,30
S0043 | src/patchy/cache.py | PatchingCache.clear | Cache lifecycle | state_clear@12

[OTHER OBSERVABLE SYMBOLS]
Read ID | File | Symbol | Document section | Behavior signals
S0090 | src/patchy/errors.py | format_error | no direct match | format@18; return@22
[[END SYMBOL DIRECTORY]]
```

`S ID` 由模块四确定性生成，模块五只选择和排列 symbol 节点。模型认为某行相关时直接 `read` 对应 `S ID`，模块六再返回该 symbol 的全部原子行为和完整源码；只有目录中找不到目标时才使用 `query`。SilentSwap 使用相同格式，但第一分区名改为 `[ORIGINAL-DOCUMENT-LINKED SYMBOLS]`。

2. SpecGap 先完整覆盖所有 document-linked symbol；SilentSwap 先完整覆盖所有 original-document-linked symbol。按文档 section 和匹配 specificity 轮转，不能让高 specificity 的前几个 section 占满预算。
3. 初始目录预算上限从 `4096` 降为 `3072` token，目录较短时不补齐。预算首先按文档 section 轮转放入 direct symbol；只有全部 direct symbol 已进入目录后，剩余预算才能展示其他公开、可观察 symbol。一跳邻居不再预先展开，由 `read` 按需返回。
4. 目录分区只说明检索来源，不暗示存在问题。

默认短路径为（这只是一种理想效果）：

```text
3072 token 以内的紧凑 direct-symbol 目录
→ 一次批量 read
→ finish
```

### 3.4 模块六：把相关文档与完整代码放在一起

读取任一 `S ID` 时，模块六按 benchmark 返回对应 symbol 的自包含证据包。

SpecGap：

```text
[[CURRENT DOCUMENT EXCERPT]]
This is from the current document, which may omit a few code behaviors.
Section: ...
<文档原文>

[[CURRENT CODE]]
Symbol: ...
Behavior evidence:
- Lines 18-19: <条件原文>
- Lines 20-21: <操作和结果原文>

Full symbol source:
<带行号的完整 symbol 源码>
```

SilentSwap：

```text
[[BEFORE: ORIGINAL DOCUMENT]]
Section: ...
<原文档片段>

[[AFTER: CURRENT CODE]]
Symbol: ...
Behavior evidence:
- Lines 18-21: <当前行为原文>

Full symbol source:
<带行号的完整 symbol 源码>
```

- `Behavior evidence` 必须覆盖该 symbol 的全部原子行为，而不是只突出请求的一个 B ID；模块六根据 `source + evidence_lines` 确定性截取少量原始条件、操作和结果行，不调用模型生成摘要。底层图不重复保存这些原文。
- 完整 symbol 源码始终保留，负责提供行为证据之外的上下文。
- SpecGap 明确 `document_after` 是可能遗漏少量行为的当前文档；SilentSwap 明确 `original_document` 是 BEFORE、当前代码是 AFTER。
- 存在一跳关系时，只在末尾追加简短的 `Related symbols: calls ...; feeds ...`，供模型决定是否继续 `read`。
- 最终定位默认引用精确行为证据行，不再用整个 wrapper symbol 范围代替。

### 3.5 query、Prompt 与运行状态

1. `query` 改为 exact/AND 优先，结果按不同 symbol 去重；普通查询最多返回 12 个 symbol。宽查询只返回按 path/namespace 分组的提示，要求模型缩小关键词，不再返回数百个行为 ID。
2. SpecGap Prompt 不再要求开放式审计整个项目，而是要求从“基本完整、仅删除少量约束”的文档中恢复缺口；逐项检查条件、默认值、异常、顺序、数据引用和依赖选择。
3. SilentSwap Prompt 固定执行“原文档 BEFORE → 当前代码 AFTER → 差异”比较，并在 finish 前检查方向、数量和多处重复操作。
4. `RUN STATE` 只显示各文档 section 下的已读/未读高优先 symbol 数量，不给结论，也不强制读满；用于减少证据已经出现却未跟进和过早 finish。（最简保守实施这一条）

### 3.6 减少输入 token

当前主要开销来自调用轮数过多，以及旧 query 和源码在后续轮次中反复发送；Prediction 输出本身不是主要问题。因此只采用两类控制：

1. 减少轮数：优先执行“目录直接选 symbol → 一次批量 `read` → `finish`”；目录找不到目标时才增加 `query`。保留 12/14 轮作为困难样本上限，但开发集的中位数目标不超过 3 个有效动作。
2. 删除重复输入：query 结果被使用后只保留查询词和已选择的 `S ID`；只保留最新一次 `read` 的完整源码，较早的 read 只保留原始文档片段和精确行为证据行；相同源码和文档片段在一次请求中只出现一份。整个过程不使用 LLM 摘要。

每次试验只需记录 action 数、input/output/total token 和重复内容数量，确认 token 下降不是由漏读造成的。

## 4. 重试规则

Raw 与 BEG 使用同一套简单规则：网络失败原样再试最多 2 次；JSON 或
`finish` 格式错误最多纠正 2 次，纠正时不执行动作、也不增加证据；证据、
grounding 或语义错误不重试。context 超限只压缩一次并原样重发，完整样本不
重跑。Judge 同样最多重试 2 次网络错误，只允许 1 次纯 JSON 格式纠正。

## 5. 实施和验证顺序

1. 离线测试：先验证新 symbol 图完整保留全部 B、由内部 B 边确定性得到全部跨 S 边、Gold 位置覆盖不下降、重复定义不被误合并；再验证 direct symbol 全覆盖、目录不再重复 symbol、read 展示全部同 symbol 行为、query 上限和重试分类正确。
2. 固定开发集：两个 benchmark 各 5 条；SpecGap 的 Raw/BEG 均使用 Luna `disabled`，SilentSwap 的 Raw/BEG 均使用 Luna `medium`。依次验证协议、目录/query、read/Prompt 和输入去重，禁止同时修改多项后无法归因。
3. 进入正式评测前冻结代码、Prompt、样本 seed、模型请求和 Judge 配置。
4. 使用未参与调参的同一批 30 条，各运行一次 Raw 和 BEG；不重跑整批，不挑最好结果。
5. 汇报配对质量差、95% 区间、总/中位 token、action 数和失败分类。未同时达到语义 `+0.10` 与 token 少于 Raw 时，如实判定目标未完成。
