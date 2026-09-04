# BEG（Behavioral Evidence Graph）与三 Benchmark 对齐重构计划

## 1. 总体方案

BEG 将 Code 与 Trace 统一表示为行为证据图：SpecGap/SilentSwap 的节点来自生产 Code，FeedbackTrace 的节点来自完整 Trace。Evidence Intake、Behavior Atomization、Relation Linking、Graph Assembly、Behavior Directory 和共用 AgentLoop 已实现；Repo 评测采用“行为目录 + 按需浏览 + 统一 AgentLoop”，FeedbackTrace 采用完整 Trace view 后一次预测。

输入边界：SpecGap 和 SilentSwap 使用“完整任务文档（图外）+ 仅含生产 Code 的 Repo 行为图”；FeedbackTrace 使用仅含完整 Trace 的行为图。Test 和普通 Repo Document 不进入图。

### 方法要解决的问题

本项目要解决的是长程 Agent 任务中的监督不足。随着任务推进，文档、代码、测试、工具结果和多轮反馈会快速累积；如果让模型反复阅读完整原始输入，关键行为容易被大量实现细节和重复内容淹没，既消耗 token，也难以判断 Agent 是否遗漏约束、偏离要求或没有正确利用反馈。

### 方法优势与目标

1. **确定性减少无关输入。** 按桌面 baseline 的评分口径，SpecGap 和 SilentSwap 在输入适配阶段直接排除 Repo Test、普通 Repo Document、CI 文件和依赖锁，只保留特殊任务文档与生产源码、运行时模板、可执行脚本、行为配置。后三个阶段不再删除保留范围内的事实，而是将事实重组为行为并合并重复关系，保证后续仍可完整检索。
2. **更清楚地发现值得汇报的行为。** 图中的基本单位不是孤立代码片段，而是“条件与数据 → 可观察结果”的行为；行为之间只用源码可证明的调用和数据关系连接。FeedbackTrace 则保留完整时间顺序与明确对象连续关系。这样能把原始材料变成更适合检查规范缺口、语义替换和反馈落实情况的阅读单位。

最终目标是在三个 benchmark 上同时实现：端到端输入 token 少于公平配对的 baseline，主要质量指标显著高于 baseline。当前四个阶段只证明图能确定性构建且保留选中范围内的事实；真正的 token 与跑分收益必须在模块五至模块七完成后，通过同一 Agent 下的 Raw Repo/Trace 与 BEG 配对实验验证。

三个 benchmark 的公平对应关系如下：

| Benchmark | 桌面正式评测 | 新统一评测 | 分流点 |
|---|---|---|---|
| SpecGap | 完整 `document_after` + Repo 目录；`search/read_file/finish`，最多 12 个动作 | 完整 `document_after` + 对应目录；`query/read/finish`，最多 12 轮 | 模块五 |
| SilentSwap | 完整 `original_document` + Repo 目录；`read_files/final`，正式计分版本不限制轮数 | 完整 `original_document` + 对应目录；`query/read/finish`，最多 14 轮 | 模块五 |
| FeedbackTrace | 完整反馈前事件，一次预测 | 完整事件经行为分组但保持原序后，一次预测 | 模块五；不启用多轮浏览 |

## 2. 四个确定性构图阶段

### Evidence Intake（证据准入）

Evidence Intake 重新实现 L3 抽取与定位，并按 benchmark 明确输入边界：

- SpecGap 单独保留完整 `3_document_after.md`，SilentSwap 单独保留完整 `original_document.md`；它们是任务规范，不属于 Repo 行为图；
- Repo 侧只抽取能够决定运行行为的生产 artifact，包括生产源码、运行时模板、可执行脚本、行为配置和依赖声明；
- Repo Test 以及 README、docs、CHANGELOG 等普通 Repo Document 不生成 L3，也不进入后续模块；
- CI、编辑器配置、依赖锁、构建产物和第三方目录同样排除；源码中的 docstring 属于源码事实，保留在对应行为中；
- FeedbackTrace 没有 Repo 或任务文档，继续保留截止点前的全部 Trace 事件。

过滤依据是正式 baseline 的 artifact 口径，不是简单匹配路径中的 `test`。明确的测试目录、fixture、runner 和 pytest 配置排除；但被正式 Gold 当作实现的 `requirements/test.txt`、`atest/*`、`src/cleo/testers/*`、`pytest_isort/*` 仍作为生产事实保留。

所有保留的 L3 必须包含逐字原文和精确位置。ID 按规范化路径、源码位置或 Trace 顺序确定性重建。

### Behavior Atomization（行为原子化）

重写 L2 组织方式，只使用 Evidence Intake 保留的生产行为事实，不再混入 Test 或 Repo Document。

Behavior Atomization 的每个原子行为只保留三个字段：

```json
{
  "behavior_id": "B0042",
  "behavior_name": "src/time.py::normalize#raise@18",
  "l3_ids": ["ev_000012", "ev_000019", "ev_000021"]
}
```

构建器仍需在内存中识别条件和结果，用它们划分行为并生成 `behavior_name`，但不输出 `role` 或 `kind`：`role` 对复杂分支并不稳定，`kind` 又与名称中的 `#结果类型@行号` 重复。Code 的 `l3_ids` 非空、去重，并按路径、起止行和 L3 ID 确定性排序；数组位置不表示 trigger、step 或 effect。名称锚定的结果必须出现在这些 L3 中。Trace 使用相同的三个字段，`l3_ids` 按原始时间排序，第一项是 `user_prompt`，随后包含截止下个 `user_prompt` 前的全部事件。

代码行为以一条可执行路径上的“触发条件 → 有序可观察结果”为单位：

- 结果包括 `return`、`raise`、输出、外部调用、公开状态修改等；
- 对 Python 使用 AST、控制路径和到达定义分析找到结果，再反向追溯必要条件和数据依赖；非 Python artifact 使用确定性文本分块；
- 同一 symbol 的不同分支分别成为行为，重复行为合并。

FeedbackTrace 则以一次完整的“用户要求 → Agent 行为 → 工具结果/后续决定”为行为单元。

`fallthrough` 表示函数存在可到达的隐式返回路径；`load` 表示某段保留事实没有独立可观察结果，但仍需进入图供后续检索。二者都不把内部上下文伪装成新的外部效果。

两个标识均由程序生成，不调用模型：

1. Code 的 `behavior_name` 使用“规范化 Repo 相对路径 `::` fully-qualified symbol `#` 最终结果类型 `@` 结果起始行”，例如 `src/time.py::normalize#raise@18`；无 symbol 时使用 `<module>` 或确定性的配置 key/block；
2. 同名候选按结果列号、触发位置和有序 evidence 位置排序，依次追加 `~2`、`~3`，保证名称唯一；
3. Trace 使用 `trace::interaction@turn37`，重复 turn 使用同样的后缀规则；
4. 全部行为按唯一 `behavior_name` 排序后依次分配 `B0001`、`B0002`，因此相同输入必然得到相同 ID。输入改变时重新执行 Behavior Atomization，并使模块五至七缓存失效。

Relation Linking 的边、Graph Assembly 的节点、模块五的目录和模块六的工具统一使用 `behavior_id`，中途不得重新编号或改名。目录显示 `behavior_name`；读取结果可把同一名称无损展开为更易读的 `Source` 和 `Result`，不重复显示名称。

### Relation Linking（关系连接）

Relation Linking 只连接真正参与运行或决策的行为。对于 SpecGap 和 SilentSwap，图中只有生产行为节点；Test 和普通 Repo Document 已在 Evidence Intake 排除，因此不产生节点或边。特殊任务文档保持在图外，只作为模块七判断行为是否符合规范的完整比较基准。

生产代码行为之间只保留两种可由源码直接证明的边：

- `calls`：A 直接调用 B；
- `feeds`：A 的返回值、明确产出或可精确匹配的状态写入被 B 使用；

同一个被调用 symbol 可能包含多个互斥结果行为，因此 `calls` 及其返回值 `feeds` 可以连接到多个可能结果；它们是静态 may-call/may-feed，不断言一次执行会同时产生所有结果。`feeds.via` 必须写明返回值或状态对象；`calls` 不带 `via`。包内绝对与相对 import 都需要解析，无法由源码静态确定的动态调用不猜边。

FeedbackTrace 的一个行为节点包含一个 `user_prompt` 及下一个 `user_prompt` 前的全部 Agent 事件，节点内严格保持原始顺序。节点之间只保留两种边：

- `precedes`：连接时间上直接相邻的行为；
- `continues`：连接操作同一明确路径、URL 或 task ID 的非相邻行为。

两种边只表达时间和对象连续性，不推断因果、授权、纠正或是否正确落实反馈。

### Graph Assembly（图装配）

Graph Assembly 不重新定义行为，只把 Behavior Atomization 的 L3 引用展开为位置和原文，加入 Relation Linking 的边，校验后输出 `behavior_graph.json`。`behavior_id` 和 `behavior_name` 必须原样保留。

`behavior_graph.json` 是 BEG 的最终事实图：在 SpecGap/SilentSwap 中是图结构化的生产 Code，在 FeedbackTrace 中是图结构化的完整 Trace。它是程序内部的检索库和校验对象，不直接发送给模型；模块六按需把一个行为的定位、原文和关系渲染为自包含文本，避免模型跨 JSON 区域查 ID。

三个 benchmark 共用四个顶层字段：

```json
{
  "input_id": "",
  "benchmark": "",
  "behaviors": [],
  "edges": []
}
```

SpecGap 和 SilentSwap 的 Code 行为展开为：

```json
{
  "behavior_id": "B0042",
  "behavior_name": "src/time.py::normalize#raise@18",
  "location": {
    "kind": "source",
    "path": "src/time.py",
    "symbol": "normalize",
    "lines": [10, 28]
  },
  "evidence": [
    {"lines": [12, 13], "text": "<原始代码>"},
    {"lines": [14, 17], "text": "<原始代码>"},
    {"lines": [18, 18], "text": "<原始代码>"}
  ]
}
```

Behavior Atomization 中的 `l3_ids` 到此被有序的位置和原文替代；后续模块打开一个行为即可获得完整证据，不再跨表查找。

FeedbackTrace 行为展开为：

```json
{
  "behavior_id": "B0001",
  "behavior_name": "trace::interaction@turn3",
  "events": [
    {"turn": 3, "type": "user_prompt", "text": "<原始事件>"},
    {"turn": 4, "type": "tool_exchange", "evidence_id": "E18", "text": "<原始事件>"}
  ]
}
```

事件数组就是原始顺序；能够作为预测证据的 Agent/Tool 事件把原 evidence ID 与原文放在一起，用户事件不写空 ID。边只引用短 `behavior_id`：

```json
{"from": "B0031", "type": "calls", "to": "B0042"}
```

```json
{"from": "B0001", "type": "continues", "to": "B0003", "via": "src/config.py"}
```

`via` 只在需要说明明确数据、状态或连续操作对象时出现。每条边的依据必须已经包含在两端行为的原文中，不重复保存第三份证据。

## 3. 模块五至模块七

从模块五开始，所有**模型可见内容**都渲染为连续文本，不再把 `behavior_graph.json` 作为上下文：模块五提供文本目录，模块六把被选中的行为、定位、原文和关系就地展开为文本，模块七在文本对话中完成判断。程序内部仍读取 JSON 来执行搜索、展开和校验；最终 `prediction.json` 仍按 benchmark schema 输出。这里不是把整张图一次性转成大文本，而是按需生成少量、自包含的行为文本。

### Behavior Directory（行为目录）

模块五把 `behavior_graph.json` 变成简短的初始阅读目录，不修改图，也不调用模型。生成函数只接收行为图、完整任务文档、token 上限和 tokenizer；Gold、patch、删除内容、修改前代码及普通 Repo 文档都不在输入接口中。最终只把 `behavior_directory.txt` 发给模型，未展示行为仍可由模块六的 `query/read` 访问。

SpecGap 与 SilentSwap 都使用两个高优先级分区：

1. 文档直接命中的 Code：SpecGap 标为 `CODE IDENTIFIED FROM DOCUMENT`，SilentSwap 标为 `ORIGINAL-DOCUMENT-LINKED CODE`；
2. 与第一分区存在一跳 `calls/feeds` 的 Code：统一标为 `ONE-HOP CONNECTED CODE`。

目录主体固定为“文件 → 完整 symbol → behavior ID 与结果标签”：

```text
[ORIGINAL-DOCUMENT-LINKED CODE]

singer/utils.py
  strftime
    Document match: singer.strftime
    B0368  raise@69
    B0369  return@80

[ONE-HOP CONNECTED CODE]

singer/messages.py
  format_message
    B0214  return@54
```

规则固定如下：

- 一个 `(path, symbol)` 是最小单位；同一 symbol 的全部行为要么完整出现，要么整组延后，绝不截断。
- 文档匹配依次使用完整限定名、路径与唯一 symbol 的邻近共现、唯一且以代码形式出现的 symbol 或配置键、`<module>` 的精确路径。标识符区分大小写，不做模糊匹配、embedding 或 LLM 摘要。
- 同一匹配强度下，将文档位置划为 16 个覆盖 lane 轮转取项，避免长文档前半部分占满目录。
- 一跳只使用 `calls/feeds`。不同来源行为轮转取邻居，每个来源先取一个，再进入下一轮；单个来源内按“向外 calls → 向外 feeds → 向内 calls → 向内 feeds”排序。
- 默认使用 `o200k_base` 和 4096 token。正文先为直接命中与一跳预留 70%/30%，未使用空间自动让给另一侧；超限整组跳过后继续尝试较小组。
- 分区与顺序只表示阅读优先级，不表示其中一定存在问题，也不为单个 Repo 添加特殊规则。

该设计同时限制噪声并保留高价值位置。规则冻结后的 100 条描述性离线审计中，SpecGap 目录保留 53.8% 的图行为，覆盖 83.5% 的正式实现位置，且 92.9% 的 condition 至少命中一个位置；SilentSwap 保留 51.6% 的图行为，覆盖 84.0% 的 swap 位置。Gold 只由独立审计脚本读取，生产生成路径永不导入该脚本；调参审计必须显式提供冻结 split，读取 full-100 还必须显式允许 locked Gold。全量数字只作为结构诊断，不作为锁定测试成绩。

FeedbackTrace 没有 Repo，因此模块五不生成目录或筛选行为；模块六直接把完整 Trace 图按原始事件顺序渲染为自包含文本。

### 模块六：确定性行为图浏览器

代码边界固定如下：`src/beg/` 只保存 BEG 构图方法；`agentloop/` 提供 SpecGap 与 SilentSwap 共用的动作循环、`query/read`、上下文压缩、重试和续跑；两个 benchmark 的轮数及资源文件名位于 `agentloop/benchmark_configs/`。完整任务文档由 Evidence Intake 在运行时传入，模块五目录也只作为输入传入，二者都不存放在 `agentloop/`。BEG 任务 Prompt 位于 `prompts/BEG/`，桌面正式 baseline Prompt 位于 `prompts/baseline/`，schema 位于 `schemas/`；模块七的消息组装位于 `src/evaluation_core/messages.py`，结果校验位于 `src/evaluation_core/contracts.py`。FeedbackTrace 不使用 AgentLoop，由 `tracereview/` 的 TraceReview 独立执行一次性预测。

SpecGap/SilentSwap 的完整交互协议只有三个动作：

1. `query(text)`：不知道行为 ID 时使用；
2. `read(ids)`：读取一个或多个已知行为 ID；
3. `finish(prediction)`：证据足够时提交符合 benchmark schema 的最终结果并结束。

模块六实现前两个只读动作，模块七实现 `finish`。不再提供单独的 `expand`：`read` 会返回所选行为的完整 symbol 原文和全部一跳邻居 ID，需要继续查看邻居时再次调用 `read`。

`read` 把定位、原文和关系放在一起：

```text
[[BEHAVIOR SUMMARY]]
Unit ID: B0042
Unit name: src/time.py::normalize#raise@18
File: src/time.py
Symbol: normalize
Lines: 10-28
Observable result: raise@18
[[END BEHAVIOR SUMMARY]]

[[ORIGINAL SOURCE CODE]]
Supports units: B0042
File: src/time.py
Symbol: normalize
Lines: 10-28
10 | <original source code with line numbers>
[[END ORIGINAL SOURCE CODE]]

[[DIRECTLY RELATED UNITS]]
Unit ID: B0042
Called by: B0031  src/parser.py::parse#return@44
Provides data to: B0057  src/output.py::emit#external_call@73
[[END DIRECTLY RELATED UNITS]]
```

区块标题和字段名固定使用上述英文。采用明确的 `key: value`，不用 `the behavior ID is ...` 等完整句子，也不使用没有字段名的竖线位置参数。虽然名称、结果和位置有少量重复，但这些标签只出现在模型主动读取的行为中，优先保证模型无需推断字段含义。所有内容只来自图中已有字段，不调用模型生成。

规则：

- `query` 默认检索行为图中的全部行为，并匹配路径、symbol、`behavior_name` 和该行为自己的局部原始证据；不重复搜索每个行为所属的完整 symbol。路径或 symbol 导航对共享同一源码范围的行为只返回一个代表；普通查询和精确 symbol 最多返回 50 条，精确完整路径最多返回 250 条目录式结果；
- `read` 一次接受 1 至 6 个 ID，返回所选行为所在完整 symbol 的逐行源码，以及全部一跳邻居 ID、名称和关系；
- 多个行为共享完全相同的 symbol 范围时，同一次响应只发送一份源码，但保留每个行为的名称和关系；
- 单次响应使用第四部分冻结的共同 token 上限；放不下的整项延后并返回 `deferred_ids`，模型再次调用 `read` 即可，不使用分页参数；
- 单个完整 symbol 不从中间截断；超过 16K 时可完整返回至 36K；
- query 结果和邻居列表始终同时显示 `behavior_id` 与 `behavior_name`，不让模型仅凭短 ID 猜测内容；
- 完整任务文档始终保留在对话上下文中，不在每个行为包中重复发送；
- 最终输出位置必须来自模型实际看过的内容。

FeedbackTrace 的模块六不提供交互工具，而是把完整 Trace 图渲染为一次性模型输入 `trace_behavior_view.txt`：

```text
[[INTERACTION]]
[[INTERACTION SUMMARY]]
Interaction ID: B0001
Interaction name: trace::interaction@turn3
Start turn: 3
[[END INTERACTION SUMMARY]]

[[USER PROMPT]]
Turn: 3
<original user prompt>
[[END USER PROMPT]]

[[TOOL EXCHANGE]]
Turn: 4
Evidence ID: E18
<original tool invocation and result>
[[END TOOL EXCHANGE]]

[[END INTERACTION]]
```

区块标题由原始事件类型确定，例如 `USER PROMPT` 或 `TOOL EXCHANGE`，不把工具交换拆成虚构的 Agent/Tool 事件。文本严格保持原始时间顺序和全部事件。可用于最终预测的事件将原 evidence ID 写在原文旁；用户事件没有 evidence ID。若两个非相邻行为操作同一明确文件、URL 或 task ID，则用固定英文句式注明对象连续关系，但不判断反馈是否被正确落实。


### 模块七：按对应协议预测

统一使用 DeepSeek-v4-flash，关闭思考模式；AgentLoop 和 TraceReview 每次模型 completion 的 `max_output_tokens` 固定为 32768。

六份任务 Prompt 按协议隔离：`prompts/BEG/{specgap,silentswap,feedbacktrace}.txt` 供 BEG 与统一 Raw/Graph A/B 使用；`prompts/baseline/` 只复现桌面正式 baseline。后者的正式来源分别是 `Desktop/SpecGAP/scripts/evaluate_specgap.py`、`Desktop/SilentSwap/scripts/evaluate_models.py` 加 `scripts/evaluate_deepseek.py`，以及 `Desktop/FeedbackTrace/prompts/feedbacktrace_predictor.txt`。正式 baseline 的 `search/read_file/finish`、`read_files/final` 和一次性预测协议不能接入 BEG 的 `query/read/finish` AgentLoop。

SpecGap/SilentSwap 使用“共用 AgentLoop system + 含输出合同的 benchmark task prompt + 运行时 document/index”；FeedbackTrace 没有额外的通用 system 层，直接把含输出合同的 `prompts/BEG/feedbacktrace.txt` 作为唯一 system prompt，user message 只承载运行标识与完整 Trace view。`schemas/` 只供程序校验，不再重复注入模型输入。

SpecGap/SilentSwap 只能通过 `finish(prediction)` 提交结果；提交成功后立即结束，不再允许调用 `query` 或 `read`。FeedbackTrace 由 TraceReview 读取完整时间线后直接提交一次预测，没有浏览动作。

Prompt 必须首先解释：

- 行为目录是中立索引，不代表其中一定存在问题；
- 初始目录只优先展示部分行为，未展示的行为仍可通过 `query` 检索；
- 完整目标文档是规范基准；
- 必须结合触发条件、外部结果和目标文档判断。

输出合同与原 benchmark 对齐：

- SpecGap：不设置人为 findings 数量上限，只允许提交桌面 baseline 接受的生产代码位置；
- SilentSwap：必须恰好输出 5 条；
- FeedbackTrace：恢复原协议需要的 verdict、criticality 和 evidence 字段。

Schema 只固定正式字段、类型和 benchmark 明确要求的数量，不增加文本长度、去重或行段数量上限，避免把桌面正式评测可接受的答案提前判为失败。Raw 与 Graph 的 runner 都只校验最终 path 和行范围确实来自成功 `read`；symbol 的精确 kind/name 由同一个正式 scorer 判断。

API 调用边界由 `--prepare-only` 固定：先选择合法 Prompt，加载 schema、完整任务文档与目录（FeedbackTrace 为 Trace view），组装首轮 messages，估算长度并将 request 落盘，然后退出。该模式不得创建 API 客户端，也不得生成 response 或 prediction；相同输入可幂等复用，不同输入不得覆盖已有 request。正式运行才从这一已验证请求继续调用模型。

保存原始响应和 `prediction.json`；支持断点续传；无效响应明确失败，不再调用模型修复。

## 4. 测试与公平评测

### 统一评测 Agent（已冻结）

共用 `AgentLoop` 已实现，Raw Repo 与行为图只替换证据后端。两端都使用 Evidence Intake 确定的同一批生产 artifact：Raw 后端保留原文件文本，Graph 后端使用行为图重组。共用 Prompt 只称其为 `repository index/unit`，不提示哪一端更可靠。桌面正式 baseline 并不统一：SpecGap 使用 `search/read_file/finish` 且最多 12 个动作；SilentSwap 使用 `read_files/final`，正式计分版本没有轮数上限。因此，旧分数只作为历史参考；主要 A/B 必须在新 Agent 下重新运行 Raw Repo baseline。

SpecGap 与 SilentSwap 共用三个动作，每轮只能返回一个 JSON：

```json
{"action":"query","text":"..."}
{"action":"read","ids":["B0042"]}
{"action":"finish","prediction":{}}
```

- `query`：对 `text` 做大小写不敏感的关键词匹配，检索路径、symbol、名称和原始证据；多个空白分隔的关键词采用 OR，只要任一关键词是字段中的连续子串即命中。精确路径、symbol 或名称命中优先，其余按稳定 ID 排序。普通查询最多返回 50 个 ID；查询完整、精确的 Repo 路径时最多返回 250 个 ID，避免大文件后半部行为被固定截断。Raw 后端检索固定源码单元及其完整源码，Graph 后端检索原子行为自己的局部证据，不用重复的完整 symbol 源码制造大量相同命中，也不做 BM25、embedding、同义词扩展或额外模型调用；
- `read`：一次读取 1 至 6 个已知 ID，按请求顺序返回自包含文本；Raw 单元使用稳定 `R` ID，Graph 行为沿用稳定 `B` ID；
- `finish`：`prediction` 直接使用对应 benchmark 的原 schema；非空结果中的位置必须来自曾经成功读取的内容。

轮数固定为：SpecGap 12 轮，SilentSwap 14 轮，均包含 `finish`。可见正式运行记录中，SilentSwap 无思考模式最多使用 11 轮，14 轮保留 3 轮余量，又避免 20 轮鼓励无效探索。每次预测模型 completion 都消耗一轮；合法 `query/read` 的参数错误返回固定错误并消耗该轮；非 JSON、未知动作或无效 `finish` 直接失败，不额外调用模型修复。仅网络重试不消耗轮数，但必须原样重发并记录。最后一轮只能 `finish`，否则样本失败。

### 统一输出上限、消息格式与上下文压缩

物理上下文与工作上下文分开控制。1M 是服务端容量，不应成为日常工作记忆；达到 128K 后即清理旧证据，避免注意力被长历史稀释。当前冻结配置为：

```text
context_window = 1_000_000
max_output_tokens = 32_768
provider_safety_margin = 32_768
physical_hard_limit = 934_464
working_prompt_trigger = 131_072
compression_target = 98_304

index_budget = 4_096
tool_result_budget = 16_384
max_atomic_unit_tokens = 36_864
query_max_results = 50
exact_path_query_max_results = 250
query_max_characters = 512
read_max_ids = 6
preferred_recent_reads = 2
network_retries = 4
```

`index_budget` 与 `tool_result_budget` 只是目录和普通单次工具结果的上限。完整单元绝不从中间截断：单个单元超过 16K 时允许完整返回至 36K，多 ID 请求则只返回能完整放入的最长前缀，其余进入 `deferred_ids`。模块五仍用冻结的 `o200k_base` tokenizer 精确生成 4096-token 目录；模块七为减少多轮组装开销，按 `ceil(UTF-8 bytes / 3)` 快速估算完整序列化消息，包括角色包装、固定提示和工具协议。该估算只驱动共同预算与压缩，API 返回的实际 usage 才用于最终 token 报告。

这些值由现有 200 个 Repo 样本验证：4096-token 目录的 Gold 定位覆盖率约为 SpecGap 83.5%、SilentSwap 84.0%，扩到 8192 的收益很小。连续六单元读取中，Graph 的 6752 组有 99.23% 可一次完整返回，Raw 的 503 组为 94.04%，合计约 98.87%，合并 P99 为 15016 token。加入行号、标签和关系后，极少数单元达到 33911 token，因此单单元上限从源码估计值 24576 修正为 36864。现有单文件最多对应 219 个不同源码范围，其精确路径查询结果约 10563 token，故 250 条能完整覆盖且不突破 16K 工具预算。模型 completion 单独保留 32K 输出空间。

本实验不依赖服务端隐藏记忆。每次请求都由 runner 按以下顺序重建消息数组：

```text
system     [[AGENT PROTOCOL]]
user       [[BENCHMARK TASK WITH OUTPUT CONTRACT]]
           [[TASK DOCUMENT]] [[INITIAL DIRECTORY]]
assistant  turn 1 action JSON: query | read | finish
user       [[TOOL RESULT turn=1]]
...        earlier action-result pairs in chronological order
assistant  latest action JSON: query | read | finish
user       [[LATEST TOOL RESULT]]
           [[RUN STATE]] remaining_rounds, action_ledger, compression_state
```

初始任务只在数组中出现一次；上一轮动作和工具结果按时间顺序追加在它后面，而不是拼到 system 前面。下一次 API 调用会重新发送整个数组。只有最后一个 `RUN STATE` 保留完整的当前状态，旧轮次不重复附带状态副本。

system、动作协议、含输出合同的 benchmark 任务、完整 `document_after/original_document`、初始目录和最新 `RUN STATE` 始终完整保留。最新工具结果必须在下一次模型调用中完整出现；任务文档和单个源码/行为单元都不得从中间截断，也不使用 LLM 摘要。旧 query 在完整展示一次后立即变为短 receipt；重复 ID 和重复源码只保留最近一份完整内容。

若去重后的请求超过 `working_prompt_trigger`，runner 按以下顺序逐单元压缩，降至 `compression_target` 后立即停止：

1. 旧 query 保持 receipt，不恢复其长结果；
2. 从最旧 read 开始，把完整单元逐个改为 read receipt，避免一次删掉整轮而过度压缩；
3. 尽量保留最近两次 read，最新工具结果绝不在首次展示前压缩；
4. 若保留内容仍高于 98304，只允许因为最新完整结果或受保护的近期证据而暂时超出目标，但不得超过物理上限；
5. 固定内容或必要完整单元超过物理上限时记为 `context_unfit`。

receipt 只保留轮次、动作、规范化参数、状态、ID、名称和位置。被压缩的 ID 可再次 `query/read`，并正常消耗一轮。若本地计数允许但服务端仍报 context-length，应急压缩会保留最新工具结果、把其余可压缩旧 read 全部改为 receipt；只有新请求确实更短时才原样重试一次，且不消耗 Agent 轮。没有可压缩内容、压缩后没有变短或再次失败时，记为 `context_unfit/provider_limit_mismatch`。

runner 持久化原始交互和结构化 action ledger，断点恢复时重新构造消息，不续接已压缩的字符串。每轮记录压缩前后估算 token、receipt、`deferred_ids`、重新读取、API 返回的实际输入/输出 token、网络重试和失败类型，最终 `finish` 也保存独立 terminal record。状态同时绑定不含密钥的模型配置和底层证据身份，模型、端点、思考模式或输入变化时拒绝复用旧响应。Raw 与 Graph 使用相同轻量估算器、消息格式、阈值、压缩顺序和 receipt；两端按各自估算长度独立触发压缩，并报告实际 usage 的配对差值及“从未触发压缩”的样本子集结果。

### 三组结果

最终同时报告：

1. 桌面正式 baseline：只用于复现和外部比较；
2. 统一 Agent + Raw Repo：主要配对 baseline；
3. 统一 Agent + behavior graph：新方法。

第 2、3 组除初始目录、`query` 索引和 `read` 内容外，样本、底层生产事实、Prompt、模型、思考模式、轮数、工具预算、上下文规则、输出 schema、Judge 与 Gold 必须完全相同。图的邻接关系和更集中的行为文本属于被测方法本身。FeedbackTrace 没有 Repo 浏览，继续使用完整 Trace 后一次提交，不强行加入三个动作。

### 结构测试

- 每个 Code 行为的 `l3_ids` 必须非空、唯一、可解析并保持确定性阅读顺序，名字锚定的结果必须存在于其中；
- `behavior_id` 和 `behavior_name` 全局唯一，相同输入重复构建结果完全一致；
- 所有证据块、边和位置都能解析；
- Code 边只能是 `calls`、`feeds`，Trace 边只能是 `precedes`、`continues`；
- 同一行为和关系不得重复；
- 所有实际选中的目录项都完整出现，图中全部行为都可通过 `query/read` 访问；
- SpecGap/SilentSwap 图中不存在 Test 或普通 Repo Document 节点；
- SpecGap/SilentSwap 完整目标文档恰好输入一次；
- FeedbackTrace 全部事件恰好出现一次且顺序不变，可输出的原 evidence ID 不重不漏；
- SilentSwap 的每条输出必须同时说明完整 `original_document` 中的语义 A 和已打开生产行为中的语义 B。

### 集成测试

每个 benchmark 至少覆盖：

- 一个完整样本；
- `query/read/finish` 的正常流程、零命中、延后 ID 和轮数耗尽；
- 消息数组重建、历史顺序及任务文档只出现一次；
- 131072 工作阈值、压缩到 98304、934464 物理边界、receipt、重新读取、`deferred_ids` 和 `context_unfit`；
- 非法动作和越界位置；
- 中断续跑；
- 最终 schema 校验。

### 配对评测

- 现有反复调试过的 5 条仅作为开发集；
- 从其余样本固定抽取 30 条验证集，冻结代码和 Prompt 后再评测；
- 剩余样本作为锁定测试集；
- 最终同时报告锁定测试集和 full-100 结果。

Baseline 与新方法必须使用相同：

- 样本、DeepSeek 模型及 disabled 模式；
- Qwen3.7 Judge 及 medium 模式；
- Prompt 外的温度、重试、超时和 `max_output_tokens=32768`；
- 统一后的 12/14 轮上限、目录/工具结果预算、1M 物理上下文和 128K→96K 工作记忆规则；
- 最终 schema 和同一 Judge 请求。

配对主结果按 intent-to-treat 统计：未在轮数内完成、协议失败或 `context_unfit` 的样本记为 0，同时单独报告完成率和各类失败率，不能只计算成功样本。

Gold 与评分范围直接复用桌面正式 baseline：SpecGap 使用 `gold_scope=implemented` 的 478 条 Gold；SilentSwap 使用 500 条生产 artifact Gold。不得改用当前工作区扩展出的 test-only SpecGap Gold 口径。

主要指标：

- FeedbackTrace：Joint；
- SpecGap：Qwen F1；
- SilentSwap：Rule Localization，Qwen Code Change 作为语义门槛。

“显著超过”只比较“统一 Agent + behavior graph”与“统一 Agent + Raw Repo”：10000 次 paired bootstrap 后，主要指标差值的 95% 置信区间下界大于 0；次要指标不得明显下降。桌面正式 baseline 不参与该因果比较。

同时记录每次 API 的实际输入/输出 token、调用次数、重试和延迟。质量达标后，再要求新方法的平均及中位总输入 token 低于配对 baseline。

## 5. 迁移与清理

- 使用 `schema_version: 2`，旧模块五至七缓存全部失效；
- 移除 `candidate_l3_ids` 缺失时依靠 explanation 词重叠的 fallback；
- 停用旧 hypothesis、candidate graph 和 SpecGap CASE 文本流程；
- 新流程验证完成后，先列出 `.tmp` 精确目录，再只保留复现实验所需的 manifest、原始响应、prediction、Judge 结果和 summary，清理其余旧轮次中间文件。
