# BEG 文件证据包实施计划

## 1. 方案结论

Raw baseline 把找证据、拼上下文和判断语义都交给模型；BEG 先用行为图把 Repo 确定性压缩为少量相关、语义完整的连续源码，并把每条行为绑定到支撑它的原始证据行，再让模型判断。它的本质优势是减少 LLM 不擅长的搜索与注意力负担，同时把答案定位到真正支撑条件和结果的精确代码位置，因此才可能用更少 token 获得更高的语义和定位分数。

```text
任务文档 + 紧凑文件目录
→ read(F) 返回局部 BEFORE 契约 + 静态相关实现位置 + 连续当前源码
→ 模型比较文档与当前实现
```

图只在后台检索、定位和组织证据；模型前台仍阅读带原始行号的连续源码，不需要理解图 ID。BEG 不创造 Raw 中没有的事实，优势是否成立最终以同一 AgentLoop 下“质量高于 Raw 且总 token 更少”的配对实验为准。

最新指定同五条正式 Judge 表明：第一版 BEG 相对 Raw 的精确定位 `+0.016`，LLM 位置判断不变，token 少 `4.2%`、轮数少 `18.8%`，但语义正确率反而 `-0.200`。目录和 inline focus 已经减少搜索，却没有把一条契约的全部实现操作组织完整。后续不再重做目录，重点改文件证据包中的局部 BEFORE—AFTER 对齐和跨位置实现路径聚合。

本轮只修改 SilentSwap 的文件证据包及其 Prompt。SpecGap 保持现状；FeedbackTrace 没有 Repo 文件，继续使用 Trace 行为图。

## 2. SpecGap 实际删除了什么

### 2.1 全量审计

审计覆盖桌面正式 SpecGap 的 100 个样本、484 条 Gold condition。BEG 的 hidden Gold 与源数据 `2_deleted_parts.json` 中的 condition、原文和删除 span 在 100 个样本上完全一致。

- 482/484 条 condition 有实现位置，共 1556 个位置、886 个唯一 Symbol；
- 每条 condition 平均删除 2.67 个精确 span；
- 严格按完整 Symbol 名匹配，删后文档仍提及相关 Symbol 的有 385/484，约 79.5%；
- 按结构化接口锚点匹配，包括完整名、反引号名称、函数或类签名，删后仍有相关 Symbol 锚点的有 449/484，约 92.8%；
- 剩余 35 条在删前和删后都没有对应的显式接口锚点，其中 27 条涉及私有 Helper；它们不是“整个 Symbol 被删除”，而是文档原本就通过公开入口或自然语言描述行为；
- 只有极少数首要实现 Symbol 出现“删前明确提及、删后不再提及”。

代表样本：

- `sg_002/kc_001`：`filter_resources` 的导入、标题和签名仍在，只删除 `fullmatch` 语义；
- `sg_003/kc_006`：`raiser` 的接口章节仍在，只删除准确的 `TypeError` 消息；
- `sg_039/kc_003`：`Deployment` 与 `ready` 签名仍在，只删除缺少字段时传播 `KeyError` 的行为。

### 2.2 确定结论

SpecGap 通常不是把整个 Symbol 从文档中删掉，而是保留 Symbol/API 的主体说明，删除其中一个条件、默认值、异常、顺序、依赖选择或数据流事实。少量 Gold 落在文档从未直接命名的私有 Helper 中。

因此，文档命中只能提高文件优先级，不能作为是否保留文件或行为的硬过滤条件。正确比较单位是：

```text
文档提到的 Symbol/API
↔ 该 Symbol 及直接依赖在代码中实际包含的 Behavior
↔ 支撑这些 Behavior 的精确 Evidence
```

以上比例来自确定性词法匹配，不是人工语义分类。完整名口径会低估别名和签名，结构化锚点口径可能对短通用名称有少量假阳性，因此 79.5%～92.8% 是稳健范围。

## 3. Evidence、Behavior、Symbol 与 File

### Evidence：原始事实

Evidence 是带文件、作用域、精确行号和原文的代码片段。它回答“代码在哪里写了什么”，用于校验展示内容和最终定位。Evidence 不代表存在问题。

### Behavior：代码路径

Behavior 是由若干 Evidence 支撑的一条结果路径，例如：

```text
条件 → 必要操作或数据来源 → return / raise / state_write / external_call
```

Behavior 的作用是让程序确定性识别值得检查的条件—结果组合，并检查证据包是否同时覆盖触发条件和结果。它不是源码块，也不是模型生成的摘要。

### Symbol：源码所有者

Symbol 是 Behavior 所属的自然代码定义，例如函数、方法、类成员、配置段或顶层完整语句。一个 Symbol 可以包含多个 Behavior。Symbol 的作用是给模型提供理解这些行为所需的连续局部源码，而不是替代 Behavior 或 Evidence。

### File：模型导航单位

File 包含多个 Symbol，是模型最熟悉、也最适合目录浏览和批量读取的单位。新方案只让模型使用文件级 `F` ID。

关系为：

```text
File contains Symbol
Symbol owns Behavior
Evidence supports Behavior
```

文件证据包只是运行时展示视图，不是新的图节点：图在后台选择 Behavior 和依赖，最终仍展示原文件中的连续源码。

### 前后台边界

- 模型的 `query/read` 只接收和返回文件级 `F` ID；
- Symbol 名只用于目录提示和源码区域标题，不能单独 `read(S)`；
- Behavior 只用于后台选择源码区域、检查条件与结果是否共同可见，不生成独立或穷举的模型可见行为清单；SilentSwap 仅显示少量、紧贴源码的中立 focus；
- Evidence 只用于原文校验、行号导航和最终 grounding，不生成独立 Evidence 目录；
- 不再存在模型可见的 `P` 包、`S` 目录或 `B/ev` 查表过程。

因此，Symbol、Behavior 和 Evidence 仍是构图与校验结构，但不是模型的浏览协议。模型只选择文件，然后阅读连续原始源码。

## 4. 文件目录

### 4.1 哪些文件进入初始目录

文件只有在能够生成非空证据包时才进入初始目录。证据包可由以下通用信号产生：

1. 任务文档直接命中的 path、Symbol、调用名或配置键；
2. 公开接口中的异常、默认值、返回、外部副作用、状态或配置等可观察 Behavior；
3. 与已选 Behavior 存在直接 `calls/feeds` 关系的候选依赖；
4. 后续 `query` 精确命中的生产代码。

文档链接只用于排序，不是过滤条件。所有保留的生产代码始终可由 `query` 找到。

SpecGap 的 625 个去重 Gold condition—file 位置中，611 个是生产源码（97.76%）；涉及配置或构建文件的 11 条 condition 也都同时具有生产源码位置。因此排序冻结为：

1. 文档直接命中的文件最先出现；精确 path/完整限定名优先，多个文档 section 轮流取一项，避免一个长 section 占满目录；
2. 其余文件先按是否与直连文件存在 `calls/feeds` 一跳关系排序；
3. 同一图距离内按生产源码 → 可执行文件/运行时模板 → 运行时配置排序，再用可观察 Behavior 和路径接近度打破平局；
4. `requirements`、`pyproject`、`setup` 等构建元数据最后，只在文档直接命中或模型精确 `query` 时优先读取。

100 条 Gold—Behavior 重叠审计还表明，相关结果以 `state_write` 和 `return` 为主，`output/external_call` 并不更重要。因此 Behavior 类型不再主导排序，只作为上述规则之后的次级信号。

两类目录都先完整放入预算内的文档直连文件，再加入极小的图桥接通道：SpecGap 最多 2 个，SilentSwap 最多 1 个。额外文件按与直连文件的 `calls/feeds` 连接、artifact 类型和路径接近度统一排序，绝不挤掉文档直连文件。该规则来自全量审计：SilentSwap 原先未进入目录的 4/500 个 Gold 文件集中在两个样本，而各加入 1 个桥接文件分别选中了模板加载器和遗漏实现文件。该上限只限制初始注意力，不限制后续查询。

### 4.2 目录的固定格式

```text
[[FILE EVIDENCE DIRECTORY]]

[FILES LINKED TO THE TASK DOCUMENT]
Read ID | File | Related document sections | Code hints (count; examples)
F0001 | src/patchy/api.py | Guarded source replacement | 6 document-linked symbols; examples: replace; _assert_ast_equal
F0002 | src/patchy/cache.py | Cache behavior | 1 document-linked symbol: PatchingCache.store

[ADDITIONAL FILES WITH OBSERVABLE BEHAVIOR]
Read ID | File | Related document sections | Code hints (count; examples)
F0003 | src/patchy/errors.py | no direct match | 1 selected symbol: format_error

[[END FILE EVIDENCE DIRECTORY]]
```

上例使用 SpecGap 的分区名；SilentSwap 的第一分区固定为 `[FILES LINKED TO THE ORIGINAL DOCUMENT]`。SilentSwap 没有额外文件时省略第二分区。

目录只展示四个不可再合并的字段：

- `Read ID`：稳定文件 ID；
- `File`：仓库相对路径；
- `Related document sections`：最多 3 个命中的文档 section 名，帮助模型回到完整任务文档比较；
- `Code hints`：先给出文档直连或后台选中 Symbol 的总数，再展示最多 4 个跨文档 section 轮流抽取的示例。示例只说明文件用途，不是完整检查清单。

四列分别解决“怎么读、读哪个文件、和文档哪里比较、文件大致包含什么”。Symbol 数量只用来明确示例并不完整，Prompt 明确不要求逐项读满。

目录明确不展示：

- `B/S/ev` ID、原始图边和内部评分；
- Behavior 类型、结果行号和其他容易被误读为完整行为清单的信号；
- 源码、Evidence 原文和大段文档；
- hypothesis、缺口结论或“建议汇报”等标签；
- 文件中的全部 Symbol 或全部 Behavior；
- Test、普通 Repo Document 和重复文件行。

每个文件恰好对应一个 `F`，在目录最多出现一次。目录预算保持 3072 token；放不下的文件不删除，只保留在查询索引中。

## 5. 文件证据包

### 5.1 设计结论

目录已经能够把相关文件提前，当前短板在 `read(F)`：旧版为一个文件枚举大量 `return/raise/state_write` 事件。真实错例中，正确源码已经出现，模型仍会被上百条事件提示分散注意力，或者把 helper、wrapper、常量和调用点误当成互不相关的变化。

因此 SilentSwap 冻结为“少量逐字 BEFORE 紧邻少量连续 AFTER”。图仍在后台选文件、owner、操作行和直接依赖；模型前台不再阅读 Behavior 清单。

### 5.2 固定结构

```text
[[FILE EVIDENCE]]
Read ID: F0001
File: src/patchy/api.py

[[BEFORE: ORIGINAL DOCUMENT]]
Section: Guarded source replacement
<original_document 中与该文件直接链接的逐字原文>
[[END BEFORE]]

[[AFTER: CURRENT SOURCE]]
Focus: replace | Lines: 48-59
Read first: lines 52, 59 (navigation only; inspect the continuous source).
48 | <当前连续原始源码>
...
59 | <当前连续原始源码>
[[END AFTER]]

[[RELATED FILES]]
- Calls: F0002 | src/patchy/cache.py
[[END RELATED FILES]]
[[END FILE EVIDENCE]]
```

字段只承担四个作用：

- `Read ID / File`：标识本次读取的文件；
- `BEFORE`：重复少量 `original_document` 逐字原文，避免模型回到长文档搜索；
- `AFTER`：显示完整、连续、带原始行号的当前源码；
- `Read first`：把注意力放到图找到的操作行，不描述操作含义，也不暗示存在 swap。

不再输出 `Document sections` 摘要、`Behavior focus` 事件列表、`supporting lines` 解释、全局行为路径块或 `B/S/ev` ID。

### 5.3 确定性选择规则

1. 以 `original_document` 直接命中的 Symbol 为种子，同时保留同文件一跳 `calls/feeds` 的 caller、wrapper、helper、常量或字段；关系双向用于导航，但不代表这些位置一定属于同一个 swap。
2. 每个入选 owner 显示完整连续源码。条件、操作和结果不得拆成孤立行；重叠或相邻 owner 按原文件顺序合并，同一源码行只出现一次。
3. SilentSwap 不再使用“文件少于 400 行就整文件发送”。文件长度不决定相关性；未被文档或一跳关系选中的 Symbol 不进入证据包，但始终可以通过 `query` 找到。SpecGap 的读取规则不变。
4. `Read first` 只汇总当前源码区域内已有 Behavior 的结果行号。它不显示 `Returns/Raises` 等解释，模型必须阅读连续源码确认执行顺序、异常和最终结果。
5. 同一文档 section 在一个文件包中只显示一次。原文必须是任务文档的逐字子串，并在长度受限时优先于句中截断保留完整 Markdown 段落。
6. 跨文件关系只列 `F` ID、路径和 `calls/feeds` 方向，不自动注入邻居源码。需要时模型再次 `read`。
7. 较早读取被压缩时，只保留逐字 BEFORE 和已选中的精确当前源码行，不保留模型生成的行为说明。

### 5.4 为什么能同时改善语义与定位

语义错误主要来自事件过多和上下文割裂。本结构把文档 BEFORE 与连续源码 AFTER 放在同一视野，并让同文件直接依赖共同出现；模型不再从几十条行为解释中猜执行过程。

定位收益来自 `Focus` 的 owner 范围和一行 `Read first` 精确行号。图负责缩小位置，连续源码负责解释含义，两者不再互相重复。目录、AgentLoop 动作、输出 schema、RawBackend、Judge 和 FeedbackTrace 均保持不变。

## 6. 各模块的作用与改动

| 模块 | 作用 | 本轮改动 |
|---|---|---|
| Evidence Intake | 保存完整生产文件和精确原文位置 | 不变 |
| Behavior Atomization | 提供条件—操作—结果证据和操作行号 | 不变；不再把全部 Behavior 解释展示给模型 |
| Relation Linking | 提供确定性的 `calls/feeds` | 不变 |
| Graph Assembly | 形成 Symbol 图并保存 Behavior 归属 | 不变 |
| 模块五：文件目录 | 一文件一 `F`，排序并生成初始目录 | 排序和格式不变；文档命中外层 owner 时，把内部真实边端点及其同文件一跳依赖纳入读取闭包 |
| 模块六：文件证据包 | 把文件图转为模型可读证据 | 重写为逐字 BEFORE、连续 AFTER、紧凑定位和跨文件导航 |
| 模块七：AgentLoop | 执行 `query/read/finish` | 只调整 SilentSwap Prompt 对新证据包的说明 |

Python import、顶层常量和映射不新增公开图边。模块六从当前可见源码中确定性补入被选 owner 直接读取的唯一顶层绑定；无法唯一解析时不猜。AgentLoop 动作、输出 schema、RawBackend、Judge、SpecGap 和 FeedbackTrace 均保持不变。

## 7. 实施顺序

1. 保持现有文件目录和排序；
2. 修复外层 Symbol 命中时的嵌套 wrapper/helper 同文件闭包；
3. 仅在 SilentSwap 删除小文件整文件规则，统一输出选中 owner 和直接绑定；
4. 渲染最小 BEFORE–AFTER 证据包，并同步历史 receipt；
5. 更新 SilentSwap Prompt 和针对性测试；
6. 先离线验证，再用固定开发样本验证语义、定位和 token。

实现不得使用 Gold、删除内容、patch、修改前代码、样本名特例或模型摘要。

## 8. 离线验证

1. 每个 BEFORE excerpt 都是 `original_document` 的逐字子串，并保留完整 Markdown 块；
2. 文档命中外层 decorator 时，参与真实边的 nested wrapper、同文件 helper 和直接 import/常量绑定均可见；
3. 未命中且无一跳关系的 sibling 不进入证据包，但仍能被 `query` 找到；
4. 每个 AFTER 区域连续、带真实行号，条件、操作和结果不被拆开；
5. 同一源码行、文档 excerpt 和 related file 在一个文件包内不重复；
6. 模型输入中不再出现 `Behavior focus`、`supporting lines`、全局行为路径或内部图 ID；
7. SpecGap 的目录、证据格式和结果不受 SilentSwap 分支改动影响；
8. 相同输入重复构建得到相同的 F ID、excerpt、源码区域和排序。

真实回归重点覆盖两类已知机制：外层 decorator 的边由 nested wrapper 发出；同一公开行为由 helper 与多个 wrapper 共同实现。测试验证通用结构，不在生产代码中加入样本规则。

## 9. 验证图是否真正带来优势

最新正式 Judge 仅比较指定的同五条 Prediction。Raw 复用 `dev_ss_file_v17/raw`，第一版 BEG 使用 `dev_ss_first_frontend_v18/graph`；两者的 split、模型请求配置、Prompt 和正式 Judge profile 相同：

| 指标 | AgentLoop Raw | 第一版 BEG | 差值 |
|---|---:|---:|---:|
| 精确定位 | 0.639 | 0.654 | **+0.016** |
| LLM 位置判断 | 0.600 | 0.600 | 0 |
| 语义正确率 | **0.600** | 0.400 | **-0.200** |
| Prediction token | 356,461 | **341,404** | **-4.2%** |
| 交互轮数 | 16 | **13** | **-18.8%** |
| 失败数 | 0 | 0 | 0 |

这五条只说明第一版的定位和效率方向正确，不能把 `-0.200` 全部归因于证据包：正式 Judge 在个别相近输出上存在判定波动。可稳定复现的结构问题有两个：正确源码已经读到时，长事件清单仍会淹没关键操作；同一变化跨 helper、wrapper、绑定或连续表达式时，旧包没有把完整实现路径共同展示。

新版本一次性验证下面三个假设：

1. 删除全量事件解释后，模型更容易按真实执行顺序理解当前源码；
2. 逐字 BEFORE 紧邻连续 AFTER 后，模型更容易建立正确的比较方向和具体语义；
3. owner 范围与紧凑操作行号仍能保留并扩大图的定位收益。

先用已分析的五条只做离线结构回归，不再根据它们增加规则；随后在未参与设计的新五条上运行一次配对试测。记录语义、定位、首次读到正确文件的轮次、读取源码行数、总 token 和失败类型。只有独立样本同时达到语义高于 Raw、定位不下降且 token 更少，才冻结 SilentSwap 并进入 30 条评测。
