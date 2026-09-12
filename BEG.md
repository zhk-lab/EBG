# BEG：Behavioral Evidence Graph

## 1. 方法目标

BEG 以确定性规则将代码仓库或执行轨迹中的分散证据组织为行为证据图，支持模型对长程 Agent 的行为进行分析与问题定位。 代码仓库和执行轨迹的总称叫什么。

             Agent Context
             /           \
      Workspace       Execution History
        (Repo)             (Trace)
             \           /
          确定性提取与组织
                  ↓
     Behavioral Evidence Graph
       原始证据 · 行为单元 · 关系
                  ↓
           面向模型的证据呈现
             /           \
      Repo 局部图      Trace 任务图
             \           /
              模型监督
          行为分析 · 问题定位

## 2. 模块一：Evidence Intake

### 作用

Evidence Intake 确定 BEG 可以使用的事实边界，并把每条事实保存成可重新定位的 **Evidence**。它直接决定定位上限；如果这里丢失原文或位置，后续构图无法补回。

### Evidence 定义

Evidence 是输入中最小的、连续的、能够独立定位一个事实的原始片段。它只说明“原始输入在哪里写了什么”，不表示存在缺口、替换、违规或授权问题。

```json
{
  "evidence_id": "E0012",
  "source_type": "code | trace",
  "locator": {},
  "content": "逐字原文"
}
```

### Repo

Repo 只接收能够决定运行行为的生产源码、运行时模板、可执行脚本和行为配置。普通 Repo 文档、测试、CI、构建产物和第三方目录不进入行为图。SpecGap 的完整 `document_after` 和 SilentSwap 的完整 `original_document` 独立保留，始终作为模型的语义基准。

Repo Evidence 的提取策略：

1. Python 文件用 AST 拆分。定义头、条件、调用、赋值、import、`return`、`raise` 等完整语句各形成一条 Evidence，并记录所在函数、方法、类或 `<module>`；
2. 多行语句必须完整保留。同一位置的语句只保存一次，不为不同 Behavior 重复复制；
3. 非 Python 源码和脚本不拆分，整个文件保存为一条 `<module>` Evidence；
4. 配置文件按配置项提取；无法识别配置项时，保存整个文件；
5. 运行时模板按非空原文行提取，并记录所在的 `block` 或 `macro`；
6. Evidence 必须保留逐字原文，不扩成整个 Python 函数。后续模块通过 `path + symbol` 读取完整函数上下文。

任务文档不生成 Evidence，也不成为行为图节点。SpecGap 的完整 `document_after` 和 SilentSwap 的完整 `original_document` 始终原样保留；模块五直接用其中出现的 path、Symbol、调用名和配置键匹配代码图，不再生成文档片段对象或额外 ID。

Code locator 固定包含：

```json
{
  "path": "src/api.py",
  "symbol": "Client.run",
  "line_start": 20,
  "line_end": 24
}
```

### Trace

Trace Evidence 的提取策略：

1. 按输入顺序保留 cutoff 前的全部用户、Agent、Tool 和前置系统事件，每个输入事件形成一条 Evidence；
2. 事件内容不拆分、不摘要、不改写；工具调用和工具结果若是两个输入事件，就分别保存；
3. Locator 保存 `turn`、`event_index`、`event_type`、`tool_name` 和 `original_evidence_id`，用于重新定位原始事件；
4. 失败调用、重复尝试和后续修正全部保留，不根据结果预先过滤事件。

### 确定规则

1. 不读取 Gold、patch、修改前代码或 cutoff 后事件；
2. `content` 必须是可见输入的逐字子串；
3. ID 由来源和位置确定性生成；
4. 相同输入重复运行得到相同 Evidence 集合和顺序。

## 3. 模块二：Behavior Atomization

### 作用

Behavior Atomization 把零散 Evidence 组合成最小完整行为，使模型看到“在什么条件下，执行了什么，产生了什么结果”。它是提升语义指标的第一层核心：Evidence 负责事实，Behavior 负责恢复一条可理解的执行路径。

### repo

#### Symbol 字段

Repo 中的 `symbol` 与 `path` 一样，是代码中确定存在的定位字段，不是需要额外抽取或显式构建的节点。它表示 Evidence 或 Behavior 所在的代码作用域，如函数、方法、类成员、配置项或模块语句；顶层语句统一记为 `<module>`。程序通过语法结构确定该字符串，相同输入必须得到相同结果。

确定规则是：函数和方法使用词法限定名，如 `Client.run`；类级成员使用其所属类与成员名；配置项使用可稳定定位的键名；无法归入更小命名作用域的顶层代码使用 `<module>`。每条 Code Evidence 恰好有一个 `path + symbol`，不调用 LLM 猜名称，也不为匿名控制块编造 Symbol。

`symbol` 对 BEG 有四个作用：

1. **防止错误组合**：模块二只允许相同 `(path, symbol)` 的 Code Evidence 组成一个 Behavior，避免把不同函数的条件和结果拼在一起；
2. **补全连续上下文**：模块六用 `path + symbol` 找回该作用域的完整范围和原始源码，帮助模型理解变量、相邻分支和默认路径；
3. **解析跨位置关系**：模块三用源端和目标端的 `path + symbol` 确定 `calls/feeds`，不依赖模型猜测调用关系；
4. **去重和检索**：相同 `path + symbol + lines` 的源码只发送一次，多个 Behavior 共用该上下文，减少重复 token。

Symbol 本身不产生新事实，也不表示存在问题。图不保存 Symbol 节点；`symbol` 只作为 Evidence、Behavior、关系端点和源码上下文中的可读定位字段，不生成 `S` ID。Behavior 仍是主要语义节点，`calls/feeds` 则用两端的 `path + symbol` 表达跨作用域关系。

#### Behavior 定义与 Schema

Repo Behavior 是一个 Symbol 内、以一个可观察结果为锚点的一条控制路径：

```text
直接触发条件 → 必要操作或数据来源 → 可观察结果
```

```json
{
  "behavior_id": "B0107",
  "artifact_kind": "source",
  "path": "src/api.py",
  "symbol": "Client.run",
  "symbol_lines": [18, 40],
  "result_type": "return | raise | state_write | output | external_call | yield",
  "trigger_evidence_ids": ["E0012"],
  "operation_evidence_ids": ["E0013"],
  "result_evidence_ids": ["E0014"]
}
```

Repo Behavior 的构建规则：

1. 以每个可观察结果为锚点，只组合相同 `path + symbol` 下的 Evidence；
2. 互斥路径分别构建，`trigger` 保存直接条件，`operation` 只保留产生结果所需的操作；
3. `result` 必须存在；`trigger` 和 `operation` 可以为空，不为无结果代码编造 Behavior；
4. 全程使用确定性语法和控制流规则，不生成摘要或问题判断；跨 Symbol 关系交给模块三。

### Trace

#### Task Scope 构建规则

任务归属在 Behavior 拆分之后判断；同一轮中的不同 Behavior 可以属于不同 Task Scope。构建时只使用以下明确可观察信号。

可观察信号分为三类：

1. **语句信号**：`continue`、`go on`、`next`、`failed`、“仍然失败”等表示继续当前任务或反馈其结果；`shift attention`、`move on`、“停止 X，改做 Y”等表示建立新任务；“使用上一个任务的结果”表示建立新任务，并由 M3 使用 `informs` 连接旧任务。
2. **稳定标识**：task/job/run ID、Issue/PR、trial/shard、分支和完整链接。相同标识与一个已有 Task 唯一匹配时，恢复该 Task；Tool 或 Assistant 新产生的标识先记录到当前 Task。单个文件名、函数名、普通 URL 或主题词不是稳定标识，不能单独决定归属。
3. **交互结构**：Assistant 提问、给出选项或要求日志后，紧随的用户回答属于同一 Task；Tool 显示 `submitted/running/queued/waiting` 后，后续进度询问属于同一 Task；同一 Prompt 中可分别对应不同稳定标识和 Tool 操作的要求属于不同 Task，共享同一不可拆操作结果的要求属于同一 Task。

判定时先识别任务切换，再匹配稳定标识，最后使用承接语句和相邻交互结构；只有主题相似或时间相邻时，不合并 Task。

这些信号均来自可见 Trace，不能读取 Gold，也不能用任务摘要或重要性判断辅助划分。

`task_id` 不承载摘要、重要性或问题判断，正如 Repo 的 `symbol` 不承载代码是否正确的判断。

`sequence_index` 只表示 Behavior 在所属 Task Scope 中的原始先后位置。Behavior 之间不建立 `precedes`、`continues`、`depends_on` 或 `leads_to`：共同任务由 `task_id` 表达，时间顺序由 `sequence_index` 表达。

Trace 角色按事件来源确定：user prompt 只能进入 `demand`，Tool 事件只能进入 `action`，Assistant 回复只能进入 `response`。这将“实际做了什么”和“向用户声称什么”分开，避免同一 Tool Evidence 同时被标成 operation 和 result。

#### Trace Behavior 

Trace Behavior 表示一条可辨认的需求履行链：用户提出的具体要求是 `demand`，Agent 实际执行的 Tool、检查、修改或验证是 `action`，Assistant 给出的决定、承诺或结果汇报是 `response`。一轮可以产生多个 Behavior；只有 demand 与后续 action/response 能直接对应时才拆分，无法可靠区分时保留复合 Behavior。没有 action 和 response 的孤立用户事件不形成 Behavior。

```json
{
  "behavior_id": "T0021",
  "task_id": "K0003",
  "sequence_index": 2,
  "start_turn": 20,
  "end_turn": 25,
  "demand_refs": [{"evidence_id": "E0201", "char_range": [35, 92]}],
  "action_evidence_ids": ["E0202", "E0203"],
  "response_evidence_ids": ["E0204"]
}
```
代码通过固定词表和可逐字匹配的对象进行识别，不让模型猜测：

1. **Action**：读取 `tool_name`、Tool 参数和结果，得到“操作类型 + 稳定对象”。固定映射如 `Read/Grep → inspect`、`Edit/Write → modify`、`test/pytest → verify`、`commit/push → publish`；文件路径、Symbol、测试名、分支和 job ID 作为稳定对象。同一对象上的读取、处理、验证和失败重试归入一个 Action 组。
2. **Demand**：按最外层编号、同级项目符号或独立问句切分 User Prompt；嵌套条目只是上级要求的细节。再提取“请求类型 + 稳定对象”，如“修改 `src/api.py` → `modify + src/api.py`”。
3. **Response**：从 Assistant 内容中提取“结果类型 + 稳定对象”。完整对象未重复时，也可使用可唯一匹配的命令、测试名等明确组成词。能与某个 Demand 对应的回复归入该 Behavior；一段回复明确覆盖多个 Demand 时分别记录对应的逐字范围，不强行唯一归属。`done`、过渡语和整体总结不单独产生 Behavior。

同一任务的一轮或多轮 Behavior 使用相同 `task_id`，并按 `sequence_index` 恢复原始顺序。任务归组只使用可见的任务目标、明确标识符、操作对象和直接承接表达；相同文件不能单独决定任务相同。`char_range` 只在一条 user Evidence 中能够明确定位某项 demand 时使用，原始 Evidence 不拆分或复制。

在 300 个正式样本（SpecGap、SilentSwap、FeedbackTrace 各 100 个）上，模块一平均生成 482.47 条 Evidence，模块二平均生成 268.57 条 Behavior。

## 4. 模块三：Relation Linking

### 作用

Relation Linking 保存单个 Behavior 无法表达的跨位置结构。边不是装饰信息，它有四个直接消费者：

1. **语义理解**：把 caller 与 callee、数据生产者与消费者、早期反馈与后续行为连在一起；
2. **局部图检索**：模块六根据边自动补齐当前行为的直接依赖，减少模型手工多跳；
3. **目录排序**：模块五把与文档直连文件存在边的文件提高优先级；
4. **精确定位与 token 控制**：边的支撑 Evidence 指出连接发生的位置，同时只展开直接邻居而不是读取整个 Repo 或 Trace。

### Repo

Repo 的运行关系只保留两类作用域边：

- `calls`：源代码作用域中存在可唯一解析的直接调用，目标是被调用作用域；
- `feeds`：源代码作用域的明确返回值、产出或状态写入被目标作用域使用。

Repo 中，底层 `calls/feeds` 仍连接由 `path + symbol` 标识的 Code Node，而不直接连接 Behavior。模块六读取时再根据边的支撑 Evidence，把每个端点映射到 Evidence 所属的 Behavior：能够确定时只展示对应 Behavior，命中多个时全部保留，不能确定时回退到完整 Symbol。这样既保留稳定、可验证的 Symbol 级图结构，又避免每次沿边都返回对侧 Symbol 的全部 Behavior。

```json
{
  "edge_id": "R0031",
  "from": {"path": "src/api.py", "symbol": "Client.run"},
  "type": "calls | feeds",
  "to": {"path": "src/output.py", "symbol": "emit"},
  "via": "result | field name | null",
  "evidence_ids": ["E0013"]
}
```

### Trace

Trace 只保留 Task Scope 之间稀疏的 `informs/supersedes`。`informs` 表示目标任务明确使用了来源任务产生的结果、决定、约束或产物；`supersedes` 表示来源任务明确停止、撤销或替换了目标任务的方案、运行或决定。Behavior 不建立关系边：所属任务由 `task_id` 表达，任务内顺序由 `sequence_index` 表达。

### 边的构建规则

1. Repo 中可唯一解析的直接调用生成 `calls`；明确的返回值、模板产出或状态写入被其他作用域使用时生成 `feeds`；
2. Trace 只有在关系语句、稳定标识和前后 Evidence 同时成立时生成 `informs/supersedes`；时间相邻、对象相同或文件相同都不能单独生成边；
3. 每条边必须有原始 Evidence 支撑；动态调用、歧义关系、传递关系和结论型关系不生成；
4. 相同端点和类型的边合并 Evidence，并按确定顺序生成稳定 ID。

在 300 个正式样本（SpecGap、SilentSwap、FeedbackTrace 各 100 个）上，模块三平均生成 71.56 条 Edge。

## 5. 模块四：Graph Assembly

### 作用

Graph Assembly 将 Evidence、Behavior、连续源码上下文和关系边形成统一的静态行为图。它保留 L3 原始事实、L2 局部行为和由关系边形成的 L1 行为拓扑，但不生成 Symbol 节点或重复的 L1 摘要节点。

### Repo Graph Schema

```json
{
  "input_id": "",
  "benchmark": "specgap | silentswap",
  "evidence": [],
  "behaviors": [],
  "source_contexts": [
    {
      "artifact_kind": "source",
      "path": "src/api.py",
      "symbol": "Client.run",
      "lines": [18, 40],
      "source": "该作用域的连续原始源码"
    }
  ],
  "edges": []
}
```

### Trace Graph Schema

```json
{
  "input_id": "",
  "benchmark": "feedbacktrace",
  "evidence": [],
  "behaviors": [],
  "edges": []
}
```

Repo 的 L1 是 `calls/feeds` 连接出的代码作用域拓扑，Behavior 通过自身的 `path + symbol` 落在相应作用域中；Trace 的 L1 是 `informs/supersedes` 连接出的稀疏 Task Scope 拓扑，Behavior 通过 `task_id` 归入相应作用域。包含关系和顺序由字段表达，不重复生成装饰性边。

### 校验规则

1. Repo Behavior 引用的 Code Evidence 必须具有相同 `(path, symbol)`；Trace 的 demand/action/response 必须分别引用 user/Tool/Assistant Evidence，并位于自己的 turn 范围内；
2. Behavior 引用的 Evidence 全部存在；Repo 的结果位置还必须位于其 `symbol_lines` 范围内；
3. Repo 边两端的 `(path, symbol)` 都能匹配至少一个 Behavior 和一份源码上下文，支撑 Evidence 也必须存在；
4. 同一 Evidence 原文以及相同 `(path, symbol, lines)` 的源码上下文只保存一次；
5. 节点、边、ID 和顺序可以由相同输入重复生成；
6. 图中不得出现 Gold、模型候选或问题结论。

Graph Assembly 只形成事实图，不决定模型先看哪个文件，也不进行 token 截断。

## 6. 模块五：Ranked Directory

### 作用

Ranked Directory 把完整图转换成少量检索入口。它主要提高首次定位成功率并减少无效 search/read；目录不承载语义判断，也不替代图。

### Repo

一个生产文件对应一个稳定 `F` ID。目录只显示：

```text
Read ID | File | Related document sections | Code hints
F0001 | src/api.py | Public API | 3 linked symbols: run; validate; emit
```

排序规则：

文档只按明确的 path、完整或限定 Symbol（类名包含其成员）、显式调用名、配置键和可唯一定位的反引号代码术语匹配；名称有歧义时不猜测。

1. 完整任务文档直接命中的文件优先；
2. 与直连文件存在一跳 `calls/feeds` 的文件随后；
3. 同级按生产 artifact、可观察 Behavior 数量和稳定路径排序；
4. 一个文档 section 不能独占目录，多个 section 轮流进入预算；
5. 未进入初始目录的文件仍保留在完整查询索引中。

目录不展示 Edge、Evidence 原文、Behavior 结论或内部评分。目录预算固定为 3072 token，作用是帮助模型选择 Local Graph root，而不是让模型仅凭目录作答。

### Trace

FeedbackTrace 不进入 Ranked Directory。

## 7. 模块六：Linear Graph Serialization（Repo / Trace）

### 作用

模块六把后台图整理成模型可以连续阅读的证据。Repo 通过 `read(F0001)` 取得对齐式 compact 局部图：直接命中的 Root、关系说明和邻居源码具有不同的明确标签；Trace 不做检索，而是把全部 Behavior 按 `task_id` 聚合为完整 Task Scope JSON。模型无需在 Evidence、Behavior 和 Edge 三张表之间来回查找。

### Repo：构建与读取规则

```text
文件 F0001
  ↓ 固定与读取意图对应的最小 Root 集合
[DIRECT ROOT] 完整 Symbol 源码
  ├─ [CONTEXT via calls] → 对应 Behavior 或完整 Symbol
  └─ [CONTEXT via feeds] → 对应 Behavior 或完整 Symbol

其他作用域通过 search → read(F0001.S0002) 单独读取
```

1. `read(F0001)` 先选择 Root。完整或限定 Symbol 的文档命中优先，其次是可唯一定位的代码 Evidence，父作用域只作兜底；没有直接命中时只选择一个代表 Symbol。Root 集合在边展开和文本渲染前固定，内容变短后不能用剩余预算补入其他 Root；
2. `<module>` 只有在文档明确对应模块级常量、配置或注册语句时才作为 Root，并且只返回命中的模块级源码片段；
3. Root 默认展示完整连续 Symbol 源码。只选择与 Root 直接相连、且有 Root Behavior Evidence 支撑的 `calls/feeds`；每个 Root 最多展开四条边且只展开一跳；
4. 底层边仍是 `Symbol → Symbol`。读取时将边的端点 Evidence 与对侧 Behavior 引用的 Evidence 匹配：能确定具体 Behavior 时只返回这些 Behavior 的源码，命中多个时全部保留；Behavior 清单不完整或无法确定时返回完整 Symbol，不能为节省 token 强行任选一个分支；
5. `calls` 通常只能精确定位调用者一端；被调用者没有返回侧 Evidence 时会回退到完整 Symbol。`feeds` 通常同时具有生产端和消费端 Evidence，更容易精确到两侧 Behavior；
6. 邻居优先保留 Root 调用的目标、向 Root 提供数据的来源，以及有直接文档命中的相邻作用域。相同邻居只展示一次，源码按原文件顺序排列，相同行在一次 read 中全局去重；
7. 未展示的函数或方法不会被删除。Agent 可以通过 `search` 获得 `F0001.S0002` 形式的专用 Read ID，再单独读取该作用域。

### Repo：Root-Aligned Compact Code Graph

模型不读取内部缩进 JSON，也不看到 Behavior ID 或 Evidence ID。Root、文档锚点和结构邻居按以下格式输出：

```text
[DIRECT ROOT]
Doc: Public API / Client.run
src/api.py::Client.run@18-40
18 | def run(...):
...
40 |     return result

[CONTEXT via calls]
Client.run@24 calls src/output.py::emit@8-19
8 | def emit(...):
...
19 |     return None
```

`[DIRECT ROOT]` 表示该源码由本次读取意图直接选中。只有文档明确出现完整 Symbol 时才显示 `Doc:`；顶层函数可以使用明确的函数名。Evidence 词、父作用域和 `__init__` 等通用方法名可以参与 Root 排序，但不能生成 `Doc:`，防止把不相关章节伪装成直接文档依据。

`[CONTEXT via calls]` 和 `[CONTEXT via feeds]` 只表示沿真实边补充的结构上下文。紧随其后的关系行明确谁在第几行通过什么关系连接谁；只有带行号的源码才是实现证据。源码行号不连续表示中间无关 Behavior 已被省略；无法可靠裁剪时则返回完整 Symbol。


### Trace

FeedbackTrace 不检索或截断任务。模块六把模块四的完整图无损整理为 `task_scopes`：同一 `task_id` 的 Behavior 放在一个 Task Scope 中，Task Scope 内保留原始事件、Behavior 顺序和它发出的 `relations`。`current_task_id` 指向截止点前最后一个 Behavior 所属的 Task Scope。

#### Complete Task Scope Graph Schema

M6 直接在每个 Behavior 内展示对应的 `demand/action/response` 内容，不再重复完整 User Prompt。M2/M4 使用 `char_range` 确定片段后，M6 只输出原始 `evidence_id` 和截取出的逐字 `content`，不向模型输出 `char_range`。`content` 不能改写或总结。

模型输入改为：

```json
{
  "current_task_id": "K0002",
  "task_scopes": [
    {
      "task_id": "K0001",
      "status": "history",
      "turn_range": [1, 40],
      "behaviors": [
        {
          "behavior_id": "T0001",
          "sequence_index": 1,
          "turn_range": [1, 2],
          "demand": [
            {"evidence_id": null, "content": "修改 src/api.py 中的缓存逻辑"}
          ],
          "action": [
            {"evidence_id": "raw-E2", "tool_name": "Edit", "content": "..."}
          ],
          "response": []
        }
      ],
      "relations": [
        {
          "edge_id": "R0001",
          "type": "informs",
          "target_task_id": "K0002",
          "support": [
            {"evidence_id": "raw-E2", "content": "..."},
            {"evidence_id": null, "content": "基于刚才的修改继续验证"}
          ]
        }
      ]
    },
    {
      "task_id": "K0002",
      "status": "current",
      "turn_range": [41, 65],
      "behaviors": [
        {
          "behavior_id": "T0002",
          "sequence_index": 1,
          "turn_range": [41, 65],
          "demand": [
            {"evidence_id": null, "content": "基于刚才的修改继续验证"}
          ],
          "action": [],
          "response": [
            {"evidence_id": "raw-E4", "content": "验证已经通过"}
          ]
        }
      ],
      "relations": []
    }
  ]
}
```

构建规则：

1. 根据 `task_id` 收集全部 Task Scope，每个 Scope 完整展示所属 Behavior；M4 中的每个 Behavior、Task Edge 及其引用的 Evidence 都必须进入模型输入；
2. Task Scope 按首次出现的 turn 排列，内部 Behavior 按 `sequence_index` 排列；顺序只用于恢复原始 Trace，不表示重要程度；
3. 每个 Behavior 直接按 `demand/action/response` 展示对应内容，并保留原始 `evidence_id` 和已有的 `tool_name`；Demand 只展示该 Behavior 对应的逐字片段，不重复整段 User Prompt；M6 不输出 `char_range`；
4. 一条 Assistant Response 明确回应多个 Demand 时，允许由多个 Behavior 引用同一原始 Evidence；每处只展示与该 Behavior 对应的逐字片段，不强行唯一归属；
5. 每个 Task Scope 的 `relations` 保存它发出的 `informs/supersedes`，并直接展示目标 Task ID 和最小直接 Evidence 内容；
6. `current_task_id` 固定为截止点前最后一个 Behavior 所属的 `task_id`；对应 Scope 标为 `current`，其余标为 `history`，不读取 Gold；
7. Behavior 之间不插入关系边；
8. 不截断 Behavior 或 Task Edge。M4 保留完整 Evidence；M6 只做逐字选段，不生成摘要或改写，最终引用仍使用原始 Evidence ID。

完整展示所有 Task Scope；`current` 只恢复截止点处正在处理的任务，不删除任何历史任务。


## 8. 模块七：AgentLoop 与 Trace Runner

### 作用

模块七控制模型何时检索、何时读取以及何时提交答案，并保证 Raw baseline 与 BEG 除证据后端外使用相同评测条件。

### Repo：AgentLoop 动作

每轮只能输出一个动作：

```json
{"action":"search","text":"keywords"}
{"action":"read","ids":["F0001"]}
{"action":"finish","prediction":{}}
```

三者区别是：

- `search` 是导航，不返回可用于最终判断的完整证据。目录没有明确目标，且当前也没有可直接读取的 `F/R` ID 时才使用；
- `read` 是取证。已经知道目录 ID 或 search 命中 ID 时直接使用；BEG 返回 Local Graph，Raw 返回原始文件；
- `finish` 是提交。所有输出项已经被成功 read 的原文支撑后使用。最后一轮只能 finish。

一次 `read` 可以批量读取 1 至 6 个已知 ID，但这是协议上限，不要求填满。默认优先一次读取少量高优先级 ID；不要为了遍历目录而重复 search。

`search` 使用确定性检索：依次优先匹配完整 ID、路径、Symbol 和 Behavior 名称；没有精确命中时，将输入按空白拆成关键词，进行大小写不敏感的 AND 匹配。结果按文件聚合，同一文件只返回一个可读取 ID。命中数不超过 12 时全部返回，实际通常为 3–8 条；超过 12 条时按匹配度排序，只返回前 12 条。每条结果采用一行目录格式：

```text
Read ID | File | Matched symbols | Match reason
F0003 | src/patchy/cache.py | PatchingCache.store | exact symbol; source: maxsize
```

### Repo：Raw 与 BEG 如何公平比较

| | AgentLoop Raw | AgentLoop BEG |
|---|---|---|
| 初始输入 | 完整任务文档 + 普通生产文件目录 | 同一完整任务文档 + 图排序文件目录 |
| `search` | 搜索 path、Symbol 和原始源码，返回 `R` 文件 ID | 搜索 path、Symbol、Behavior 和 Evidence，返回 `F` 文件 ID 及命中根 |
| `read` | 返回对应原始生产文件 | 返回以该文件为入口的对齐式 compact Local Graph |
| `finish` | benchmark 原生 schema | 相同 benchmark 原生 schema |

两端必须使用同一模型、reasoning 配置、最大轮数、输出上限、样本、任务定义和 Judge。Prompt 只解释各自会看到的证据格式，不能暗示哪一端更可靠。

### Trace：Trace Runner

FeedbackTrace 不使用 Repo AgentLoop。Raw Trace baseline 一次读取完整原始轨迹；BEG 一次读取模块六生成的完整 Task Scope JSON 图。正式样本都存在一个重要的 Verification Point，模型只输出该验证点、支撑 Evidence 和严重程度；程序保存时统一补充 `verdict: KEY`。两端使用同一模型、Prompt 任务定义、输出上限、Prediction Schema 和 Judge。

Trace 图必须保留每个 Behavior 内的全部原事件及原顺序；没有形成 Behavior 的孤立用户事件不进入模型图。Behavior 通过 `task_id` 归入 Task Scope，并按 `sequence_index` 恢复顺序；`informs/supersedes` 只表达有原始 Evidence 支撑的跨任务信息传递或替换。


### 当前实验配置（Repo / Trace）

以下为当前代码与 `.env` 的配置快照。Baseline（`raw`）与 BEG（`graph`）共用模型、思考参数和同一 benchmark 的运行预算。

**模型配置**

当前被测模型：`BEG_MODEL_PROFILE=GPT_LUNA`；当前评分模型：`JUDGE_MODEL_PROFILE=QWEN`。

| Profile | 模型 ID | 发给模型的思考参数 |
|---|---|---|
| `GPT_LUNA` | `gpt-5-6-luna` | `reasoning_effort="none"` |
| `GPT_TERRA` | `gpt-5-6-terra` | `reasoning_effort="none"` |
| `GPT_SOL` | `gpt-5-6-sol` | `reasoning_effort="none"` |
| `DEEPSEEK_FLASH` | `deepseek-v4-flash` | `thinking={"type":"disabled"}` |
| `DEEPSEEK_PRO` | `deepseek-v4-pro` | `thinking={"type":"disabled"}` |
| `GLM` | `glm-5-2` | `thinking={"type":"disabled"}` |
| `QWEN` | `qwen3.7-max-2026-06-08` | `enable_thinking=false` |
| `CLAUDE` | `claude-sonnet-5` | `thinking={"type":"disabled"}` |
| `KIMI` | `kimi-k3` | `reasoning_effort="low"` |

Kimi 的 `THINKING_MODE=required` 是本地校验标记，不传给 API。当前所有 Profile 均未设置 `temperature`、`top_p`，采用服务端默认值。

切换模型可修改 `.env` 的两个 Profile 选择项，也可给 `predict.py` / `judge.py` 传 `--model-profile <PROFILE>`。参数优先级为命令行 > 对应角色的 `BEG_` / `JUDGE_` 配置 > Profile；`--show-config` 可查看实际生效配置，不发起模型请求。

**Repo AgentLoop 与 Trace Runner**

| 配置 | Repo：SpecGap / SilentSwap | Trace：FeedbackTrace |
|---|---|---|
| 调用方式 | 多轮 `search` / `read` / `finish` | 一次性输入完整 Raw Trace / Task Scope JSON |
| 最大轮数，包含 finish | SpecGap 8；SilentSwap 6 | 1 次，无工具调用 |
| 初始目录预算 | 3072 token | 不适用 |
| search 限制 | 最多返回 12 条；查询最长 512 字符 | 不适用 |
| 单次 read 最大 ID 数 | 6 | 不适用 |
| 普通工具结果 / 单个完整原子单元上限 | 32768 / 65536 token | 不适用 |
| 每次请求输出上限 | 32768 token | 32768 token |
| 本地上下文预算 / 安全余量 | 1000000 / 32768 token | 1000000 / 32768 token |
| 输入硬上限（扣除输出与安全余量） | 934464 token | 934464 token |
| 工作上下文压缩阈值 / 目标 | 131072 / 98304 token | 不压缩、不截断；超限报错 |
| 网络重试 | 每次请求最多额外重试 2 次 | 每次请求最多额外重试 2 次 |
| 输出格式修正 | 每次完整运行：SpecGap 最多 2 次；SilentSwap 最多 1 次 | 无 |
| 引用校验 | 引用范围须由已读取代码覆盖；允许跨过原始源码中确认的空白行，不允许跨过未展示的代码或注释；baseline 和 BEG 共用规则 | 引用须来自输入中的证据 ID |

上下文使用 `utf8_bytes_div3_v1` 估算；初始目录使用 `o200k_base` 计数。表中上下文数值是程序配置的预算，并非各模型服务端窗口的声明。

Repo 达到压缩阈值或服务端拒绝上下文长度时触发压缩：清理已消费或重复的旧 search 结果、已有完整副本的重复读取内容，保留完整任务文档和唯一证据。必要时可高于压缩目标，但不能超过输入硬上限。

**全量运行与 Judge**

| 配置 | 当前值 |
|---|---|
| 默认范围 | `full`，三个 benchmark，`raw` + `graph`；当前各 100 个样本，共 600 份预测 |
| 预测 / Judge 并发 | 2 / 3 |
| 请求超时 | 600 秒 |
| SilentSwap 自动重测 | 首轮全部结束后，对所有预测失败样本从头重测 1 次（包括格式、引用校验和网络失败）；两组规则一致 |
| SpecGap 自动重测 | 首轮全部结束后，对预测失败的样本从头补测 1 次；补测仍失败就结束，断点续跑不增加次数；两组规则一致，每次运行仍最多修正格式 2 次 |
| Repo token 汇总 | 仅统计成功那次完整运行的用量；失败运行的原始记录和实际用量单独保留 |
| Repo 重测报告 | 首次预测失败数、首次格式失败数、自动重测数、重测成功数、重测后仍失败数 |
| Judge | 上表 Qwen 配置；同一 benchmark 的两组共用评分规则 |
| Judge 输出上限 | 32768 token |
| Judge 重试 | 网络额外重试最多 2 次；JSON 格式修正最多 1 次；新批次失败样本在队尾补评 1 次，已有实验沿用 manifest 中的策略 |
| 结果目录 | `experiments/<benchmark>/<模型目录>/<baseline或BEG>/`；`--experiment-name` 传相对路径，如 `specgap/gpt5.6/BEG` |
| 预测结果目录 | 实验目录下 `runs/<input-id>/`；保留样本内的 attempt、请求、响应和状态 |
| Judge 结果目录 | 实验目录下 `judges/<judge-model>/<input-id>/`；保留评分尝试和补评记录 |
| 汇总 | 实验根目录唯一的 `summary.json`：`prediction` 保存预测汇总，`judges[模型名]` 保存对应评分汇总 |
| 续跑 | 同名实验按已完成结果续跑，并校验配置；更换配置使用新实验名 |

`benchmark`、`arm` 和阶段保留在 manifest 中，不再重复放入单组实验的目录路径。历史上同一实验同时运行 raw 和 graph 时，仅额外保留 arm 层以避免同名样本冲突。根目录汇总的 schema_version 为 2，两个阶段更新各自部分；后续预测汇总不会覆盖已保存的 Judge 汇总。

旧目录迁移需在该实验停止写入后执行：`python -m scripts.main.migrate_layout --experiment-name specgap/deepseek_flash/BEG`。加 `--check` 只检查路径和冲突；不指定实验名时处理 `experiments/` 下的全部实验。迁移保留原始答案、响应、分数和断点，合并旧汇总，并更新 Judge 续跑所用的预测路径。

配置入口：`scripts/model_config.py`、`agentloop/config.py`、`tracereview/config.py`；批量默认值见 `scripts/main/predict.py` 与 `scripts/main/judge.py`。



### 为什么指标会高

BEG 的目标是 benchmark 指标高于 Raw baseline，同时消耗更少的总 token。它针对 baseline 的三类主要负担进行设计。

**提高语义指标。** Raw baseline 只提供普通文件目录或原始长轨迹，模型需要自己寻找相关文件、任务边界和执行结果。BEG 先把 Evidence 组合成完整 Behavior：Repo 使用 `path + symbol` 组织代码行为并保留 `calls/feeds`，Trace 使用 `task_id` 组织需求履行链并保留直接的 `informs/supersedes`。模块六再把同一张图整理为模型可连续阅读的对齐式 compact 文本，明确区分直接 Root 与结构邻居，减少模型自行拼接或误判跨函数事实的负担；Trace 仍使用完整 Task Scope JSON。

**提高定位指标。** Raw baseline 给模型的是一段段普通文本，模型即使理解了问题，也容易只指出整个文件、整个 Symbol，或只定位结果行，无法准确说明哪些原文共同支撑判断。BEG 从模块一开始就为每条 Evidence 保存真实文件、Symbol、行号或 turn；Behavior 将结果锚点与其触发条件绑定，关系边也保存连接发生的位置。局部图把这些位置和原文放在对应节点旁边，使最终答案能够落到真正支撑条件和结果的精确 Evidence，而不是宽泛的相关区域。