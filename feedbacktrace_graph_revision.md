# FeedbackTrace 图结构审查与修改建议

## 1. 结论

当前 FeedbackTrace 的主要问题不是遗漏 Evidence，而是不同任务的 Behavior 被平铺成一条长链，任务边界和明确的跨任务信息传递没有被表达。

建议采用：

```text
完整 Trace Graph（后台保留）
        ↓
用 task_id 标记 Task Scope
        ↓
Task Scope --informs/supersedes--> Task Scope
Task Scope 包含按需求履行链划分的 Behavior
Behavior 按 sequence_index 保留原始顺序
        ↓
模型判断 Verification Point
```

这里的 `task_id` 与 Repo 的 `path + symbol` 作用相同：它是 Behavior 的作用域、分组和检索字段。Task Scope 不生成摘要或重要性判断；`informs/supersedes` 只连接有明确 Evidence 证明信息传递或替换关系的两个任务。

## 2. 结论一：图覆盖了关键事实，但全局长链弱化了末端交互

146/146 条 Gold Evidence 都完整进入了旧图，问题不在事实遗漏，而在组织方式。旧图从 Trace 起点开始，将全部 Behavior 用 `precedes` 串成一条全局长链；关键交互位于长链末端，容易被前面大量历史事件淹没。

- 100/100 个 Gold 都位于最后一个 Behavior；Gold 在 turn 序列中的中位位置达到 98.4%；
- ft_001 和 ft_013 的 BEG 预测都误选了更早任务中的决定，说明完整输入不等于关键事实得到突出；
- 旧 `continues` 只连接相同对象，只有 24/100 个最终 Behavior 被接回相关前文。

新结构按 `task_id` 将 Trace 划分为多个 Task Scope：每个任务分别组织自己的 Behavior，Behavior 用 `demand / action / response` 呈现“要求—行动—回应”，任务之间只用有直接 Evidence 支撑的 `informs/supersedes` 连接。这样既保留完整 Trace，又避免把所有任务平铺成一条难以阅读的长链。

## 3. 各模块修改

### M1：Evidence Intake

**修改：无。**

对应结论一：Gold Evidence 已完整进入图，因此不调整提取策略。

### M2：Behavior Atomization

**修改：必须。**

解决结论一、二：提高 Behavior 的语义粒度，并明确区分用户要求、实际动作和 Agent 回应。

Trace 不再沿用 Repo 的 `trigger / operation / result`，也不再把整轮交互直接视为一个 Behavior。一轮交互只提供原始时间边界：从一个 `user_prompt` 开始，到下一个 `user_prompt` 之前结束；其中可以包含一个或多个 Behavior。

```text
一项 demand + 对应 action + 对应 response = 一个 Behavior

同一轮交互
  ├─ Behavior 1：Demand 1 → Actions 1 → Response 1
  └─ Behavior 2：Demand 2 → Actions 2 → Response 2
```

三部分含义是：

- `demand`：用户提出的一项要求、问题、约束、批准或选择；
- `action`：Agent 为处理该 demand 执行的 Tool、检查、修改和验证；
- `response`：Assistant 针对该 demand 给出的决定、解释、承诺、完成声明或结果汇报。

Behavior 使用以下结构：

```json
{
  "behavior_id": "T0021",
  "task_id": "K0003",
  "sequence_index": 2,
  "start_turn": 20,
  "end_turn": 25,
  "demand_refs": [
    {
      "evidence_id": "E0201",
      "char_range": [35, 92]
    }
  ],
  "action_evidence_ids": ["E0202", "E0203"],
  "response_refs": [
    {
      "evidence_id": "E0204",
      "char_range": [0, 68]
    }
  ]
}
```

`demand_refs` 和 `response_refs` 引用原始 Evidence；`char_range` 记录该 Behavior 对应的逐字片段，省略时表示引用完整事件。完整 Evidence 仍只在 M4 保存一次；多个 Behavior 可以引用同一条 Demand 或 Response，但不能改写原文。

#### 单轮 Behavior 划分信号

代码通过固定词表和可逐字匹配的对象进行识别，不让模型猜测：

1. **Action**：读取 `tool_name`、Tool 参数和结果，得到“操作类型 + 稳定对象”。固定映射如 `Read/Grep → inspect`、`Edit/Write → modify`、`test/pytest → verify`、`commit/push → publish`；文件路径、Symbol、测试名、分支和 job ID 作为稳定对象。同一对象上的读取、处理、验证和失败重试归入一个 Action 组。
2. **Demand**：按最外层编号、同级项目符号或独立问句切分 User Prompt；嵌套条目只是上级要求的细节。再提取“请求类型 + 稳定对象”，如“修改 `src/api.py` → `modify + src/api.py`”。
3. **Response**：从 Assistant 内容中提取“结果类型 + 稳定对象”。完整对象未重复时，也可使用可唯一匹配的命令、测试名等明确组成词。能与某个 Demand 对应的回复归入该 Behavior；一段回复明确覆盖多个 Demand 时分别记录对应的逐字范围，不强行唯一归属。`done`、过渡语和整体总结不单独产生 Behavior。

例如，Tool `Edit(file_path="src/api.py")`、Demand“修改 `src/api.py`”和 Response“已修改 `src/api.py`”都会得到同一个键 `modify + src/api.py`，因此组成一个 Behavior。匹配时以对象完全相同为主，固定动词类别相同为辅，文本先后顺序只在唯一候选时补充使用。

只有识别出两个以上 Demand，且每个 Demand 都有自己可唯一匹配的 Action 或 Response，才拆成多个 Behavior。匹配到同一 Demand 的 Action 和 Response 合并；其余准备、验证或总结内容归入时间上最近的已确认 Behavior。只有一个 Demand 或无法唯一匹配时保留为一个复合 Behavior。

每个 Behavior 固定包含一个 Demand、一个 Action 组和一个 Response 组，但组内可以引用多条 Evidence。原始 Trace 可能只有问答而没有 Tool，或在 Tool 执行后、回复前截止，因此三个字段必须存在，Demand 必须非空，Action 与 Response 至少一个非空；缺失部分保留空数组，不能编造 Evidence。

对应关系是：

```text
Repo                              Trace
path + symbol                     task_id
  ├─ Behavior 1                     ├─ Behavior 1
  └─ Behavior 2                     └─ Behavior 2
```

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

### M3：Relation Linking

**修改：必须。**

解决结论一：删除不能表达任务边界的弱 Behavior 边，只保留有 Evidence 直接支撑的跨任务关系。Repo 的 `calls/feeds` 不修改；Trace 的 Behavior 不建立关系边。

#### 边的定义

- `Task A --informs--> Task B`：B 明确使用 A 已产生的结果、决定、约束或产物。
- `Task B --supersedes--> Task A`：B 明确停止、撤销或替换 A 的方案、运行或决定，使 A 不再是当前选择。

#### 可观察信号

1. **语句信号**：`use/reuse/based on/apply` 表示使用；`stop/cancel/revert/replace/instead of` 表示替换。
2. **稳定标识**：两边出现同一 job/run ID、Issue/PR、分支、产物路径，或同一段可逐字匹配的决定或约束。
3. **交互结构**：A 的 action/response 先产生或持有该对象，B 的 demand/action 随后使用、取消或替换它。

#### 连边判断

1. 由 B 的语句信号发现候选关系；
2. 用稳定标识把对象唯一定位到旧任务 A；
3. 验证 A 确实产生该对象，B 确实使用或替换它；
4. 三类信号同时成立才连边：使用时建立 `informs`，取消或替换时建立 `supersedes`。

Behavior 不建立关系边。相同任务由共享的 `task_id` 表达，任务内顺序由 `sequence_index` 表达，避免用边重复保存作用域和时间位置。

### M4：Graph Assembly

**修改：必须。**

落实结论一：在完整后台图中保存 Task Scope、细粒度 Behavior、原始顺序和稀疏的 `informs/supersedes`。

完整 Trace Graph 继续作为唯一后台事实源。Task Scope 由 `task_id` 确定，不生成任务摘要节点；Behavior 保存 `task_id` 和 `sequence_index`，`informs/supersedes` 使用 Task ID 作为端点：

```json
{
  "evidence": [],
  "behaviors": [
    {
      "behavior_id": "T0003",
      "task_id": "K0001",
      "sequence_index": 1,
      "start_turn": 20,
      "end_turn": 30,
      "demand_refs": [
        {"evidence_id": "E0201", "char_range": [35, 92]}
      ],
      "action_evidence_ids": ["E0202", "E0203"],
      "response_refs": [
        {"evidence_id": "E0204", "char_range": [0, 68]}
      ]
    }
  ],
  "edges": [
    {
      "edge_id": "R0001",
      "type": "informs",
      "source_task_id": "K0001",
      "target_task_id": "K0002",
      "evidence_ids": ["E0204", "E0205"]
    }
  ]
}
```

新增校验：

1. 每个 Trace Behavior 必须有且只有一个 `task_id` 和至少一个 `demand_ref`；
2. `demand_ref` 必须指向 user Evidence，`response_ref` 必须指向 Assistant Evidence；已有 `char_range` 必须是对应原文的有效连续范围；
3. `action_evidence_ids` 只能指向 Tool Evidence，且它与 `response_refs` 不能同时为空；
4. 相同 `task_id` 下的 Behavior 按原始时间生成唯一、连续的 `sequence_index`，并允许任务暂停后继续；
5. `task_id` 和 Behavior 划分只能使用可见输入确定性重建，不生成摘要、重要性或问题结论；
6. `informs/supersedes` 两端的 Task ID 必须存在，并由原始 Evidence 直接证明信息传递或明确替换；
7. Graph Assembly 不复制 Behavior 或 Evidence，也不生成 Behavior 间关系边。

### M5：Ranked Directory

#### Repo

**修改：无。**

#### Trace

**修改：无。**

FeedbackTrace 不通过目录筛选任务，避免引入位置或相关性偏置。

### M6：Task Scope Graph Serialization

**修改：重点。**

解决结论一、二：完整保留事实，同时用 Task Scope、`action/response` 和稀疏任务关系提高输入的语义可读性。

M4 继续保存由 Evidence、Behavior 和 Task Edge 组成的完整 Trace Graph。M6 不重新构图，只把同一张图整理成模型可直接阅读的 Task Scope JSON：按 `task_id` 将 Behavior 聚合到 `task_scopes`，再把每条 `informs/supersedes` 放入来源 Task Scope 的 `relations`。`current_task_id` 指向包含截止点前最后一个 Behavior 的 Task Scope。模型读取全部 Task Scope，Behavior 不再被拼成一条跨任务长链。

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

### M7：Trace Runner 与 Prompt

**Prompt 必须修改；配置无。**

解决结论一、二：向模型解释新的图结构、当前任务和 Evidence 引用方式。

Prompt 的判断范围改为：

> 审查完整 Task Scope Graph，在所有任务中找出最需要用户确认的具体决定，并使用原始 Evidence 支撑。

Prompt 还应说明：

- `task_id` 与 Repo 的 `path + symbol` 一样，用于表示作用域；
- `current_task_id` 是截止点前最后一个 Behavior 所属的任务，模型先读取该 Scope，再结合 `history` Scope 恢复相关前提；
- 一个任务可能包含一轮或多轮 Behavior；
- `informs` 表示目标任务明确使用了来源任务产生的结果、决定、约束或产物；
- `supersedes` 表示来源任务明确停止、撤销或替换了目标任务的方案、运行或决定；
- Behavior 通过 `task_id` 归入 Task Scope，并按 `sequence_index` 恢复原始顺序；
- 每个 Behavior 的 `demand/action/response` 直接展示原始 Evidence ID 和对应的原始内容或逐字片段；模型不读取 `char_range`；
- `demand` 是用户的要求、问题、约束、批准或选择；
- `action` 是 Agent 实际执行的 Tool、检查、修改和验证；
- `response` 是 Assistant 给出的决定、解释、承诺、完成声明或结果汇报；
- 最终引用使用原始 Evidence ID。

模型继续只输出：

```json
{
  "verification_point": "...",
  "supporting_evidence_ids": ["..."],
  "criticality": "must_disclose"
}
```

程序保存时统一补充 `verdict: KEY`。

模型、reasoning、Token 上限、输出上限和 Judge 暂时不修改。应先验证结构改动本身，再决定是否调整配置。


FeedbackTrace 当前的核心价值已经从“构建大量语义边”转向“用 Task Scope、Behavior 顺序和 demand/action/response 结构确定性地显式组织隐含关系。

原因：

语义边的不足：
1. 真正的跨任务语义边因为关系类型复杂，难以用少数边类型完整表示。
2. 清晰信号不足：确定性程序只能依赖关键词和稳定标识，规则严格则几乎无边，规则宽松则容易误连。

隐式结构关系的好处：
1. Task 包含 Behavior：代替contains边。
2. 顺序排列：Behavior 的时间关系边。
3. Behavior 包含 Demand / Action / Response：因果边
