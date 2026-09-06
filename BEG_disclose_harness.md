**整体流程**

```text
Hook 持续记录会话和代码状态
        ↓
用户要求披露检查
        ↓
Harness 按顺序列出用户 Prompt 全文和已有 Plan 文件
        ↓
Codex 选择属于本次任务的 Prompt 和相关 Plan
        ↓
Harness 自动关联对应 Trace、任务前后 Repo 快照
        ↓
Codex 根据所选原文整理最终有效要求
        ↓
BEG 将每条要求与相关 Repo / Trace 原文组织在一起
        ↓
Codex 核对证据，披露值得关注的问题
```

1. Codex 调用接口 `beg_list_task_sources`。Harness 用程序按时间顺序列出用户      Prompt 全文和已有 Plan 文件及版本，通过工具返回结果输出到 Codex 上下文中。

2. Codex 根据 Prompt 时间线给出本次任务的连续起止范围，并选择相关 Plan。例如，100   条 Prompt 中，本次任务对应 P98～P100：

```yaml
start_prompt: P98
end_prompt: P100
plan_ids: [相关 Plan 编号]
```

3. Codex 调用 Harness 接口 `beg_select_task`，将上面 YAML 中的起止编号和 Plan 编号作为参数传入。Harness 据此取出 P98～P100 的要求原文、从 P98 开始到 P100 所在轮次结束的 Trace，以及 P98 开始时和 P100 所在轮次结束时的 Repo 快照，同时读取选中的 Plan。如果 P100 本身是披露检查请求，应将它排除在原任务之外，此时任务范围可以是 P98～P99。

4. Harness 固定这些材料，生成任务编号，将任务编号、所选 Prompt 全文和 Plan Markdown 原文返回到 Codex 上下文中。对应 Trace 和前后 Repo 快照保留在 Harness 内部，供后续 BEG 处理。

5. Codex 阅读返回的 Prompt 和 Plan，整理最终有效的要求，保留每条要求的原文引用，并体现后续修改或取消，区分用户要求与 Agent 自拟计划。

6. Codex 调用接口 `beg_build_evidence_groups`，传入任务编号、整理后的要求及原文引用。Harness 用 BEG 将这些要求与已关联的 Trace、Repo 材料进行证据处理，通过 path、symbol，以及 demand/action/response 等关联，生成按需求组织的原文证据组，再返回到 Codex 上下文中。

7. Codex 核对证据组中的要求与实际执行，判断哪些问题值得披露，向用户说明差异、影响和证据位置。BEG 负责组织证据，Codex 负责作出判断；未匹配到记录不等于行为未发生。披露过程不修改代码、不运行新测试或执行修复。

**会话数据保留**

Harness 只保留当前一个会话的数据。同一会话内，保留已采集的 Prompt、Plan、Trace 和各轮 Repo 快照，供检查该会话中的任务；继续或恢复当前会话时保留原有记录。

切换到新会话时，清理上一会话的记录、快照和证据缓存，再记录新会话。清理以切换会话为单位，不按任务结束清理，也不设置历史保留期限；清理后无法再检查上一会话的任务。清理仅涉及 Harness 保存的数据，不删除目标 Repo 文件。

**Skill 的作用与流程指导**

Skill 是给 Codex 的操作说明：告诉它什么时候进行披露检查、按什么顺序调用工具，以及如何使用返回材料。工具定义说明参数格式，Codex 根据 Skill 和具体任务填写参数，Harness 执行调用；Hook 负责记录过程。

Skill 中的流程指导可以写成：

```text
当用户要求检查任务中有哪些问题值得披露时：
1. 调用 beg_list_task_sources，查看用户 Prompt 全文和已有 Plan 列表。
2. 确定任务的连续 Prompt 起止范围及相关 Plan，排除本次披露检查请求。
3. 调用 beg_select_task，传入起止编号和 Plan 编号。
4. 阅读返回的 Prompt / Plan 原文，整理最终有效要求并保留原文引用，体现修改和取消，区分用户要求与 Agent 自拟计划。
5. 调用 beg_build_evidence_groups，传入任务编号、整理后的要求及引用。
6. 核对返回的原文证据组，披露差异、影响和证据位置；证据不足时明确说明，不把未匹配当成未执行。不修改代码、不运行新测试或执行修复。
```

文档中的 YAML 只是参数示例。Codex 直接按工具定义传参，不需要先向用户输出 YAML 再调用工具。

**证据组格式与返回策略**

返回采用有层级的 YAML，包含披露说明、需求清单和按需求组织的原文证据组。以下为结构示意；实际内容保留来源位置和原文，省略不需要的字段。

```yaml
beg_disclose_prompt: 核对要求与实际执行，判断值得披露的问题；匹配仅表示相关。
requirements:
  R1: 整理后的有效要求
evidence_groups:
  R1:
    requirement:
      demand:
        - source: Prompt 编号及原文位置
          content: 用户要求原文
      plan:
        - source: Plan 版本及原文位置，注明用户提供或 Agent 自拟
          content: 相关计划原文
    actual:
      repo:
        - source: 快照、文件路径、符号及行号
          content: 相关代码或改动原文
          match: path / symbol 关联依据
      trace:
        action:
          - source: 事件编号及类型
            content: 工具调用或结果原文
        response:
          - source: 事件编号
            content: Agent 声明原文
        match: demand / action / response 关联依据
```

单次返回预算内，直接返回全部证据组；超出时，返回证据组目录、部分完整证据组及剩余材料的读取方式。Codex 按返回的引用继续调用 `beg_build_evidence_groups` 读取所需材料，边读边核对，最后汇总披露；尚未检查的部分应明确说明。

未匹配要求可附带 `navigation`，按引用所在 Plan 的章节名称与文件名的词面对应提供候选，明确标为导航而非实现证据；普通叙述中的单词不能仅因与函数同名就成为直接匹配。`repository` 提供全部已采集文件的读取入口，避免没有匹配就无法继续取证。

文件入口先返回 Symbol 目录，可按引用读取单个作用域的连续原文，也可读取完整文件；均使用已固定的 Repo。大证据的分页目录在预算允许时保留来源和 `content_ref`，可直接续读原文。导航不生成问题结论，不修改原文要求，也不保证自然语言需求都能找到实现。

任务内未直接匹配要求的调用、结果和 Agent 回复保留在 `trace_context`，供核对只读测试的动作及没有文件名的整体完成声明；不自动把同轮事件归入某条要求，超预算时提供续读引用。
