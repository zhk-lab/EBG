---
name: beg-disclose
description: 用户要求检查当前会话中任务的完成情况、需求遗漏、计划偏差或验证缺口时，用 BEG 原文证据判断值得披露的问题。
---

1. 调用 `beg_list_task_sources` 查看 Prompt 和 Markdown Plan 候选版本。默认使用当前会话，也可传 Hook 提供的 session_id。需要读取当前 Plan 或没有已登记仓库时，传 `repo_path`；没有 Hook 历史也可检查当前代码。
2. Prompt、Plan 至少有一项有效内容即可。有历史任务时选择连续 Prompt 区间及可选 Plan，排除本次检查请求，end_prompt 须属于已结束的轮次。没有有效历史区间但有 Plan 时，直接选择 Plan，不因缺少历史 Prompt 而停止。
3. 历史检查调用 `beg_select_task(start_prompt, end_prompt, plan_ids)`，无 Plan 时传空列表。仅用 Plan 时调用 `beg_select_task(plan_ids=[...])`，同时省略两个 Prompt 端点；Harness 保存当前 Repo 用于静态实现核对，不关联检查请求的 Trace，也不生成历史差异。读取返回的 sources 原文，保留 task_id；已指定但无效的历史区间不能悄悄降级为当前代码检查。
4. 整理最终有效要求，体现后续修改、取消、条件和例外，保留原文引用；不能根据实现反向改写要求。Plan 来源不明时保持 unknown；有上下文依据时在引用中注明 origin 为 user_plan 或 agent_plan。
5. 调用 `beg_build_evidence_groups(task_id, requirements)`，每条要求格式如下：

```json
{"id":"R1","check":"最终有效要求","refs":[{"source_id":"P98","quote":"原文中的连续片段"}]}
```

合并或修订要求时保留相关来源；引文重复时附字符 start。工具参数直接传入，无需先向用户输出 YAML。

6. 核对证据中的要求、代码、操作结果与 Agent 声明，披露差异、影响和 source 位置。匹配仅表示相关，未匹配不等于未执行；无法确认及尚未检查的部分明确说明。只作披露，不修改代码、运行新测试或执行修复。

最终回复使用可读的来源：代码与 Plan 写为 `文件路径@行号`（如 `test_harness/retry.py@1-8`）；执行记录写实际命令和关键结果，声明引用必要的回复原文。省略快照、事件、任务、Prompt、Plan 和证据组的内部编号及读取引用，这些仅用于内部取证与工具调用。检查范围用任务内容及 Plan 路径说明；需要区分历史版本时写“任务开始前／结束后”，不要将历史行号说成当前文件位置。无须引用的细节直接省略。

三个工具返回过大时会给出 read_ref 和 next。继续调用同一个工具：传 read_ref 与 offset；证据读取还需 task_id，省略 requirements。目录条目含 read_ref 时可继续展开，next 读取后续目录或文本。按需分批核对，保留发现和引用，不把未展开的材料当成已检查。changes 引用可检查未关联到要求的代码改动。

历史检查只使用选定的历史材料，不用当前文件补作历史快照。仅用 Plan 时，依据选定的 Plan 原文和选择时保存的当前 Repo 判断实现遗漏或计划偏差；说明是当前代码核对，缺少历史 Trace 不妨碍静态检查，但不能据此确认过去运行过测试、任务改动范围或历史完成声明。新会话会清理旧会话记录，旧引用失效。
