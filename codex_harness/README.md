# BEG Codex Harness

独立运行的 BEG 应用：Hook 记录过程并提供检查信号，Codex 先读相关上下文，必要时用 BEG 查代码证据，主动披露影响结论或决策的问题。核对不修改目标代码、不运行测试；处理问题后可恢复原任务。

| 工具 | 作用 |
|---|---|
| `beg_context` | 创建或读取执行中检查点，返回相关 Trace、Prompt 和 Plan；也用于保存 Codex 的核对结论 |
| `beg_evidence` | 根据原文引用和具体疑问，获取检查点对应的代码证据及历史版本读取入口 |
| `beg_list_task_sources` | 返回历史 Prompt、Plan；传 `repo_path` 可读取当前 Markdown Plan，无 Hook 历史也可使用 |
| `beg_select_task` | 选择 Prompt 区间及可选 Plan，关联历史 Trace 和快照；只传 `plan_ids` 时保存当前 Repo，按 Plan 核对实现 |
| `beg_build_evidence_groups` | 接收任务编号、需求及原文引用，返回证据组；超预算时按 `read_ref` / `next` 继续调用本工具 |

在本目录安装并生成接入配置（Python 3.11+）：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m codex_harness --state-dir .state/runtime setup --output .state/integration
```

将生成的 `config.toml` 合入目标项目 `.codex/config.toml`，`hooks.json` 合入 `.codex/hooks.json`，保留已有配置；将生成的 `skills/beg-disclose/` 放入 `.agents/skills/`。在 Codex 中加载配置并信任 Hook 后开始任务；主动检查使用前两个接口，用户要求完整核对时保留原三个接口。移动目录后重新运行 setup。macOS/Linux 使用 `.venv/bin/python`。

主动检查的三个触发类型为 `result`（采用／汇报结果）、`adjustment`（受阻或重要方案调整）、`ambiguity`（关键要求不明确）。Hook 识别结构化工具失败（非零 exit_code／exitCode、MCP isError）和标准 shell 退出状态，返回检查点提示，不替换原工具结果，也不直接判定违规。方案调整、重要歧义和采用意图由 Codex 按 Skill 显式提交。

直接运行 Python 的 smoke/test/verify/validate/check 脚本或 `-m pytest/unittest` 返回后，Hook 会创建 `result` 检查点，在汇报前提示核对；读取或 echo 这些命令不触发。该识别范围有限，其他入口仍由 Skill 显式触发。上下文沿对应工具输出中的文件引用补取冻结结果材料；`execution_scope` 指出较新的验证检查点，旧材料不能替新执行背书。提示只要求核对，不将命令返回或记录中的模式自动判成成功／失败。

```text
beg_context(trigger="result", focus="准备采用本轮提升结果")
→ 读取 Prompt、相关 Trace 和 Plan，保留 check_id
→ 必要时 beg_evidence(check_id=..., question=..., refs=[{source_id: ..., quote: ...}])
→ 判断并按需要披露
→ beg_context(check_id=..., conclusion="clear/issue/uncertain", summary="依据、影响与处理")
```

Hook 已给出 check_id 时直接读取，不重新创建。每个检查点固定触发时的事件截止位置、Repo 文本与未采集文件信息，并保留此前快照。指定 event_ids 时返回完整调用／结果配对；歧义检查补充相关轮次。无关联时采用当前轮次，完整 Trace 和 Markdown 候选可通过 all_trace／all_sources 展开。保留会话中较早的用户原文，避免遗漏约束；其中可能混有其他任务，由 Codex 判断适用范围，Harness 不自动推断任务边界或 Plan 身份。

读取不等于完成检查。conclusion／summary 保存的是 Agent 判断，不证明已向用户披露。相同焦点、事件和采集状态可复用；变化则建立新检查点，prior_assessments 用于避免重复提醒。二进制内容仍未采集，不能声称复用已验证了二进制内容一致。中断后可用同一状态目录及 check_id 续读。

Stop 在没有对应核对结论时请求一次续跑，并将续跑提示与用户原始要求区分；已处于 Stop 续跑时不再次拦截，避免无限循环。这是兜底机制，不保证收回已经显示的回复；采用／汇报前仍应显式调用 beg_context。Hook 协议依据 [OpenAI Hooks 文档](https://learn.chatgpt.com/docs/hooks)，需在实际 Codex 环境启用并信任；单元测试和 MCP 集成测试不能替代模型主动披露效果评测。

`src/codex_harness/` 下：`tools/` 是接口，`hooks/` 是采集，`skills/` 是调用指导，`application/` 是处理流程，`beg/` 是独立 BEG 实现。无需原项目其他目录。

Prompt、Plan 至少一项非空。仅用 Plan 时可检查当前实现，不证明历史改动范围或测试是否执行。同一状态目录只供一个会话使用；切换会话清理旧记录与缓存，恢复当前会话保留数据。快照复用未变文件，仅保存采集规则允许的文本；Trace 仅覆盖收到的 Hook 事件。Plan 来源不自动推断。默认单次返回预算为 12,000 个估算 token，可通过 `--token-budget` 调整；分页不代表全部材料已检查。

测试：`.\.venv\Scripts\python.exe -m unittest discover -s tests -t . -q`

文件目录区分“内容已采集”和“存在但未采集内容”。二进制、超限和非 UTF-8 文件保留路径、大小与原因，不提供内容读取引用；未采集不等于不存在，存在也不证明可运行。被忽略目录、Git 忽略项和采集错误仍可能造成目录不完整。历史快照保留当时的存在信息，不用当前文件补齐；旧快照未保存的信息无法恢复。

缺少 Hook Trace 不自动构成任务问题。仅 Plan 检查可按材料身份核对随附初始代码、实验记录和汇报；只在缺口影响具体要求或结论时说明。更新后重新启动 MCP 会话，并同步生成的 Skill 到目标项目 `.agents/skills/beg-disclose/`。

未命中的要求会附带 `navigation`：仅按引用所在 Plan 章节与文件名的词面对应提供候选，不视作实现证据。`repository` 可展开全部已采集文件，候选及文件目录可继续展开 Symbol，按引用读取单个作用域的冻结原文；没有 Symbol 的文件仍可全文读取。大证据的分页目录保留可容纳的来源位置和 `content_ref`，减少逐层展开。所有读取继续使用同一个工具及原 task_id。

要求原文中的显式标识符和大写缩写可关联 Python 值的使用位置，并带出一跳调用关系及嵌套函数的最近外层作用域；这是静态上下文，不证明实际执行。同组包含关系的源码片段合并，跨组重复片段保留 `content_ref` 指向冻结材料。普通描述和歧义函数名不会因此变成确定的符号匹配。

所选任务中未直接匹配要求的调用、结果及 Assistant 回复保留在 `trace_context`，避免漏掉只读测试的动作或“全部测试通过”等整体声明；它们是待核对的任务上下文，不自动归属某条要求。大返回可沿该字段的 `read_ref` 续读。

历史任务另有 `history_context`：按回合查看调用和前后快照，并读取中间版本的文件。用于核对某个实验的代码来源，不仅比较任务起点和终点。入口限制在所选任务内，读取冻结内容；回合边界快照需结合该轮调用与改动判断，不能直接当作执行版本证明。

Autoresearch 的显式 JSON/JSONL 实验记录可返回 `research_context`：以 `experiment_id` 关联 `record_type: claim` 与 `record_type: experiment`，比较 `revision`、`metric`、声明的 `conditions` 与实验 `parameters`；可选 `baseline_experiment_id` 用于比较基线条件。`training_row_ids` / `evaluation_row_ids` 仅做字符串交集统计，需核对 ID 命名空间及实际数据用途。重复编号保留歧义。此功能目前需要这些显式字段，不自动理解任意日志格式，也不把记录当作执行证明或自动判定违规。比较附带冻结原文引用；两组评测须获得相同记录。

短引文同一原文行里的显式文件名仍作为取证锚点返回，不扩充任务要求。引文匹配允许 CRLF/LF 的展示差异，保存的引用仍指向原始文本与偏移；其他文字差异不接受。

关联到唯一基线与候选记录时，另比较实际 `evaluation_row_ids` 的样本及重复次数；不能用相同数据文件名替代实际评估样本比较。这个比较只覆盖记录里的 ID，不证明样本内容或标签未变。

Skill 按真实完成、验证充分性、比较条件和结论依据四个方面核对；要求从实际证据判断结论能否成立，例子不自动构成违规。重要限制应与推荐一起出现在最终回复。工具细节和完整审查流程放在 `skills/beg-disclose/references/tool-workflows.md`，setup 会一并安装。Case3 整理了 [普通组记录](../case_study/case3/plain_trace.jsonl) 与 [BEG 记录](../case_study/case3/beg_trace.jsonl)：普通组遗漏预算差异的影响，BEG 在报告和完成回复中披露，但未提前处理搜索歧义。它们是开发过程中选出的单次运行，不代表稳定成功率；完整历史保存在本地 `.tmp/case3-history-20260907/study`。
