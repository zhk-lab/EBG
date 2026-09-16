# BEG Codex Harness

面向 autoresearch 的独立 BEG 应用：Hook 记录实验过程并触发阶段审查，每项审查必须调用 beg_evidence，核对数据隔离、实验可比性和结论归因。审查工具不修改目标代码、不运行实验。

阅读入口：[流程与案例](BEG_disclose_harness.md)、[四个 Skill 中文译文](四个Skill中文译文.md)、[案例运行说明](../harness_case_study/README.md)。运行用 Skill 位于 `src/codex_harness/skills/`，包含一个总入口和三个阶段。

| 工具 | 作用 |
|---|---|
| `beg_review` | 创建或读取检查点，返回相关要求、执行记录、拟作出的声明或决定和检查条目；不默认查询代码 |
| `beg_evidence` | 根据疑问查询冻结代码及关联材料；统一展开日常审查的所有材料和分页 |
| `beg_record` | 保存判断、依据及处理情况，仅返回简短确认，不代表已经向用户披露 |

在本目录安装并生成接入配置（Python 3.11+）：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m codex_harness --state-dir .state/runtime setup --output .state/integration
```

将生成的 `config.toml`、`hooks.json` 合入目标项目 `.codex/` 并保留已有配置；将生成的四个 Skill（beg-review、beg-ambiguity、beg-adjustment、beg-result-review）放入 `.agents/skills/`。升级时移除旧 beg-disclose Skill，保留四个新目录的相对位置。在 Codex 中加载配置并信任 Hook；移动目录后重新运行 setup。macOS/Linux 使用 `.venv/bin/python`。

三个阶段：Plan 执行前的 `ambiguity`、结束时的 `adjustment` 和 `result`。授权内的调整先记录、汇报时统一披露；需要用户决定的关键取舍在落实前提问。总 Skill 负责流程，分阶段清单明确 autoresearch 的数据、实验及归因检查要求。

两个审查信号：读取待执行 Plan 后的 `PostToolUse` 提醒 ambiguity；`Stop` 分别判断 adjustment/result，仅检查已触发的阶段；同时触发时先 adjustment、后 result。自动识别限于明确执行请求和 Read/read_file 或常见 shell 读取命令；未识别的意图由 Codex 主动调用 beg_review(trigger="ambiguity")。Plan 已在上下文时直接提醒。

工具失败与验证结束只记录，不立即审查。Stop 按 session 累计条件分别判断：result 在执行过 Plan、调用达到 10 次、活动耗时达到 300 秒或记录了结果限制时触发；adjustment 在出现工具失败、记录了关键歧义或重要调整时触发。无问题的 ambiguity 检查不触发 adjustment。可用 `--review-call-threshold`、`--review-seconds-threshold` 配置，setup 同步写入 MCP 与 Hook 参数。排除 Harness 自身调用、重复事件和回合间闲置；恢复进程不计断开时间。已审查且内容未变则复用。

```text
beg_review(trigger="result", focus="准备采用本轮提升结果")
→ 读取 Prompt、相关 Trace 和 Plan，保留 check_id
→ 每项必须 beg_evidence(check_id=..., question=具体审查问题)，核实代码与执行证据
→ 全部材料展开均用 beg_evidence(check_id=..., read_ref=..., offset=...)
→ 判断后 beg_record(check_id=..., conclusion="clear/issue/uncertain", summary="依据、影响与处理")
→ 按适用时机披露重要问题或继续任务
```

Hook 已给出 check_id 时直接读取，不重新创建。每个检查点固定触发时的事件截止位置、Repo 文本与未采集文件信息，并保留此前快照。指定 event_ids 时返回完整调用／结果配对；歧义检查补充相关轮次。无关联时采用当前轮次，完整 Trace 和 Markdown 候选可将 all_trace／all_sources 的引用交给 beg_evidence 展开。保留会话中较早的用户原文，避免遗漏约束；其中可能混有其他任务，由 Codex 判断适用范围，Harness 不自动推断任务边界或 Plan 身份。

仅支持 `ambiguity / adjustment / result` 检查点流程，材料展开统一使用 `beg_evidence`。旧的 Prompt 区间选择、Plan-only 审查入口及 `prompts/disclose.md` 已移除；证据响应使用检查点提示词。更新后重启 MCP 连接，旧模式的任务记录不能继续查询，需通过 `beg_review` 创建检查点。

读取不等于完成检查。beg_record 在没有成功执行 beg_evidence 问题查询时拒绝保存判断；仅翻阅 review 分页不算取证。程序不保证证据充分或结论正确，必须结合清单判断。相同焦点、事件和采集状态可复用；变化则新建检查点。二进制内容未采集，不能声称已验证其一致。

执行中用 `beg_record(note_kind="adjustment"、"ambiguity"或"limitation", decision_status="proposed"或"executed", summary=原因及影响)` 分别记录重要调整、关键歧义或结果限制，不传 check_id/conclusion。结束审查读取这些记录，再通过证据形成判断。

需要澄清时，取证后用 `beg_record(check_id=..., conclusion="uncertain", summary=问题与影响, waiting_for_user=true)`，立即提问并结束回答。Stop 对待澄清状态放行。真实用户答复解决问题后，用原 check_id 和 `resolution=答复如何解决问题` 清除状态；仅收到新消息不会自动清除。

Stop 在没有对应核对结论时请求一次续跑，并将续跑提示与用户原始要求区分；已处于 Stop 续跑时不再次拦截，避免无限循环。这是兜底机制，不保证收回已经显示的回复；采用／汇报前仍应显式调用 beg_review 并用 beg_record 记录判断。Hook 协议依据 [OpenAI Hooks 文档](https://learn.chatgpt.com/docs/hooks)，需在实际 Codex 环境启用并信任；单元测试和 MCP 集成测试不能替代模型主动披露效果评测。

`src/codex_harness/` 下：`tools/` 是接口，`hooks/` 是采集，`skills/` 是调用指导，`application/` 是处理流程，`beg/` 是独立 BEG 实现。无需原项目其他目录。

主动检查需要已记录的 Prompt；执行前启用 Hook。同一状态目录只供一个会话使用；切换会话清理旧记录与缓存，恢复当前会话保留数据。快照复用未变文件，仅保存采集规则允许的文本；Trace 仅覆盖收到的 Hook 事件。Plan 来源不自动推断。默认单次返回预算为 12,000 个估算 token，可通过 `--token-budget` 调整；分页不代表全部材料已检查。

测试：`.\.venv\Scripts\python.exe -m unittest discover -s tests -t . -q`

文件目录区分“内容已采集”和“存在但未采集内容”。二进制、超限和非 UTF-8 文件保留路径、大小与原因，不提供内容读取引用；未采集不等于不存在，存在也不证明可运行。被忽略目录、Git 忽略项和采集错误仍可能造成目录不完整。历史快照保留当时的存在信息，不用当前文件补齐；旧快照未保存的信息无法恢复。

缺少部分 Hook Trace 不自动构成任务问题。按材料身份核对随附初始代码、实验记录和汇报；只在缺口影响具体要求或结论时说明。

未命中的要求会附带 `navigation`：仅按引用所在 Plan 章节与文件名的词面对应提供候选，不视作实现证据。`repository` 可展开全部已采集文件，候选及文件目录可继续展开 Symbol，按引用读取单个作用域的冻结原文；没有 Symbol 的文件仍可全文读取。大证据的分页目录保留可容纳的来源位置和 `content_ref`，减少逐层展开。所有读取继续使用 `beg_evidence` 及原 `check_id`。

要求原文中的显式标识符和大写缩写可关联 Python 值的使用位置，并带出一跳调用关系及嵌套函数的最近外层作用域；这是静态上下文，不证明实际执行。同组包含关系的源码片段合并，跨组重复片段保留 `content_ref` 指向冻结材料。普通描述和歧义函数名不会因此变成确定的符号匹配。

图缓存中的 Evidence 和作用域仅保存文件快照 ID、行范围与匹配索引；Trace Evidence 引用原始事件 ID，行为与边继续引用 Evidence ID。解析和建立关系时仍需读取源码，展示时从冻结快照提取原文。生成的源码输出也按引用保存，分页与重启后继续读取同一版本；旧图缓存会按版本自动重建，已保存的历史输出仍可读取。

Repo 展示按连通分量组织：所有能直接对应引用原文或查询线索的作用域作为 Seed，保留全部一跳关系，不限制边数、不递归扩展。每个分量在 Seed 中选不同邻居最多的节点作为 Root；入边、出边均计入，同一邻居只计一次，不计自环。并列按所引原文中的首次提及位置，再按路径和 Symbol 排序。从 Root 开始广度优先展示，节点块保留 `component`、`node`、`role`、`root`、源码及紧邻的 `relations`（真实方向、类型、依据位置）。共享节点只展开一次；嵌套作用域共用包含它们的源码片段，通过 `content_ref` 和 `included_in` 保留各节点身份。预算只影响分页，不删除已选节点和边。

所选任务中未直接匹配要求的调用、结果及 Assistant 回复保留在 `trace_context`，避免漏掉只读测试的动作或“全部测试通过”等整体声明；它们是待核对的任务上下文，不自动归属某条要求。大返回可沿该字段的 `read_ref` 续读。

历史任务另有 `history_context`：按回合查看调用和前后快照，并读取中间版本的文件。用于核对某个实验的代码来源，不仅比较任务起点和终点。入口限制在所选任务内，读取冻结内容；回合边界快照需结合该轮调用与改动判断，不能直接当作执行版本证明。

Autoresearch 的显式 JSON/JSONL 实验记录可返回 `research_context`：以 `experiment_id` 关联 `record_type: claim` 与 `record_type: experiment`，比较 `revision`、`metric`、声明的 `conditions` 与实验 `parameters`；可选 `baseline_experiment_id` 用于比较基线条件。`training_row_ids` / `evaluation_row_ids` 仅做字符串交集统计，需核对 ID 命名空间及实际数据用途。重复编号保留歧义。此功能目前需要这些显式字段，不自动理解任意日志格式，也不把记录当作执行证明或自动判定违规。比较附带冻结原文引用；两组评测须获得相同记录。

短引文同一原文行里的显式文件名仍作为取证锚点返回，不扩充任务要求。引文匹配允许 CRLF/LF 的展示差异，保存的引用仍指向原始文本与偏移；其他文字差异不接受。

关联到唯一基线与候选记录时，另比较实际 `evaluation_row_ids` 的样本及重复次数；不能用相同数据文件名替代实际评估样本比较。这个比较只覆盖记录里的 ID，不证明样本内容或标签未变。

使用一个总 Skill `beg-review` 和三个阶段 Skill `beg-ambiguity`、`beg-adjustment`、`beg-result-review`。beg_review 按 trigger 仅返回对应清单及共同规则；共同规则和工具细节位于 `skills/beg-review/references/`，setup 一并安装。检查点冻结当时规则，旧检查点不自动替换。

三个历史案例的简要说明见 [BEG_disclose_harness.md](BEG_disclose_harness.md)。案例用于说明审查问题，不代表当前接口的配对重测结果或稳定成功率。
