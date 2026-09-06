# BEG Codex Harness

独立运行的 BEG 应用：Hook 记录会话，Codex 选择任务并整理要求，BEG 返回原文证据组，Codex 判断值得披露的问题。披露过程不修改目标代码、不运行测试。

| 工具 | 作用 |
|---|---|
| `beg_list_task_sources` | 返回历史 Prompt、Plan；传 `repo_path` 可读取当前 Markdown Plan，无 Hook 历史也可使用 |
| `beg_select_task` | 选择 Prompt 区间及可选 Plan，关联历史 Trace 和快照；只传 `plan_ids` 时保存当前 Repo，按 Plan 核对实现 |
| `beg_build_evidence_groups` | 接收任务编号、需求及原文引用，返回证据组；超预算时按 `read_ref` / `next` 继续调用本工具 |

在本目录安装并生成接入配置（Python 3.11+）：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m codex_harness --state-dir .state/runtime setup --output .state/integration
```

将生成的 `config.toml` 合入目标项目 `.codex/config.toml`，`hooks.json` 合入 `.codex/hooks.json`，保留已有配置；将生成的 `skills/beg-disclose/` 放入 `.agents/skills/`。在 Codex 中加载配置并信任 Hook 后开始任务；之后提出披露检查，或显式使用 `$beg-disclose`。移动目录后重新运行 setup。macOS/Linux 使用 `.venv/bin/python`。

`src/codex_harness/` 下：`tools/` 是接口，`hooks/` 是采集，`skills/` 是调用指导，`application/` 是处理流程，`beg/` 是独立 BEG 实现。无需原项目其他目录。

Prompt、Plan 至少一项非空。仅用 Plan 时可检查当前实现，不证明历史改动范围或测试是否执行。同一状态目录只供一个会话使用；切换会话清理旧记录与缓存，恢复当前会话保留数据。快照复用未变文件，仅保存采集规则允许的文本；Trace 仅覆盖收到的 Hook 事件。Plan 来源不自动推断。默认单次返回预算为 12,000 个估算 token，可通过 `--token-budget` 调整；分页不代表全部材料已检查。

测试：`.\.venv\Scripts\python.exe -m unittest discover -s tests -t . -q`

未命中的要求会附带 `navigation`：仅按引用所在 Plan 章节与文件名的词面对应提供候选，不视作实现证据。`repository` 可展开全部已采集文件，候选及文件目录可继续展开 Symbol，按引用读取单个作用域的冻结原文；没有 Symbol 的文件仍可全文读取。大证据的分页目录保留可容纳的来源位置和 `content_ref`，减少逐层展开。所有读取继续使用同一个工具及原 task_id。

所选任务中未直接匹配要求的调用、结果及 Assistant 回复保留在 `trace_context`，避免漏掉只读测试的动作或“全部测试通过”等整体声明；它们是待核对的任务上下文，不自动归属某条要求。大返回可沿该字段的 `read_ref` 续读。
