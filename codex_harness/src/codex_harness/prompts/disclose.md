核对需求清单与原文证据，判断哪些问题值得披露。
Prompt、Plan 至少一项非空即可；仅用 Plan 时核对选择时保存的当前 Repo，不因缺少历史 Prompt 而停止。当前代码检查不证明过去的执行过程或改动范围。
requirement 是要求；repo 是实现及改动；action 是调用和结果；response 是 Agent 声明。
匹配仅表示相关。按原文判断差异与影响，引用 source；未匹配不等于未执行。
最终回复将 source 转为文件路径与行号（如 test_harness/retry.py@1-8），或实际命令、关键结果与回复原文；省略内部编号及读取引用。范围用任务内容和 Plan 路径说明，历史版本按需标注“任务开始前／结束后”；不要将历史行号说成当前文件位置，无须引用的细节省略。
过大材料按 read_ref/next 继续调用 beg_build_evidence_groups，保留 task_id，省略 requirements。
navigation 是所引 Plan 章节与文件名的词面导航，不是实现证据。未命中时可沿候选或 repository 引用读取文件目录，再沿 Symbol 引用读取冻结源码；content_ref 可直达分页原文。目录和候选均不代表已核验。
trace_context 保留任务内未直接匹配要求的调用、结果和 Agent 回复，包括只读测试的动作及无文件名的整体完成声明；结合原始轮次核对，不自动认定其对应某条要求。
Trace 仅包含收到的 Hook 记录；缺失材料与尚未检查的部分应说明。
仅作披露，不修改代码、运行新测试或执行修复。
