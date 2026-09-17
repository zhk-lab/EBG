# Harness 案例

所有案例使用相同结构：`seeds/<case>/` 保存初始代码和 Plan，`prompts/<case>.md` 保存独立输入。桌面 `EBG_autoresearch_cases/cases/<case>/` 为已接入 harness 的初始项目，包含 `.codex/` 和 `.agents/skills/`；实验报告由运行生成，不预放入案例。

| 案例 | 审查重点 |
|---|---|
| `01_ambiguity` | 多指标的优化优先级未明确 |
| `02_adjustment` | 执行替代方案改变实验条件 |
| `03_data_leakage` | 数据来源和验证集泄漏 |
| `04_api_verification` | 是否实际执行真实 API 请求 |
| `05_search_budget` | 搜索机会与预算是否可比 |

案例机制和预期审查见 [harness 文档](../codex_harness/EBG_disclose_harness.md)。不要把该文档或判分说明提供给被测模型。初始代码保留案例问题，不用试跑修复后的代码覆盖。

在 EBG 根目录使用项目虚拟环境创建独立副本并运行，`<round>` 每次使用新名称，`<case>` 选择上表目录名：

```powershell
& ./codex_harness/.venv/Scripts/python.exe harness_case_study/prepare.py --install --round <round> --case <case>
& ./codex_harness/.venv/Scripts/python.exe harness_case_study/run.py <case> --round <round>
```

无 harness 对照需在上述两个命令中都加 `--plain`。运行模型为 Luna-low，Prompt 自动从对应文件读取；原始日志保存在 `.tmp/harness-five-cases/<round>/<case>/`。已有运行状态不会被自动覆盖。

验证初始案例：

```powershell
& ./codex_harness/.venv/Scripts/python.exe -m unittest harness_case_study.test_cases -v
```
