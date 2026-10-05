# Harness case studies

Five compact cases correspond to Section 5.2.3 and Appendix F. Each case has
an initial workspace under `seeds/` and an English input under `prompts/`.

| Case | Review target |
| --- | --- |
| `01_ambiguity` | Unspecified priority between competing metrics |
| `02_adjustment` | Fallback changes experimental conditions |
| `03_data_leakage` | Data provenance and validation-set leakage |
| `04_api_verification` | Evidence for an actual API request |
| `05_search_budget` | Comparable search opportunities and budgets |

Use Python 3.11+ and a current, authenticated Codex CLI. Check the actual executable
on `PATH` before preparing a case (use `--codex` if several versions are installed):

```bash
npm install -g @openai/codex@latest
codex exec --dangerously-bypass-hook-trust --help
codex login
```

From the repository root, install in a virtual environment. The `case-studies`
extra includes NumPy for case 05; the standalone harness does not require it:

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -e "./harness[case-studies]"
```

Prepare and run one case:

```bash
python harness/case_studies/prepare.py --install --round r1 --case 01_ambiguity
python harness/case_studies/run.py 01_ambiguity --round r1 --model gpt-6-luna --effort low
```

Use `--plain` in both commands for the comparison arm. Workspaces are created
under `outputs/harness/workspaces/<round>/<arm>/<case>/`, and raw runs under
`outputs/harness/runs/`. Existing workspaces are not overwritten; choose a new
round for a fresh run. The runner supports `--codex`, `--model`, `--effort`,
`--workspace-root`, and `--output-root`. Running it invokes Codex and may incur
model costs. Completed runs are reused; failed runs retain a nonzero exit code.
Interrupted runs preserve their PID and raw logs and are not silently restarted.
Inspect them before choosing a new round. The runner prepends the active Python
environment to `PATH`, checks CLI support before creating a run, and records the
completed-turn event and token usage. A completed turn is not a task-quality verdict.
API cases accept an explicit local `--env-file` during preparation. Without live
credentials, case 04 can verify local behavior and disclose that live verification
remains incomplete; a cached response does not count as a live request.

Keep seeds unchanged. Give the evaluated Agent only its case prompt and seed,
without review criteria or provenance notes. The existing historical results
have not been rerun during repository preparation. Local historical trajectories
are retained under `outputs/harness/cases/<case>/with_ebg/` and
`outputs/harness/cases/<case>/without_ebg/`. Selected historical pairs and their evidence limits are described in
[historical results](historical_results.md). The local mapping is fixed in
`outputs/harness/final_selection.json`; the selected API pair is `harness6`
versus `plain5`. Public record links will be added when available.

Offline fixture checks:

```bash
python -m unittest discover -s harness/case_studies -p "test_*.py" -q
```

See the [2026-10-05 Luna Light validation](luna_light_results.md) for current runs, repairs, and remaining limits. These are separate from the paper's historical results.

See [provenance](provenance.md) for the source mechanisms behind these cases.
