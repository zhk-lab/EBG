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

Install the harness first. From the repository root:

```bash
python harness/case_studies/prepare.py --install --round r1 --case 01_ambiguity
python harness/case_studies/run.py 01_ambiguity --round r1
```

Use `--plain` in both commands for the comparison arm. Workspaces are created
under `outputs/harness/workspaces/<round>/<arm>/<case>/`, and raw runs under
`outputs/harness/runs/`. Existing workspaces are not overwritten; choose a new
round for a fresh run. The runner supports `--codex`, `--model`, `--effort`,
`--workspace-root`, and `--output-root`. Running it invokes Codex and may incur
model costs. API cases accept an explicit local `--env-file` during preparation.

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
python -m unittest discover -s harness/case_studies -p test_cases.py -q
```

See [provenance](provenance.md) for the source mechanisms behind these cases.
