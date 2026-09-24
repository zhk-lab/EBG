# Sensitivity

Figures E2–E5 vary relation-expansion hops and interaction limits. The config
files preserve individual run settings and sample IDs. Configure endpoints in
a local config copy and provide the named API-key environment variables.

```bash
python experiments/sensitivity/scripts/run.py plan
python experiments/sensitivity/scripts/run.py prepare
python experiments/sensitivity/scripts/run.py predict
python experiments/sensitivity/scripts/run.py judge
python experiments/sensitivity/scripts/summarize.py
```

Select conditions with `--config`, `--experiment`, `--model`, and `--benchmark`;
see `--help` for details. Saved predictions and judgments support resume.
Default outputs are under `outputs/analysis/sensitivity/`. Main-run comparisons
require the archived runs referenced by the configuration.

`run_read_budget.py` and `retry_missing_stage2.py` preserve supplementary run
procedures. `report_qwen.py` renders the historical Qwen supplement, including
explicit author-assigned zeros; it is specific to those saved runs.

Offline tests: `python -m unittest discover -s experiments/sensitivity/scripts -p "test_*.py" -q`.
