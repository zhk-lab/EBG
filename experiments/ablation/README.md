# Ablation

Table E4 removes behavior organization, relation structure, or task structure
while retaining the applicable input and evaluation contract.
`configs/` records models, request options, and sample IDs; `prompts/` contains
variant-specific task prompts. Set model endpoints in a local config copy and
provide the referenced API-key environment variables before model runs.

```bash
python experiments/ablation/scripts/run.py --help
python experiments/ablation/scripts/run.py prepare
python experiments/ablation/scripts/run.py predict
python experiments/ablation/scripts/run.py judge
python experiments/ablation/scripts/analyze.py
```

Use `--config`, `--model`, `--benchmark`, and `--variant` to select the intended
run. Prediction and judging resume saved samples. `pipeline.py` preserves the
historical Luna execution schedule; `run.py` is the general entry point.

Outputs are under `outputs/analysis/ablation/`. Comparisons also require the
saved main runs referenced in the configuration. The analysis retains paired
bootstrap intervals and does not reinterpret incomplete runs as successful.

Offline tests: `python -m unittest discover -s experiments/ablation/scripts -p "test_*.py" -q`.
