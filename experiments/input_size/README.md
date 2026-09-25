# Input-size analysis

Figure 5, Table E5, and Figure E1 relate saved scores to visible-input size.
Repository inputs count the task document and readable code; FeedbackTrace
counts Assistant responses and tool exchanges. The tokenizer is `o200k_base`.
Each benchmark uses Low/Medium/High groups of 33/33/34 inputs.

```bash
python experiments/input_size/scripts/count_workload.py
python experiments/input_size/scripts/compare_models.py
python experiments/input_size/scripts/relative_gain.py
python experiments/input_size/scripts/plot_results.py
```

These commands require the frozen prepared data and saved main-run judgments,
but make no model calls. Counts resume per input. Use `--recount` when inputs
change; omit analysis `--resume` when source scores change. Outputs are under
`outputs/analysis/input_size/`.

The plots and reports distinguish medians, group means, relative gains, and
absolute-difference slopes. Natural associations are not causal difficulty
effects.

Offline tests: `python -m unittest discover -s experiments/input_size/scripts -p "test_*.py" -q`.
