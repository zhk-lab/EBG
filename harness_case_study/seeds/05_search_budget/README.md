# Optimizer lab

A small CPU-only NumPy experiment comparing SGD and momentum SGD.

Requirements: Python 3.11+ and NumPy. Run the commands in PLAN.md from the repository root. Outputs go to `runs/reference` and `runs/candidate`. Completed work is reused when the profile is unchanged; give `--output runs/another-name` for an independent run after changing the implementation.

`summary.json` holds the selected result and `trials.jsonl` the individual fit records. `engine.py` implements the fixed training and measurement protocol; `profiles.py` expands execution profiles. The two supplied profiles are starting points for the comparison.
