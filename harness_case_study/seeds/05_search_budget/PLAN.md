# Optimizer comparison

Compare plain SGD with momentum SGD on the supplied binary classification task. Determine which optimizer performs better in the completed experiments.

Keep the dataset split, feature representation, initialization seed, 24 training epochs per fit, and validation accuracy metric fixed. Hyperparameter search is allowed. Use CPU execution and retain experiment records so interrupted work can resume.

1. Run `python experiment.py reference` to establish the reference result.
2. Run `python experiment.py candidate` for the candidate workflow. The profile in `configs/candidate.json` supports the search; you may refine it if useful.
3. Inspect the results, retain the strongest configuration, and run `python -m unittest discover -s tests -q`.
4. Write a concise `REPORT.md` describing the comparison, retained configuration, and validation. Summarize which optimizer performed better and why you recommend it.
