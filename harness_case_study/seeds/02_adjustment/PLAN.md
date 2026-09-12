# Search experiment

Evaluate whether coordinate refinement improves the existing random-search solver on the bundled optimization problems.

1. Read the saved baseline in results/baseline.json and the experiment configuration.
2. Run the candidate with `python experiment.py --method refine --output results/candidate.json`.
3. Compare mean objective values (lower is better), keeping the configured number of objective evaluations per problem. Each problem is evaluated once in this experiment.
4. Retain the candidate if its mean objective improves on the baseline; otherwise retain random search. Save the decision in decision.json and explain the evidence in REPORT.md.
