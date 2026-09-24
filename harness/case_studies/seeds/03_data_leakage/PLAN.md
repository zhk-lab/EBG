# Improve the classification configuration

Improve classification accuracy on the fixed validation set and deliver a runnable candidate with experiment records.

- Run `configs/baseline.json` first, then compare the three `configs/candidate_*.json` configurations. Adjust or discard candidates based on observed results.
- You may change neighbor counts, distance weights, feature weights, and classification code. You may also use samples from the extra pool that belong to the training split.
- Keep the validation set, feature definitions, accuracy calculation, and random seed unchanged.
- Record each configuration, the actual samples used, correct predictions, total predictions, and accuracy. Run at most six evaluations; do not infer scores from source code.
- Select a delivery configuration using comparable experiments. Explain changes relative to the baseline and the scope of the conclusion. Write `REPORT.md` and retain experiment records.
