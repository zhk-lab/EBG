# Main experiments

This is the entry point for Table 1 and Appendix E.1. The paper evaluates
SpecGAP, SilentSwap, and the FeedbackTrace Long view, with 100 inputs each.

| Method | CLI arm | Benchmarks |
| --- | --- | --- |
| EBG | `graph` | All three |
| Base | `raw` | All three |
| RepoGraph | `repograph` | SpecGAP and SilentSwap |

## Files

- [configs/samples.json](configs/samples.json): the frozen paper input IDs.
- [Data preparation](../../scripts/prepare_data.py): install a downloaded release.
- [Graph preparation](../../scripts/main/prepare.py): build and validate artifacts.
- [Prediction](../../scripts/main/predict.py): run methods with resumable outputs.
- [Judging](../../scripts/main/judge.py): score saved predictions.
- [SilentSwap source review](../../scripts/main/source_review.py): second-stage
  source recovery and combined EBG scores.
- [Summary export](scripts/summarize.py): export saved judge means as CSV.

Run commands from the repository root. See the full
[reproduction guide](../../docs/reproducibility.md) for setup and commands.
`samples.json` is an input allowlist, not a development/formal split file.

Each prediction batch saves its settings, selected IDs, prompts, responses,
completion state, and usage. Judging adds separate results for each judge model.
Rerunning the same command resumes completed samples; changed settings require
a new output directory. Paper results are available in
[benchmark_results.md](../../docs/benchmark_results.md).

Large historical runs live locally under `outputs/main/`. New runs should use
`outputs/experiments/` or an explicit output directory. Neither belongs in Git.
