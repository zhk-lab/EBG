# SilentSwap construction

SilentSwap introduces five semantic substitutions per DeNovoSWE repository.
Construction verifies changed behavior while retaining the original tests,
then records locations, intended behavior differences, and human review.

Use Linux/WSL, a dedicated Python environment, Git, curl, and Docker. The builder
downloads `denovoswe_public.jsonl` from
[DeNovoSWE](https://huggingface.co/datasets/AweAI-Team/DeNovoSWE/tree/main).
Copy [.env.example](.env.example) to `benchmarks/silentswap/.env`, fill in the key,
and export it in Bash before running from the repository root:

```bash
python -m pip install -r benchmarks/silentswap/requirements.txt
set -a
. benchmarks/silentswap/.env
set +a
python benchmarks/silentswap/scripts/build_formal_samples.py --resume
python benchmarks/silentswap/scripts/build_batch_samples.py --count 100 --resume
python benchmarks/silentswap/scripts/expand_multi_swaps.py
```

These commands can call a model and run containers. Inspect `--help` before
construction. The formal builder creates seed samples; the batch builder fills
the collection; expansion adds and validates multiple substitutions. Default
outputs are the ignored `benchmarks/silentswap/data/` and `.work/` directories.
The expansion and review tools consume that layout.
For an initial check, run only the formal builder (three seed samples). The
full expansion/review pipeline expects 100 samples. Model and endpoint defaults
are documented in `.env.example`; they are currently fixed in the builder.

Prompts and acceptance rules are inline in the build scripts. The frozen
`review/review_plan.json` assigns swap categories and double-review cases.
Human reviewers record their decisions in the local `review/reviews.jsonl`;
`apply_gold_reviews.py` applies reviewed annotations and `validate_dataset.py`
checks the resulting collection. Review decisions are not invented by this
repository migration.

Use [reviews.example.jsonl](review/reviews.example.jsonl) as the record format.
Fill every required sample/swap/role from `review_plan.json`. For `accept`, copy
the semantic fields and evidence exactly from that swap's `gold.json`; use
`revise` for corrected annotations. Resolve double-review disagreements through
an `adjudicator` record, then run:

```bash
python benchmarks/silentswap/scripts/apply_gold_reviews.py
python benchmarks/silentswap/scripts/validate_dataset.py --require-adjudicated
```

Keep the completed `benchmarks/silentswap/data/` and the review decisions.

`code_locations.py` contains shared patch and symbol-location helpers. Paper
prediction and judges are maintained separately in the repository root.

Run `python -m pytest benchmarks/silentswap/tests -q`. Full dataset checks skip
when constructed data is absent; Docker-dependent tests require Docker.
