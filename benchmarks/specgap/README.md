# SpecGAP construction

SpecGAP removes repository-supported requirements from DeNovoSWE task documents.
The pipeline selects conditions, verifies exact deletions, and independently
maps each condition to implementation and test locations. Human review compares
the two mappings and resolves annotation disagreements.

Install `requirements.txt` in a dedicated environment. From the repository root:

Download `denovoswe_public.jsonl` from
[DeNovoSWE](https://huggingface.co/datasets/AweAI-Team/DeNovoSWE/tree/main)
to `data/source/denovoswe.jsonl`, preserving the upstream JSONL fields. Copy
[.env.example](.env.example) to `benchmarks/specgap/.env`
and set the API key. Python dependencies and Git are required.

```bash
python -m pip install -r benchmarks/specgap/requirements.txt
python benchmarks/specgap/scripts/generate_specgap_v2.py --source local --input data/source/denovoswe.jsonl --output-dir data/construction/specgap --work-dir outputs/construction/specgap --limit 100 --target-successes 100 --resume
python benchmarks/specgap/scripts/validate_specgap_v2.py --data-dir data/construction/specgap
```

`--limit` sets the candidate budget; increase it if too few candidates qualify.
The generator also supports `--source hf`. Configure the selected provider
through environment variables or a local `.env`; see `--help` for model and
endpoint options. Credentials and downloaded repositories stay outside Git.
For a small first run, use `--limit 10 --target-successes 1`; acceptance is not
guaranteed for every candidate. Use `--provider`, `--model`, and `--base-url`
to select another supported provider without editing code.

Construction prompts are defined in `scripts/generate_specgap_v2.py` beside
their input assembly. `remap_dual_mappings_v2_2.py` supports mapping updates.
Each sample includes the before/after document, removed conditions, both code
mappings, the repository snapshot, and validation patches. These are construction
artifacts; the paper uses the frozen EBG release described in
[reproducibility.md](../../docs/reproducibility.md).

Before release, check that deletions preserve the task and that both mappings
support each removed condition. Resolve disagreements in the sample annotations,
then rerun the validator. Keep the complete `data/construction/specgap/` directory.

Run offline checks with `python -m pytest benchmarks/specgap/tests -q`.
