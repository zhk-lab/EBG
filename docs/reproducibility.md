# Reproducing the experiments

Use an editable checkout and run these commands from the repository root.
Install the package as described in the [README](../README.md).

## Dataset release

The Google Drive download location is recorded in
[dataset.yaml](../benchmarks/dataset.yaml). The dataset is publicly available on
[Google Drive](https://drive.google.com/drive/folders/1_xWCQDQUYEFxZ2yxF8uenkTpysUl4Des?usp=drive_link).
The authoritative release is the
frozen EBG evaluation set listed in
[samples.json](../experiments/main/configs/samples.json).
Construction projects provide builders; their current generated samples are
not interchangeable with the frozen evaluation set.

Open the Google Drive folder above and download `specgap.tar.gz`,
`silentswap.tar.gz`, `feedbacktrace.tar.gz`, and `samples.json` into
`data/agentmonbench/` (no login required). If Drive bundles the download as a
ZIP, unpack that ZIP first. Then extract the three archives:

```bash
python -m tarfile -e data/agentmonbench/specgap.tar.gz data/agentmonbench
python -m tarfile -e data/agentmonbench/silentswap.tar.gz data/agentmonbench
python -m tarfile -e data/agentmonbench/feedbacktrace.tar.gz data/agentmonbench
```

After extraction, the data has this structure:

```text
agentmonbench/
  specgap/artifacts/
    visible_bundles/sg_001/input_manifest.json
    visible_bundles/sg_001/documents/...
    visible_bundles/sg_001/repository/...
    hidden_gold/sg_001.json
  silentswap/artifacts/...
  feedbacktrace/artifacts/
    visible_bundles/ft_001_long/input_manifest.json
    visible_bundles/ft_001_long/trace/model_input.json
    hidden_gold/ft_001_long.json
```

`visible_bundles` contains only monitor-visible inputs. `hidden_gold` contains
evaluation targets and is read only by the judges. SpecGAP Gold includes the
frozen `formal_reference` annotations used by its scorer. Maintainers can package
older Gold with `scripts/package_specgap_gold.py`; released data already includes
these fields. Construction material and
generated graphs are not required in this download. FeedbackTrace uses the
100 KEY Long inputs.

```bash
python -m scripts.prepare_data --source /path/to/agentmonbench --resume
python -m scripts.main.prepare build --workers 4
python -m scripts.main.prepare validate
```

The default destination is `data/prepared/`. Use `--destination` for a different
data location, then pass it as `--evaluation-root` to graph preparation and
`--artifact-root` to prediction and judging. Preparation validates each visible
manifest and keeps Gold separate. It resumes sample by sample.

## Prediction and judging

Copy `.env.example` to `.env`, then set your model profiles and credentials.
`EBG_MODEL_PROFILE` selects the monitor and `JUDGE_MODEL_PROFILE` selects the
judge. Record actual provider model IDs and request options with each run.
The following commands make paid API calls unless `--prepare-only` is used
for prediction.

```bash
python -m scripts.main.predict --experiment-name main-model-a --phase full
python -m scripts.main.judge --experiment-name main-model-a --phase full
```

By default, prediction runs all supported benchmark/arm combinations. Restrict
a run with repeated `--benchmark` and `--arm` flags. The CLI uses `graph`, `raw`,
and `repograph` for EBG, Base, and RepoGraph. FeedbackTrace supports EBG and
Base. `--phase full` selects all prepared inputs; prepare a clean dataset
directory with the frozen allowlist above.

Use a distinct experiment name per monitor configuration. To add a second
judge, change the judge profile and rerun the judging command; its results are
stored separately. Paper main scores average the two judges as described in
[evaluation.md](evaluation.md). Preserve rule-based metrics without double
counting them. Do not average rounded display values for final paper tables.

## SilentSwap EBG source review

SilentSwap EBG uses a saved first-stage decision followed by source recovery.
Run this benchmark/arm in its own batch so source review has a single run per ID:

```bash
python -m scripts.main.predict --experiment-name silentswap-ebg-model-a --benchmark silentswap --arm graph
python -m scripts.main.judge --experiment-name silentswap-ebg-model-a --benchmark silentswap --arm graph
python -m scripts.main.source_review --source outputs/experiments/silentswap-ebg-model-a --output outputs/experiments/silentswap-ebg-model-a-source --then-judge --judge-model YOUR_JUDGE_MODEL_ID
```

Pass the exact first-stage judge model ID to `--judge-model`. Repeat source
judging for the second judge with `--judge`. The combined result retains
first-stage localization and uses source review for code-change correctness.
Use the combined summary for the paper EBG row.

## Resume and analysis

Prediction and judging save progress after individual calls and samples. Repeat
the same command to resume. Settings and input identities must match saved
manifests; use a fresh directory after changing them. Historical runs may retain
their original absolute paths and should be treated as archived records.

The [experiment index](../experiments/README.md) maps analyses to paper outputs.
Generated reports go under `outputs/`, while compact reviewed exports belong
under `results/`. The [release notes](release_notes.md) list pending metadata and
paper/table revision alignment.

## Offline checks

```bash
python -m scripts.demo
python -m unittest discover -s tests -t . -q
python -m unittest discover -s harness/case_studies -p test_cases.py -q
```

Builder tests and the independent harness have separate dependencies and test
commands in their READMEs. Dataset-dependent integration tests skip when their
external data is absent. These checks do not rerun model experiments.
