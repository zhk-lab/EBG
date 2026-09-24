# AgentMonBench

AgentMonBench evaluates whether a monitor identifies consequential decisions
and grounds its findings in evidence from agent work. It accompanies
**What Did the Agent Actually Do? Evidence-Grounded Oversight for Long-Horizon Agents**.

This release contains the frozen evaluation inputs and annotations used by
EBG (Evidence-Grounded Behavior Graph), a training-free oversight method.
Code: [EBG anonymous repository](https://anonymous.4open.science/r/EBG-B67B/).
The paper link will be added when available.

## Components

| Component | Examples | Task | Source |
| --- | ---: | --- | --- |
| SpecGAP | 100 | Identify omitted requirements using repository evidence | [DeNovoSWE](https://huggingface.co/datasets/AweAI-Team/DeNovoSWE) |
| SilentSwap | 100 | Identify semantic substitutions that preserve existing tests | [DeNovoSWE](https://huggingface.co/datasets/AweAI-Team/DeNovoSWE) |
| FeedbackTrace | 100 | Identify consequential decisions before subsequent user feedback | [SWE-chat](https://huggingface.co/datasets/SALT-NLP/SWE-chat) |

FeedbackTrace uses the complete pre-feedback Long view. The three components
are evaluation sets, not training/validation splits. `samples.json` lists
the exact released IDs.

## Files

The complete evaluation release is available on [Google Drive](https://drive.google.com/drive/folders/1_xWCQDQUYEFxZ2yxF8uenkTpysUl4Des?usp=drive_link).
The folder contains the three archives below, `samples.json`, and a README.

### Download complete inputs

Each component is distributed as a `.tar.gz` archive. Extract all three into
the same directory:

```text
agentmonbench/
  specgap/artifacts/
    visible_bundles/<input_id>/input_manifest.json
    visible_bundles/<input_id>/...
    hidden_gold/<input_id>.json
  silentswap/artifacts/...
  feedbacktrace/artifacts/...
```

`visible_bundles` contains the monitor inputs. Each manifest enumerates its
visible files. `hidden_gold` contains evaluation targets and must only be read
by the scorer, never provided to the monitor. SpecGAP Gold includes the frozen
`formal_reference` annotations. Graphs are rebuilt from visible inputs.

## Download and reproduce

Download the three `.tar.gz` files and `samples.json` from the Google Drive
folder into `data/agentmonbench/`. If downloaded together as a ZIP, unpack it
first. Run these commands from the EBG repository root:

```bash
python -m tarfile -e data/agentmonbench/specgap.tar.gz data/agentmonbench
python -m tarfile -e data/agentmonbench/silentswap.tar.gz data/agentmonbench
python -m tarfile -e data/agentmonbench/feedbacktrace.tar.gz data/agentmonbench
```

From the EBG code checkout, with Python 3.11 or newer:

```bash
python -m pip install -e ".[test,analysis]"
python -m scripts.prepare_data --source data/agentmonbench --destination data/reproduction --resume
python -m scripts.main.prepare build --evaluation-root data/reproduction --workers 4
python -m scripts.main.prepare validate --evaluation-root data/reproduction
```

Configure monitor and judge credentials in `.env` using `.env.example`.
Run each component in a separate batch, replacing `specgap` below with
`silentswap` or `feedbacktrace` for the other components:

```bash
python -m scripts.main.predict --experiment-name reproduce-specgap --benchmark specgap --arm graph --phase full --artifact-root data/reproduction
python -m scripts.main.judge --experiment-name reproduce-specgap --phase full --artifact-root data/reproduction
```

SilentSwap EBG additionally requires its second-stage source review:

```bash
python -m scripts.main.source_review --source outputs/experiments/reproduce-silentswap --output outputs/experiments/reproduce-silentswap-source --artifact-root data/reproduction/silentswap/artifacts --then-judge --judge-model YOUR_JUDGE_MODEL_ID
```

Use the exact first-stage judge model ID. Paper scores use two judges;
follow `docs/reproducibility.md` in the code repository for second-judge
evaluation and aggregation. Prediction and judging incur model API usage.

## Construction and limitations

SpecGAP omits selected requirements and maps them to repository evidence.
SilentSwap introduces semantic substitutions and checks that existing tests
still pass. FeedbackTrace separates pre-feedback trajectories from later
feedback used for annotation. Construction includes model assistance and
human review; builder code lives under `benchmarks/` in the EBG repository.

Regeneration can produce different examples and annotations. Use this frozen
release to reproduce the paper evaluation. Match monitor versions, request
settings, and judges when comparing scores; API nondeterminism can still
change individual predictions.

## Attribution and terms

Upstream datasets and repository snapshots retain their respective licenses
and access conditions. DeNovoSWE lists CC BY 4.0; SWE-chat lists ODC-By.
Consult the linked source cards and individual repository licenses for their
terms and attribution requirements. The EBG code's MIT license does not relicense upstream
content. Dataset-specific release terms are pending finalization for this draft.

## Citation

Paper bibliographic information and the final citation will be added when available.
