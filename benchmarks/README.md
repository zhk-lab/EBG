# AgentMonBench

Each component contributes 100 examples to the paper.

| Component | Source | Construction |
| --- | --- | --- |
| [SpecGAP](specgap/README.md) | DeNovoSWE | Omit requirements and independently map them to repository evidence |
| [SilentSwap](silentswap/README.md) | DeNovoSWE | Introduce five semantic substitutions while preserving existing tests |
| [FeedbackTrace](feedbacktrace/README.md) | SWE-chat | Annotate decisions visible before subsequent user feedback |

The benchmark is publicly available on
[Hugging Face](https://huggingface.co/datasets/ZhaoHongKang/AgentMonBench).
Its pinned revision is recorded in [dataset.yaml](dataset.yaml). See the
[dataset card](dataset_card.md) for contents and download instructions.
Only the 100 KEY FeedbackTrace examples are in scope. Downloaded data belongs
under the ignored root `data/` directory or an explicitly supplied local path.

Each builder has independent dependencies. Construction can call models and,
for SilentSwap, run Docker workloads. Human review remains a separate required
step. Released annotations define the paper evaluation; model-based regeneration
does not promise identical annotations.

The builders reproduce the construction method and task types, not the exact
released samples. Each component README lists source data, configuration,
generation, and review steps. Completed native construction directories are
the outputs; conversion to the frozen EBG evaluation format is separate.
End-to-end generation with live model calls has not been rerun for this release.

Construction prompts live beside their builders. Prediction and judge prompts
live in the root `prompts/` directory. Current paper judges live in `evaluation/`.
Legacy standalone model-evaluation drivers are not part of benchmark construction.
