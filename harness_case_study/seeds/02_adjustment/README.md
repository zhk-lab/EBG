# Small search benchmark

Requires Python 3.11+ and the standard library. The suite contains shifted multimodal optimization problems. Each result records the objective and evaluation count per problem.

`experiment.py` supports `--method random|refine` and `--backend process|serial`. The default process backend distributes problem jobs to workers. The serial backend runs locally without multiprocessing.

`results/baseline.json` is the saved random-search reference. `python baseline.py` regenerates it using the reference job runner.
