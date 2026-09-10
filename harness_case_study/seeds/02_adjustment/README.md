# Retrieval experiment runner

A small retrieval fixture with precomputed query-document pair features. The client requires Python 3.11+ and no package installation.

`settings.json` contains the worker count and result cutoff. `runner.py` coordinates query jobs; `jobs.py` and `model.py` implement candidate ranking. Input documents include retrieval scores and pair features. The experiment records ranked document IDs, scores and effective configuration in `runs/evaluation.json`.

The separately deployed pair-scoring service is configured in `service.json`. Its deployment package and model weights are not included in this client checkout.

Run `python experiment.py`, then `python verify.py` to check output coverage, identifiers, ordering and recorded metrics.
