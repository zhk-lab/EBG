# Failure analysis

Appendix G groups failures into file-selection omissions, evidence-presentation
omissions, and model-judgment failures. These are operational diagnostics based
on saved requests and judgments, not independently established causal labels.

```bash
python experiments/failure_analysis/scripts/summarize.py
python experiments/failure_analysis/scripts/attribute.py
python experiments/failure_analysis/scripts/report_attribution.py
```

The analysis requires the saved main-run predictions, actual requests, judges,
and prepared Gold. It writes to `outputs/analysis/failure_analysis/` and caches
per-run input reviews. No model requests are made. `configs/` contains coverage
rules and supplementary adjudications. Reports preserve their distinctions.

SpecGAP includes 482 conditions with localization references out of 484 total
conditions. SilentSwap uses first-stage localization failures and first-stage
inputs. FeedbackTrace checks whether Gold events were presented before
attributing a decision-selection failure. The hop-analysis scripts additionally
require the corresponding saved sensitivity runs.

Offline tests: `python -m unittest discover -s experiments/failure_analysis/scripts -p "test_*.py" -q`.
