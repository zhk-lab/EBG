# Experiment index

| Experiment | Paper location | Entry point |
| --- | --- | --- |
| [Main](main/README.md) | Table 1; Appendix E.1 | `scripts/main/` |
| Judge agreement | Table E3 | `docs/judge_agreement.md` |
| [Ablation](ablation/README.md) | Table E4 | `ablation/scripts/pipeline.py` |
| [Input size](input_size/README.md) | Figure 5; Table E5; Figure E1 | `input_size/scripts/count_workload.py` |
| [Sensitivity](sensitivity/README.md) | Figures E2–E5 | `sensitivity/scripts/run.py` |
| [Failure analysis](failure_analysis/README.md) | Appendix G | `failure_analysis/scripts/attribute.py` |
| [Cost](cost/README.md) | Token usage analysis | `cost/scripts/summarize.py` |
| [Harness cases](../harness/case_studies/README.md) | Section 5.2.3; Appendix F | `harness/case_studies/run.py` |

Reusable execution code lives in `scripts/` and `src/`. Experiment-specific
configs and analysis programs live here. Generated data, reports, and figures
go to `outputs/analysis/<experiment>/`; publish selected aggregates in `results/`.

Analysis programs that consume saved model runs require those local artifacts.
Their offline unit tests use synthetic inputs. They do not call model APIs.
