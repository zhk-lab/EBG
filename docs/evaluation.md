# Evaluation protocol

Main experiments use 100 SpecGAP, 100 SilentSwap, and 100 KEY FeedbackTrace Long
inputs. Base and EBG run on all three components; RepoGraph runs on repository tasks.

| Component | Rule-based metric | Semantic metrics |
| --- | --- | --- |
| SpecGAP | Localization F1 | Semantic F1, Question Quality |
| SilentSwap | Localization Score | Localization Correctness, Change Correctness |
| FeedbackTrace | Evidence Hit Rate | Verification Point Alignment, Evidence Support Score |

Code retains historical field names such as `f1`, `code_change_correct`, and
`evidence_location_score`. GLM-5.2 and Qwen3.7-Max judge semantic metrics
independently; the main paper table uses their arithmetic mean. Rule-based
metrics are identical across judge records.

SpecGAP and SilentSwap allow eight and six rounds, including `finish`.
FeedbackTrace uses one request without tool calls. See `src/agentloop/config.py`,
benchmark configuration modules, and `src/tracereview/config.py` for defaults.

SilentSwap EBG localization comes from stage 1; source review provides the
stage-2 code-understanding score. Later code exposure is excluded from stage-1
failure attribution. Preserve stage identities when aggregating results.

Run manifests and public settings record model IDs, reasoning settings, sample
selection, and budgets. Credentials belong in `.env`. Frozen settings identify
the historical experiment independently of changing service defaults.
