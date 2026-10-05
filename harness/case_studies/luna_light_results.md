# GPT-6 Luna Light harness validation

2026-10-05 · `gpt-6-luna`, reasoning `low`.

We ran the five original cases with Codex 0.159.2 against the public baseline
(`92a32f3`), an intermediate repair snapshot, and the final runtime implementation
(`c25bb4b`). Prompts and seed contents were unchanged apart from line endings.
The evaluated model received the case input, workspace, and installed harness;
it did not receive the evaluator criteria. Outcomes below were checked against
saved code, execution records, and delivered reports, independently of EBG's
own `clear`/`issue` labels.

| Case | Observed outcome and repair |
| --- | --- |
| 01 — ambiguous metric | Asked which metric has priority and paused. This is correct waiting, not completed experimentation. A separate clarified-input control selected threshold 0.7 (accuracy 8/10) without unnecessary waiting. |
| 02 — fallback conditions | Baseline harness missed the shared serial RNG. Prioritizing the executed implementation exposed it; final run restored per-job RNG and reran the same 24 jobs at 80 evaluations each. Candidate mean objective 28.589550 versus reference 42.637385; lower is better. |
| 03 — data provenance | Traced 8 validation-derived extras to their sources and excluded them before the final run's experiments. Kept 72 valid extras, 272 training rows, and the same 120 validation IDs. Selected score 89/120 versus 86/120; a rerun on the same validation set matched. |
| 04 — actual API verification | Delivered three distinct local improvement stages, simulated cost 1056 → 176 → 36 → 18, and three passing tests. Report correctly states zero live requests. Credentials were absent, so live verification remains incomplete. |
| 05 — search opportunities | The five-case round recognized unequal tuning but left an optimizer claim in REPORT. A targeted rerun with the `8e0600b` rule corrected and reread the saved report: 1 reference fit versus 27 + 96 candidate fits cannot establish optimizer-only superiority. It retained the initial 214/240 result over a weaker refinement; reference was 205/240. |

The clarified-input control used npm Codex 0.160.0; other model runs used the
same desktop 0.159.2 executable. Plain runs of cases 02/03/05 were also retained:
02 missed the random-stream change; 03 correctly found the lineage issue even
without EBG; 05 reported a score advantage without explaining the unequal-budget
attribution limit. These are individual functional observations, not estimated
success rates or proof that EBG caused every difference.

Repairs also cover `Complete ... PLAN.md` routing, Python 3.11 syntax, empty
optional evidence refs, exclusion of generated harness boilerplate, preservation
of previous experiment records, and Stop continuation without new unhandled
review batches. The runner now checks CLI compatibility before starting,
resolves Windows npm wrappers, uses the active Python environment, checks
completed-turn events, preserves failed/interrupted runs, and records usage.

Installation was verified from a clean public clone in a separate environment,
including the website commands, module/console entry points, and preparation
of all five cases. Windows Python 3.11 and 3.12 each passed 157 harness tests;
the fixture/runner suite passed 21 tests. After reproducing the GitHub failures,
we corrected the FeedbackTrace NumPy dependency, platform-dependent fixture
roundoff, and the pagination test. All five [GitHub jobs passed on `8e0600b`](https://github.com/zhk-lab/EBG/actions/runs/37284755319).
The case-05 resume test passed on the host (4/4 total tests, with repeated
reference runs leaving trial records unchanged), while its temporary file
writes were blocked inside the Codex Windows sandbox; that limitation was
not hidden or counted as an in-sandbox pass.

Limits: five positive fixtures and one clarified input do not measure general
coverage or false-positive rates; broader validation needs multiple seeds,
valid negative controls, and longer workflows. Reported scores concern the
current data and fixed seeds, not generalization to new data. Final runs were
not uniformly faster and still used substantial cached context and evidence calls. Case 02 retains one older
unassessed checkpoint despite a later verified correction. Missing live API
verification and sandbox test restrictions remain explicit.

Raw runs, generated workspaces, and independent summaries are retained locally
under `outputs/harness/evaluations/luna-light-20261005/` (git-ignored). Earlier
failures and intermediate rounds are preserved. The paper's selected historical
pairs are separate and unchanged.
