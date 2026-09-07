---
name: beg-disclose
description: Use source evidence to check and disclose issues affecting completion or improvement claims, consequential approach changes, unresolved requirements, or a user-requested full review.
---

## Autoresearch review

Check whether the actual work satisfies the applicable requirements and whether the evidence supports the proposed conclusion. For each consequential claim, connect the required condition to the observed execution, then decide what conclusion that evidence permits. Matching numbers or listing configuration values is evidence collection, not the decision. Apply the relevant checks below; do not announce a checklist on every iteration.

### 1. Was the required work actually completed?

- Trace the entry point, effective configuration, executed branch and resulting output. Distinguish code that could run from evidence that the required operation did run. Check what happened after errors or missing prerequisites, not just the outer command's exit status.
- Compare the operation and its result with the success conditions. Request construction, actual transmission and a valid response establish different facts. If a substitute ran instead of the required operation, report what actually succeeded and which requirement remains unmet; do not promote the substitute's success to full completion. Creating an output file likewise does not establish the required result.
- Examples: mock or dry-run fallback, a local substitute for an external service, a disabled feature flag, swallowed exceptions, skipped batch items, empty outputs, and cached or old results returned instead of fresh execution. Such behavior is an issue only when it changes the required operation or leaves a claim unsupported; an explicitly permitted fallback or cache can satisfy the task.

### 2. Was verification sufficient for the claim?

- Match the tests or experiments actually executed to the behavior, scope and version being claimed. Identify what a passing result establishes and what it leaves unverified. If the claim depends on an untested behavior, narrow the claim or obtain the missing verification. Lack of coverage does not itself prove the implementation is wrong. Reading test code is not evidence that tests ran.
- Examples: a test filter selecting only a subset, skipped tests, expected failures counted as successes, disabled assertions, unit tests presented as end-to-end verification, mocked integration boundaries, untested error paths, or a test that checks only output existence. Require real integration or broader coverage only when the task or claim depends on it.
- After a material fix, check whether the relevant behavior was verified on the retained implementation. Earlier passing results cannot validate later changes. A recovered environment error need not remain a reported defect. If evidence is insufficient, narrow the claim or perform justified verification within existing authorization; do not expand every task into an exhaustive test campaign.

### 3. Is the claimed improvement comparable?

- State the intended comparison: the method itself, particular configurations, or complete workflows. Identify allowed changes, required controls and consequential unspecified conditions. Resolve important assumptions before choosing a search or adopting results. Permission to vary a condition does not make its effect on the comparison irrelevant; do not invent a prohibition where none exists.
- Compare actual evaluation inputs and measurement methods: sample membership, splits, preprocessing, labels, metric definitions, denominators and aggregation. The same filename does not establish the same evaluated data. Where independent evaluation is required, examine data origins and use: renamed duplicates, fitting on held-out examples, or answer-bearing caches can compromise independence. Validation-guided hyperparameter selection is not automatically training-data leakage.
- Compare relevant configuration and resource conditions: hyperparameters, training epochs or steps, stopping rules, search attempts across all phases, restarts, seeds, compute or API budgets, hardware and cache state. Fixed cost per fit does not establish equal total search opportunities. Check how best, mean or final-run results were selected and compared.
- For each material difference, decide whether it is the intended change, an established comparable condition, or another possible explanation for the gap. If the evidence cannot separate those effects, explicitly identify the difference and explain what cannot be concluded. The measured score can remain valid while the method comparison remains inconclusive. A tuned configuration may still be recommended, with that limitation attached. Unequal budgets and maximum selection are not violations by themselves.

### 4. Is the conclusion supported by the delivered evidence?

- Match each completion, effectiveness or improvement claim to its execution records, configuration and retained code or artifact version. Check that reported numbers belong to the selected run and that a later edit or revert did not invalidate the connection.
- Consider the full relevant record, including regressions and failures. Examples: reporting only favorable trials, comparing a candidate maximum with a baseline mean, ignoring earlier search phases, combining metrics from different versions, or treating a small single-seed difference as a reliable improvement.
- Evaluate the proposed wording, not just the arithmetic: does it claim completion, effectiveness or superiority beyond what the evidence establishes? When several factors changed, distinguish the measured observation from its possible explanations. Replace an unsupported claim with a supported one and explain the specific limitation. A generic statement such as "in these experiments" does not explain a known confound; a caveat about one condition does not address a different changed condition.

These examples are investigation leads, not automatic findings or an exhaustive list. Follow the evidence relevant to the claim. Static code, a missing trace, or an unmatched reference alone does not prove what happened. Use `beg_evidence` for concrete unresolved premises rather than turning suspicions into findings.

Disclose supported issues that affect the conclusion or the user's decision, with their impact, evidence and remedy. A material limitation can require disclosure even when no task rule was violated. If evidence establishes a mismatch, explain it; if an essential premise is unknown, state the uncertainty. Put the specific difference and its consequence alongside the recommendation in the final response, not only in an artifact. Describe repaired issues according to the final state and avoid repeating resolved or unchanged findings.

Assess the claim actually being delivered: use `issue` for a supported unmet condition or material limitation needing disclosure, `uncertain` for an unresolved essential premise, and `clear` only when the proposed conclusion is supported and any material limits are already reflected in its user-facing wording. Do not record `clear` solely because records match, tests passed, changes were permitted, or no universal claim was made.

## Active checks during ordinary work

Use `result` before adopting or reporting results, `adjustment` for consequential changes after a blocked step, and `ambiguity` for important unresolved requirements. Routine implementation choices need no warning. Hooks cannot infer every decision, so state the proposed claim or decision explicitly.

1. Read a notified checkpoint with `beg_context(check_id=...)`, or create one with `beg_context(trigger=..., focus=...)`. After verification, use its new checkpoint; if `execution_scope` identifies a later verification, read `latest_check_id`. Without a notification, create a `result` check explicitly.
2. Apply the four review areas to the returned requirements, implementation, linked materials and execution trace. When necessary, use `beg_evidence(check_id, question, refs)` to investigate a specific factual premise; refs are optional when frozen sources provide the anchors. Use returned `read_ref` values to expand existing evidence.
3. Disclose material findings promptly, or continue if none are supported. Save the assessment with `beg_context(check_id, conclusion, summary)`: `clear`, `issue` or `uncertain`, with evidence and handling. Reading a checkpoint is not an assessment; saving one is not user-facing disclosure.

Reuse checks only while the focus, execution and captured state remain applicable. Use `prior_assessments` to avoid repetition. Never use an old judgment to verify a new result. Review is read-only; return to ordinary work for authorized repairs or verification. Stop is a fallback continuation and cannot retract an already displayed response; check before adoption and the first completion report.

For exact quotation formats, evidence expansion, historical sources or pagination, read the relevant section of [Tool workflows](references/tool-workflows.md).

## Full review explicitly requested by the user

Read the full-review section of [Tool workflows](references/tool-workflows.md) before using `beg_list_task_sources`, `beg_select_task` and `beg_build_evidence_groups`. Ordinary completion reporting uses the two-tool workflow above. A full review discloses findings; it does not itself authorize edits or new experiments.
