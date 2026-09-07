---
name: beg-disclose
description: Use source evidence to check and disclose consequential ambiguities or approach changes before acting, and issues affecting conclusions when reporting verified results.
---

## Autoresearch review

Review the applicable requirements, actual evidence, and their impact on decisions or conclusions. The checks below are for internal review. Report consequential findings, not a checklist of every item inspected.

Normally consolidate disclosure when verification has finished and you are about to report the round's results. Results may be successful, failed, partial or uncertain; do not wait for the expected outcome before checking. Internal work may involve repeated edits, tests and reverts. Check before adopting results, but do not report every trial. Routine failures that were repaired and reverified need no repeated warning.

Disclose earlier when continuing requires interpreting a consequential ambiguity or making an approach change that affects the original requirements. Explain the issue before implementing that decision; ask for clarification when a consequential choice belongs to the user.

### A. Before continuing: consequential ambiguities and approach changes

1. **What was required?** Identify the relevant goals, constraints and success conditions in the Prompt and Plan.
2. **What is unclear or blocked?** Distinguish missing information, conflicting requirements, observed execution failures and routine repairable errors. Identify the specific ambiguity or failure evidence; repeated failure does not authorize relaxing requirements.
3. **What will you do instead?** State the proposed interpretation, assumption or alternative and how it differs from the planned approach.
4. **What will change?** Check the impact on scope, actual operations, resources, acceptance conditions and the meaning of the conclusion. Explain which requirements remain satisfiable and which cannot be met or verified. Do not treat a substitute's success as fulfillment of the original operation.
5. **Can you decide autonomously?** Continue when existing authorization covers the action and requirements remain satisfied. Clarify choices that require the user's decision or exceed existing authorization. Routine retries and equivalent implementations need no repeated confirmation.

When disclosure is needed, explain **the problem, the proposed action and its effect on the original requirements**. Use `ambiguity` for consequential ambiguities and `adjustment` for consequential approach changes.

### B. After verification, before reporting: execution, verification and result analysis

Research conclusions require actual execution, verification of the intended behavior, comparisons that support attribution, and a report that reflects the evidence and its limits. A successful command, passing test or higher score alone does not establish that the research objective was achieved.

1. **Did the required operation actually occur?**
   - Trace this invocation through its entry point, effective parameters, executed branches and outputs. Inspect smoke scripts and the implementations they call: did a real backend failure lead to mock, dry-run, local or simplified execution? Were exceptions swallowed, key steps skipped or old caches returned? Do not rely only on test names, top-level success markers or output-file existence.
   - Use this execution's records to establish the backend used, items processed and output provenance. A fallback branch in code does not prove it ran. Obtain targeted verification or state uncertainty when essential records are missing. A substitute's success establishes only the operation that actually ran, not completion of the original one.

2. **Did verification test the success conditions?**
   - Match success conditions to the tests and assertions actually executed. Check whether only smoke tests or a subset ran, important tests were skipped, integration boundaries were mocked, or assertions checked only shapes or file existence instead of correctness, actual calls or the intended effect.
   - After a repair, verify that the intended behavior occurs, not merely that the previous symptom disappears. For filtering, joins or routing changes, check the expected retained and excluded items, their counts and reasons. Excluding everything, producing no output or taking a default path can also make a check appear to pass. Verify both the required behavior and the absence of the original fault before claiming the repair worked.
   - Distinguish a runnable process, an operation that occurred and an outcome that meets requirements. Performance gains cannot compensate for failing required correctness, tolerance or coverage conditions. Missing verification does not itself prove faulty implementation. Obtain the verification needed for the claim within existing authorization, or narrow the claim; do not demand exhaustive testing on every iteration.

3. **Does the evidence support effectiveness or improvement?**
   - Identify the intended change and whether the comparison concerns a method, tuned configurations or complete workflows. To attribute improvement to a change, control other consequential conditions or use comparison experiments to separate their effects. When several factors changed and their contributions cannot be separated, do not attribute the improvement to one factor.
   - Check the actual evaluation samples, splits, preprocessing, labels, scoring code, denominators and aggregation for baseline and candidate, not just filenames or configuration declarations. Check exclusion of difficult cases, failures or abstentions, and whether evaluation data entered training or result selection. Where independent evaluation is required, data used for tuning cannot also provide independent confirmation; ordinary validation-set tuning is not automatically leakage.
   - Establish data separation at the source level. New IDs, different filenames or transformed rows do not establish independent samples. When inputs are exported, augmented or joined, follow their manifests and mappings back to the original records and assigned splits, then check which derived records were actually consumed. An ID-overlap test is sufficient only when the IDs identify the same underlying units across both sets.
   - Compare conditions that should remain comparable: hyperparameters, training epochs, stopping rules, seed schedules, search attempts across all phases, compute resources, hardware and cache state. Parameters that constitute the intended change may vary. Additional data, resource or configuration changes that could affect the gap must enter the comparison and attribution judgment. A score advantage caused by different evaluation data or experimental conditions cannot establish the effectiveness of the intended change.
   - For a selected best result, compare the selection opportunities on both sides: which parameters were searched, how many trials ran, and how the reported result was chosen. Equal conditions within each trial do not establish an equal comparison between search procedures. If one side received more tuning, distinguish "this selected configuration scored higher" from "this method is better"; explain that tuning effort remains an alternative explanation unless appropriate controls separate it. Permission to search does not establish method-level attribution. This is separate from whether the evaluation estimates performance on new data.
   - Check consistency of best-value, mean and trial-selection rules, including selective retention of favorable results. If conditions do not support the comparison, rerun under comparable conditions or obtain necessary controls within authorization. Otherwise identify the differences not ruled out and do not claim that the intended change is superior. Listing differences or saying "only in this experiment" does not explain the attribution limit.

When disclosure is needed, explain **what was actually completed and what material limitations remain**. Use `result`. Describe repaired issues according to the final state and avoid repeating resolved or unchanged findings.

### Shared judgment rules

Disclose evidence-supported issues affecting the task or the user's decision; state uncertainty when an essential premise is unknown. Examples are investigation leads, not automatic findings. Whenever review raises a concern about completion, verification or attribution, you MUST call `beg_evidence` with a concrete question before resolving it, even when uncertain. Do not dismiss a concern based only on summaries or success markers, and do not report it as established before investigation. Follow returned references until the evidence addresses the original concern; a directory, partial excerpt or check of a weaker condition does not resolve it. If evidence remains insufficient, expand the relevant materials, return to authorized work for missing verification, or state uncertainty; an unresolved essential premise cannot be `clear`.

Assess the claim or decision actually being delivered: use `issue` for a supported unmet condition or material limitation needing disclosure, `uncertain` for an unresolved essential premise, and `clear` only when the conclusion is supported and material limits are already reflected in its user-facing wording. Matching records or passing tests alone do not justify `clear`.

## Active checks during ordinary work

Use `result` before adopting or reporting results, `adjustment` before consequential changes following a blocked step, and `ambiguity` for important unresolved requirements. Routine implementation choices need no warning. Hooks cannot infer every decision, so state the proposed claim or decision explicitly. A checkpoint notification calls for review, not automatic disclosure; use the timing and impact criteria above.

1. Read a notified checkpoint with `beg_review(check_id=...)`, or create one with `beg_review(trigger=..., focus=...)`. After verification, use its new checkpoint; if `execution_scope` identifies a later verification, read `latest_check_id`. Without a notification, create a `result` check explicitly before adopting or reporting results.
2. `beg_review` returns original requirements, execution records, the proposed claim or decision and review criteria; it freezes code but does not retrieve code evidence automatically. Apply the checks for the relevant stage. For every concern, call `beg_evidence(check_id, question, refs)` with a concrete question; refs are optional. Use `beg_evidence(check_id, read_ref, offset)` for all further material reads, including review pages, code and linked inputs. Evidence lookup cannot replace an experiment that never ran.
3. Disclose consequential findings at the applicable stage, or continue if none are supported. Save the assessment with `beg_record(check_id, conclusion, summary)`: `clear`, `issue` or `uncertain`, with evidence and handling. Recording returns a short acknowledgement only. Reading a checkpoint is not an assessment; saving one is not user-facing disclosure.

Reuse checks only while the focus, execution and captured state remain applicable. Use `prior_assessments` to avoid repetition. Never use an old judgment to verify a new result. Review is read-only; return to ordinary work for authorized repairs or verification. Stop is a fallback continuation and cannot retract an already displayed response; check before adoption and the first completion report.

For exact quotation formats, evidence expansion, historical sources or pagination, read the relevant section of [Tool workflows](references/tool-workflows.md).
