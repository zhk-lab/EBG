---
name: beg-result-review
description: Review actual execution, verification and result analysis before adopting or reporting experimental results. Use beg-disclose for requirements and approach changes during execution.
---

## Autoresearch review

Use this result review after verification and before adopting or reporting results, including failed, partial or uncertain results. Internal iterations need no repeated disclosure; report material findings with the round's conclusion. Repaired and reverified routine failures need no repeated warning.

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

If a consequential requirement remains ambiguous or an approach change still needs authorization, return to the process review in `beg-disclose` and clarify it before final adoption. Reporting a limitation does not settle a pending user choice.

## Evidence workflow

Use `result` and follow the shared [evidence and disclosure rules](../beg-disclose/references/review-rules.md); `beg_review` includes these with the result checklist. Read [tool workflows](../beg-disclose/references/tool-workflows.md) when expanding evidence or citing sources.
