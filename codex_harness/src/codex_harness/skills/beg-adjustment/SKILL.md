---
name: beg-adjustment
description: Review autoresearch execution changes and keep/revert reasons at the end of execution, before result review.
---

## Autoresearch review

Reconcile actual experiments with the original goal and Plan. Use beg_evidence to inspect relevant code, before/after versions, execution records and validation; summaries alone cannot establish what happened.

1. Why did the approach change or revert? Do the failing invocation, executed branches and effective parameters support the reason? A fallback in code does not establish it ran.
2. Did the run switch data, splits, preprocessing, metrics, budget, seeds, backend or scoring? Did a real operation become mock, cached, proxy or smaller-sample execution? Trace the concrete change and its effect on the original requirement.
3. Were validation steps skipped, difficult cases excluded, or failed runs discarded? Was the test set used to tune, select models or decide which changes to keep?
4. Are baseline and candidate still comparable? Changed evaluation data require a comparable rerun; extra compute or tuning can confound method-level attribution. If data change is the intended research variable, check the agreed controls instead of forbidding it.
5. Did reverting restore the intended implementation and conditions? Do code versions and run records match the selected result? Identify unfinished validation and whether the baseline needs rerunning.

Separate **procedure completed** from **research objective achieved**. Trace the intended effect through the attempted change, acceptance result and retained implementation. An authorized fallback or successful rollback may restore correctness while leaving the requested improvement unachieved. State that outcome first, explain the code-backed reason, and distinguish a defective implementation from an infeasible research direction. Record no material adjustment only when supported. A pending unauthorized user choice still requires prompt clarification rather than retrospective approval.

## Evidence workflow

Use beg_review(trigger="adjustment"), mandatory beg_evidence, then beg_record. Follow [shared rules](../beg-review/references/review-rules.md).
