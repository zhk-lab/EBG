---
name: beg-adjustment
description: Review autoresearch execution changes and keep/revert reasons at the end of execution, before result review.
---

## Autoresearch review

At Stop, trigger this stage for tool failures, recorded material ambiguities or important adjustments. Plan execution or call/time thresholds alone do not trigger adjustment. When result is also triggered, review adjustment first.

Autoresearch uses experiments to decide whether to keep a proposed improvement. A recovery can produce valid outputs while changing what the comparison measures. Assess the adjustment against the experimental question and the evidence needed for that decision, not only software correctness. Routine repair authority does not establish experimental equivalence.

1. **Recover the comparison.** Identify the intended improvement, baseline, measured outcome and keep/revert rule. Distinguish the variable under study from conditions needed to interpret its effect: inputs and preprocessing, allocation of work and resources, state initialization and reuse, measurement, and selection opportunities. Derive these from the task and implementation; do not invent a requirement that every internal detail must be identical.
2. **Reconstruct the adjustment.** Connect the failure or constraint to the actual replacement, retry, omission or rollback. Read the original path used by the baseline and the replacement path used by the candidate, including the relevant callees. A note saying "equivalent" is a claim to check. Configuration values, argument names and successful exit codes do not establish that both paths implement the same experimental conditions.
3. **Compare effective behavior.** For each changed path, follow how inputs and state are created, transformed, consumed and reset; what work actually runs; and how outputs enter the score. Record the relevant before/after difference and its evidence. Expand code and execution references until both sides of the comparison are visible. Check for changes in population, dependence between observations, effective effort, aggregation, excluded failures, or opportunities to select a favorable result. Investigate only mechanisms implicated by this adjustment, not an exhaustive checklist of hypothetical risks.
4. **Test the attribution.** Could changing the execution procedure alone alter the measured result even if the proposed algorithmic improvement were absent? Where plausible, separate that additional effect from the intended improvement. A small same-method comparison under the two procedures, a baseline rerun under the adopted conditions, or appropriate repeated trials can help; choose the control that addresses the identified mechanism. Do not demand bitwise identity from stochastic experiments. A distribution-preserving change may be acceptable with suitable evidence, but one favorable run or a generic caveat about variability does not resolve a concrete procedural difference.
5. **Choose a supported disposition.** If the relevant conditions are preserved, explain the evidence and proceed. If they changed, restore the comparison or perform the necessary authorized control before using the result to justify adoption. If that is not feasible, identify which attribution or decision remains unsupported and narrow or defer it. Distinguish intended research changes from incidental changes; an intended change still needs an appropriate control. Missing evidence means uncertainty, not equivalence; a procedural difference alone does not prove the candidate ineffective.

In beg_record, give a concise chain: **adjustment → effective difference or demonstrated preservation → consequence for comparison → handling and remaining limit**. Mark clear only when relevant equivalence is supported or the material effect has been addressed and verified. Record an unresolved concrete threat as issue, or insufficient evidence as uncertain. For a repair or rollback, verify the final implementation and state, and state whether the research objective was achieved. Follow existing authorization and clarification rules; this review does not require asking permission for routine authorized repairs or controls.

## Evidence workflow

Use beg_review(trigger="adjustment"), mandatory beg_evidence, then beg_record. Follow [shared rules](../beg-review/references/review-rules.md).
