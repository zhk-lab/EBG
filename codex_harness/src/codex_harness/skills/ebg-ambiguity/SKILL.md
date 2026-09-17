---
name: ebg-ambiguity
description: Clarify autoresearch goals and experimental protocol before executing a Plan, at a EBG ambiguity checkpoint.
---

## Autoresearch review

Before implementing the Plan, use ebg_evidence to verify goals, Plan and relevant implementation premises. No ambiguity is a valid finding.

1. **Decision rule:** derive an executable selection rule from the authorized goal: primary objective, trade-offs, minimum requirements and keep/revert criteria. Computing several metrics does not define their priority. If reasonable priorities can select different winners and the task does not settle them, this is a consequential ambiguity: ask before selecting or implementing a winner. Do not invent an objective or tie-break rule from customary practice merely because the Plan says "best".
2. Which data versions and splits are used for training, validation and independent testing? Is tuning allowed on validation only? Inspect data loading and preprocessing when the premise depends on code.
3. What may change, and what must stay comparable: evaluation samples, scoring, compute budget, seeds, repetitions and search opportunities?
4. Are conflicting instructions, missing data or implicit defaults consequential to implementation or attribution? Respect settled choices and distinguish the user's protocol from an agent-proposed Plan.

For each question, identify evidence and consequences. Clear data and execution constraints do not resolve a missing decision rule. If an unresolved choice needs user input, investigate that point only, record waiting_for_user=true and ask promptly; do not finish the whole review or implement the dependent choice first. Routine details do not require clarification.

Keep clarification grounded in the unresolved choice. Do not add unverified scores to justify asking: omit numbers that are unnecessary, or verify them with the supplied metric implementation in a minimal calculation. Reading metric code is not evidence that a claimed numerical result was computed correctly.

## Evidence workflow

Use ebg_review(trigger="ambiguity"), mandatory ebg_evidence, then ebg_record. Follow [shared rules](../ebg-review/references/review-rules.md).
