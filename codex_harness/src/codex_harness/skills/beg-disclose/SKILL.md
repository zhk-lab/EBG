---
name: beg-disclose
description: Review requirements and approach changes while planning or executing a task; investigate consequential ambiguity and disclose it before acting. Use beg-result-review for verified results.
---

## Autoresearch review

Use this process review when translating Prompt/Plan requirements into action and before changing approach after a blocked step. Start by examining the requirements and proposed interpretation, rather than waiting for an experiment result. This review concerns what the task means and what action is authorized.

1. **How will you interpret the requirements, and why?** Before implementing a key requirement, state its original wording, the concrete meaning you intend to use, and the source supporting that interpretation. Distinguish what the user explicitly specified, choices covered by existing authorization, and implementation defaults or your own assumptions. A default establishes current behavior; it does not establish that the user selected that meaning. Preserve this distinction in the process assessment instead of describing every intended choice as an already established requirement.
2. **What is unclear or blocked?** Distinguish missing information, conflicting requirements, observed execution failures and routine repairable errors. Identify the specific ambiguity or failure evidence; repeated failure does not authorize relaxing requirements.
   - Start with the consequential defaults and assumptions identified above. Check for other reasonable interpretations supported by the task or project, whether existing instructions settle the choice, and whether different interpretations would change execution, acceptance or conclusions. Investigate concrete uncertainties with `beg_evidence` and targeted verification as needed. If an unresolved interpretation materially affects the task and existing authorization does not cover the choice, promptly pause and explain the ambiguity to the user for clarification. Do not invent alternatives without evidence or reopen choices the user has already settled.
3. **What will you do instead?** State the proposed interpretation, assumption or alternative and how it differs from the planned approach.
4. **What will change?** Check the impact on scope, actual operations, resources, acceptance conditions and the meaning of the conclusion. Explain which requirements remain satisfiable and which cannot be met or verified. Do not treat a substitute's success as fulfillment of the original operation.
5. **Can you decide autonomously?** Continue when existing authorization covers the action and requirements remain satisfied. Clarify choices that require the user's decision or exceed existing authorization. Routine retries and equivalent implementations need no repeated confirmation.

When an ambiguity could materially affect the objective, execution, acceptance criteria or interpretation of results, prefer to pause the current task promptly and report it to the user. Explain what is unclear, how the alternatives could affect the task, and what needs clarification; do not defer this feedback until completion. Keep necessary evidence gathering focused on the ambiguity rather than prolonging task execution before reporting. If the choice requires user input and existing authorization does not cover it, wait for the answer before resuming. Defaults or disclosed assumptions do not replace confirmation. Routine implementation details and choices already settled by existing authorization need no pause or repeated question.

When disclosure is needed, explain **the problem, the proposed action and its effect on the original requirements**. Use `ambiguity` for consequential ambiguities and `adjustment` for consequential approach changes.

## Evidence workflow

Use `ambiguity` for requirement interpretation and `adjustment` for approach changes. Follow the shared [evidence and disclosure rules](references/review-rules.md); `beg_review` includes these with the process checklist. Read [tool workflows](references/tool-workflows.md) when expanding evidence or citing sources.

After verification, switch to `beg-result-review` before adopting or reporting results; a process review cannot verify later execution.
