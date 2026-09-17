---
name: ebg-review
description: Coordinate EBG autoresearch review when executing a research Plan or reviewing experimental adjustments and results at a EBG checkpoint.
---

# Autoresearch review workflow

Autoresearch iterates on hypotheses, code, experiments, baseline comparisons and keep/revert decisions. Review experimental validity, fair comparisons and justified attribution, not only runnable code.

- Only Plan execution needs ambiguity review: after reading the Plan, before acting. A Plan already in context can be reviewed directly. If the hook missed execution intent, call ebg_review(trigger="ambiguity", focus=the specific Plan).
- During execution retain adjustment/failure evidence; ordinary failures do not require immediate review. Record a material change with ebg_record(note_kind="adjustment", decision_status="proposed" or "executed", summary=reason and impact), note_kind="ambiguity" for a material ambiguity, or note_kind="limitation" for a result limitation. Omit check_id/conclusion: this is an unreviewed note, not permission or an assessment.
- At Stop, select stages independently: result for Plan execution, session call/time thresholds or recorded result limitations; adjustment for tool failures, recorded material ambiguities or important adjustments. A clear ambiguity check alone does not trigger adjustment. Review only applicable stages; when both apply, adjustment precedes result. Waiting for user clarification takes priority. Reuse unchanged completed checks.
- ebg_review returns the matching stage checklist and shared rules. Load only that stage: [ebg-ambiguity](../ebg-ambiguity/SKILL.md), [ebg-adjustment](../ebg-adjustment/SKILL.md), or [ebg-result-review](../ebg-result-review/SKILL.md).
- Every assessment requires ebg_evidence with a concrete question. Follow [shared rules](references/review-rules.md); use [tool workflows](references/tool-workflows.md) for expansion.
- A material user choice not covered by authorization: gather only necessary evidence, ebg_record(waiting_for_user=true, conclusion="uncertain", summary=question and impact), then immediately ask in the final response and wait. Stop must permit that question. After a real reply resolves the choice, record resolution on the waiting checkpoint. A new message alone is not resolution.

Track data isolation, evaluation comparability, allowed variables, resource/search budgets and evidence provenance against the research protocol. Test contamination cannot support independent generalization; scores on different evaluation sets do not establish algorithmic improvement.
