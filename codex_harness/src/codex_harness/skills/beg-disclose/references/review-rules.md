### Evidence and disclosure

Read a notified checkpoint with `beg_review(check_id=...)`, or create one with the stage's `trigger` and a concrete `focus`. It returns requirements, records and stage-specific guidance, and freezes code without retrieving implementation evidence. Use the current checkpoint for the decision; new execution needs a current check.

For every concern, even an uncertain one, call `beg_evidence(check_id, question)` before resolving it. Expand returned `read_ref` values through `beg_evidence` until the evidence addresses the concern. Optional `refs` must cite source IDs returned by the tools and exact quotations; filenames alone are not source IDs. Evidence lookup cannot replace missing verification. Obtain targeted verification within authorization or state uncertainty.

Disclose evidence-supported findings that affect the task or decision. Examples are investigation leads, not automatic faults. Save evidence and handling with `beg_record`: `issue` for a supported material problem, `uncertain` for an unresolved essential premise, `clear` only for a supported decision whose material limits are reflected in the user-facing wording. A pending user choice cannot become clear merely because it was disclosed.

Recording is not user-facing disclosure or permission to continue. Report at the applicable stage; avoid repeating resolved findings. Stop is a fallback and cannot retract an already displayed response. Review is read-only; return to ordinary work for authorized changes or verification. Reuse prior judgments only while their requirements, execution, state and focus remain applicable.
