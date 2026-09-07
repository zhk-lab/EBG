# BEG tool workflows

Read the relevant section for evidence lookup, a user-requested full review, or source and pagination details. Apply the decision criteria in SKILL.md.

## Evidence lookup

- `beg_context(check_id=...)` reads a frozen checkpoint. `beg_context(trigger=..., focus=...)` creates one; supply relevant `event_ids` or `plan_ids` when known. `beg_context(check_id, read_ref, offset)` expands `all_trace`, `all_sources`, or a context page. Plan candidates are not automatically requirements: determine which prompts, changes, cancellations and Plan versions apply.
- `beg_evidence(check_id, question, refs)` investigates a concrete factual premise. Optional refs cite contiguous original Prompt, Plan or Trace text, for example `{"source_id":"s3","quote":"contiguous original text"}`; use `start` for repeated quotes. Without refs, the checkpoint's original sources provide anchors. For returned code, use its actual repo/context/diff read reference rather than a display label. Follow linked configuration or input provenance when code alone cannot establish the premise.
- `beg_evidence(check_id, read_ref, offset)` expands saved code or related evidence; omit question and refs. A code evidence expansion returned by `beg_context` is read with `beg_evidence`, not `beg_context`. Unexpanded content has not been reviewed.
- `beg_context(check_id, conclusion, summary)` records `clear`, `issue` or `uncertain` with the evidence, impact and completed or pending handling. Do not record `clear` merely to end a check when important evidence remains insufficient. Frozen checkpoints retain their original sources, code and execution cutoff; later execution needs a current check.

## Full review explicitly requested by the user

1. Call `beg_list_task_sources` for Prompt and Markdown Plan versions. Use the current session or a hook-provided `session_id`. Pass `repo_path` when reading current Plans or when no repository is registered. Current code can be reviewed without hook history.
2. At least one applicable Prompt or Plan is sufficient. For historical work, select a continuous Prompt interval and optional Plans, excluding the review request; `end_prompt` must belong to a completed turn. Without an applicable interval, select a Plan directly.
3. Use `beg_select_task(start_prompt, end_prompt, plan_ids)` for history. For Plan-only review, pass `plan_ids` and omit both Prompt endpoints; this captures current code without inventing historical Trace or differences. Read the original sources and retain `task_id`. Do not silently replace an invalid requested historical scope with current code.
4. Consolidate effective requirements, preserving later changes, cancellations, conditions, exceptions, and original quotations. Do not infer requirements backward from implementation. Keep unknown Plan provenance as `unknown`; use `user_plan` or `agent_plan` only when supported.
5. Call `beg_build_evidence_groups(task_id, requirements)`, for example `{"id":"R1","check":"effective requirement","refs":[{"source_id":"P98","quote":"contiguous original text"}]}`. Preserve relevant citations when merging requirements and include `start` for repeated quotations. Submit arguments directly; no preliminary YAML report is needed.
6. Compare requirements, implementation, execution results, and claims. Disclose supported differences and their impact with readable sources. A missing Trace is not itself a task failure. Do not turn each uninspected requirement into an alleged incomplete item. This full-review workflow only discloses; it does not modify code or run repairs or new tests.

## Sources and continuation

In user-facing reports, cite code and Plans as `path@lines`, such as `test_harness/retry.py@1-8`. Cite actual commands and relevant output for execution evidence. Quote only the necessary claim text. Omit internal snapshot, event, task, source, checkpoint, and paging identifiers. Identify historical versions as before or after the task; do not present historical line numbers as current locations.

Continue large outputs with the same tool and its `read_ref` and `offset`. Evidence-group continuation also needs `task_id` and omits `requirements`. Directory entries and `next` provide further pages; `changes` can expose changes not matched to a requirement. Unexpanded material has not been reviewed.

Repository `exists` records observation at capture time. `content_status=not_collected` means unread content, not a missing file. Binary metadata does not prove loadability. Unlisted paths may reflect ignore rules or capture limits, so do not infer absence without checking scope.

Historical reviews use only the selected historical evidence. Plan-only reviews use the selected Plan and code captured at selection. Distinguish supplied initial code, experimental records, simulated reports, and hook history according to their stated origin; not every valid record must come from a hook. A new session clears prior session records and invalidates old references.
