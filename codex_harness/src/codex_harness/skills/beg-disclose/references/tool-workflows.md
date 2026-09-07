# BEG tool workflows

Read the relevant section for evidence lookup or source and pagination details. Apply the decision criteria in SKILL.md.

## Evidence lookup

- `beg_review(check_id=...)` reads review inputs at a frozen checkpoint. `beg_review(trigger=..., focus=...)` creates one; supply relevant `event_ids` or `plan_ids` when known. It returns original Prompts, related call/results, Plan candidates and review criteria without automatically querying code. Plan candidates are not automatically requirements: determine which prompts, amendments and Plan versions apply.
- Whenever a concern arises, even if uncertain, call `beg_evidence(check_id, question, refs)` before judging it. Optional refs cite contiguous original Prompt, Plan or Trace text, for example `{"source_id":"s3","quote":"contiguous original text"}`; use `start` for repeated quotes. Without refs, original requirements and relevant execution calls supply anchors. For returned code, use its actual repo/context/diff source reference. A question is not a requirement or an established finding.
- `beg_evidence(check_id, read_ref, offset)` expands ALL active-review materials: review pages, `all_trace`, `all_sources`, code, linked configuration or input provenance, and history. Omit question and refs when reading a saved reference. Follow `next` with this same tool. Unexpanded content has not been reviewed; an unsuccessful lookup does not prove absence. Missing actual verification requires authorized task execution or an explicit limitation, not more claims based on retrieval alone.
- `beg_record(check_id, conclusion, summary)` saves `clear`, `issue` or `uncertain` with the evidence basis and completed or pending handling. It returns a short acknowledgement, without reloading the evidence package. Do not use `clear` when an essential premise remains unresolved. Recording a judgment does not establish that it was disclosed to the user. Frozen checkpoints retain their original sources, code and execution cutoff; later execution needs a current check.

## Sources and continuation

In user-facing reports, cite code and Plans as `path@lines`, such as `test_harness/retry.py@1-8`. Cite actual commands and relevant output for execution evidence. Quote only the necessary claim text. Omit internal snapshot, event, task, source, checkpoint, and paging identifiers. Identify historical versions as before or after the task; do not present historical line numbers as current locations.

Continue all outputs through `beg_evidence` with `check_id`, `read_ref` and `offset`, omitting `question` and `refs`. Directory entries and `next` provide further pages; `changes` can expose changes not matched to a requirement. Unexpanded material has not been reviewed.

Repository `exists` records observation at capture time. `content_status=not_collected` means unread content, not a missing file. Binary metadata does not prove loadability. Unlisted paths may reflect ignore rules or capture limits, so do not infer absence without checking scope.

Use the checkpoint's frozen evidence; historical references retain their original version and scope. Distinguish supplied initial code, experimental records, simulated reports, and hook history according to their stated origin; not every valid record must come from a hook. A new session clears prior session records and invalidates old references.
