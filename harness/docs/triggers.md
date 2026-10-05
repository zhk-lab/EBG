# Trigger rules

Setup generates SessionStart, UserPromptSubmit, PreToolUse, PostToolUse, and
Stop hooks. Explicit plan execution intent triggers ambiguity review after the
plan is read. Intent that is not recognized automatically can be reviewed with
`ebg_review(trigger="ambiguity", focus=...)`.

Stop evaluates two stages independently. Tool failures, material ambiguity, or
adjustment notes trigger adjustment review. Plan execution, ten tool calls,
300 seconds of active work, or result-limitation notes trigger result review.
When both apply, adjustment comes first. Configure the thresholds with
`--review-call-threshold` and `--review-seconds-threshold` during setup.

Harness calls, duplicate events, idle intervals, and time disconnected from a
restored process are excluded from activity accounting. Unchanged reviewed
material can be reused. A pending clarification allows the Agent to ask and
wait. The Stop fallback requests one continuation and avoids an endless loop.
Its internal ending records the final snapshot without creating another
unhandled review batch; a later user turn can still require a fresh review.

Hooks request evidence-based review. They do not run the task evaluator or
establish task success. Independent tests and paired case records are needed
to assess completion and disclosure quality. The Agent should explicitly
review a result before adopting or reporting it.
