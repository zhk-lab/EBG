# Review workflow

State belongs to a session. Additional prompts and resumed turns reuse that
session; unrelated sessions have separate records.

1. Before implementing a plan, read its requirements and review material
   ambiguity. Existing authorization resolves routine implementation choices.
2. During execution, record material adjustments, blockers, and result limits.
3. Before completion, review triggered adjustments and then results. Inspect
   code, data lineage, execution, and verification with `ebg_evidence`.
4. Record `clear`, `issue`, or `uncertain` with supporting reasons. Communicate
   material findings and answer the original task.

Checkpoints freeze the event cutoff and collected repository text. Later tool
activity requires a new checkpoint. References and text matches establish
relevance, not execution, authorization, or correctness. Binary contents and
uncollected files remain outside the snapshot; collection limits are reported.

When an unresolved choice needs user input, record `waiting_for_user=true` with
the question and impact, then ask the user. Resolve the original checkpoint
only when the reply actually settles the choice. Disclosure alone does not
grant approval. Independent work and already authorized decisions can proceed.

The coordinator skill is `ebg-review`; stage guidance is in `ebg-ambiguity`,
`ebg-adjustment`, and `ebg-result-review`. These are application instructions
distributed with the harness, not the benchmark judge.
