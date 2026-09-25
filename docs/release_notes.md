# Release preparation

The publication layout separates benchmark construction, methods, evaluation,
analyses, and the independent harness. Local data, raw runs, private sessions,
and previous drafts are excluded from Git.

This reorganization changes the working tree; it does not rewrite older commits.
For a new public repository, use a source export without the existing `.git`
directory, since older commits include raw session records.

## Pending metadata

- Final dataset terms; the public Google Drive release location
  is recorded in `benchmarks/dataset.yaml`.
- Public paper URL and author information after anonymous review.
- External locations for complete paired harness case records.
