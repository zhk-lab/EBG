# Release preparation

The publication layout separates benchmark construction, methods, evaluation,
analyses, and the independent harness. Local data, raw runs, private sessions,
and previous drafts are excluded from Git.

This reorganization changes the working tree; it does not rewrite older commits.
For a new public repository, use a source export without the existing `.git`
directory, since older commits include raw session records.

## Pending metadata

- Final dataset terms; the public Hugging Face release and pinned revision
  are recorded in `benchmarks/dataset.yaml`.
- Public paper URL and author information after anonymous review.
- External locations for complete paired harness case records.

## Result alignment

Numeric records were not changed by the layout migration. The supplied paper
and development tables contain different experiment revisions. The API case in
Table 2 and Appendix F describes different verification outcomes. The dataset
has 484 SpecGAP conditions, while failure analysis uses 482 conditions with
localization references. Select the corresponding case runs and denominator
before final release.

Some historical token entries use a prediction batch different from the scoring
batch. Preserve those source distinctions when exporting cost results.
