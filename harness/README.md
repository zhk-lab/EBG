# EBG Codex review harness

An independent application of EBG for reviewing experimental work. Hooks record
session events and repository snapshots; review tools expose frozen evidence
for decisions about ambiguity, adjustments, and results.

```bash
python -m pip install -e ./harness
python -m codex_harness --state-dir outputs/harness/demo setup --output outputs/harness/integration
```

The setup command generates MCP configuration, hook configuration, and skills.
Use the generated integration files in the target Codex workspace. The
[case runner](case_studies/README.md) prepares this integration in isolated
workspaces for the five paper cases.

| Tool | Purpose |
| --- | --- |
| `ebg_review` | Create or read a checkpoint and its review items |
| `ebg_evidence` | Retrieve frozen code, trace, and referenced materials |
| `ebg_record` | Record findings, process notes, or pending clarification |

Each review must retrieve evidence before recording a conclusion. A recorded
finding does not itself communicate that finding to the user. The Agent must
explain consequential, supported issues in its response.

`src/codex_harness/` contains `tools/`, `hooks/`, `skills/`, `application/`, and
an independent `ebg/` implementation. It does not import the root EBG package.
See [workflow](docs/workflow.md) and [trigger rules](docs/triggers.md).

Run tests from `harness/` after installing its dependencies:

```bash
python -m unittest discover -s tests -t . -q
```

Tests exercise synthetic sessions, frozen snapshots, MCP calls, and standalone
installation. Model behavior in a live Codex session requires separate paired
case evaluation. Runtime state and raw sessions stay in ignored directories.
