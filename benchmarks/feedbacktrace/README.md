# FeedbackTrace construction

The paper evaluates 100 KEY examples using the complete pre-feedback Long
trajectory. Construction uses an authorized local SWE-chat copy and keeps the
target user feedback separate from model-visible inputs.

Use Python 3.12+ and install `requirements.txt`. The pipeline has three stages:

Obtain [SWE-chat](https://huggingface.co/datasets/SALT-NLP/SWE-chat/tree/main)
and place its six source files together under `data/source/swe-chat/`:
`conversations.parquet`, `sessions.parquet`, `checkpoints.parquet`,
`commits.parquet`, `repositories.parquet`, and `session_logs.parquet`.
Copy the provenance example to `benchmarks/feedbacktrace/provenance/swe_chat_source.json`
and `.env.example` to `benchmarks/feedbacktrace/.env`; fill both before running.

```bash
python -m pip install -r benchmarks/feedbacktrace/requirements.txt
python benchmarks/feedbacktrace/scripts/build_feedbacktrace.py extract --provenance /path/to/swe_chat_source.json --source-dir /path/to/swe-chat
python benchmarks/feedbacktrace/scripts/build_feedbacktrace.py pack --provenance /path/to/swe_chat_source.json --source-dir /path/to/swe-chat
python benchmarks/feedbacktrace/scripts/build_feedbacktrace.py validate --provenance /path/to/swe_chat_source.json --source-dir /path/to/swe-chat
```

Replace the two paths with your provenance file and source directory. The
`validate` command runs after human review, not immediately after `pack`.

Copy the [provenance example](provenance/swe_chat_source.example.json) and fill
in the source revision, download date, and actual file sizes. Source files must
remain outside the construction project directory. `extract` joins source
relations, selects candidates, and builds chronological traces plus selectable
evidence units. Model selection and annotation use the two files in `prompts/`.

Before `pack`, configure `DEEPSEEK_FLASH_MODEL_NAME`,
`DEEPSEEK_FLASH_BASE_URL`, `DEEPSEEK_FLASH_REASONING_EFFORT`, and the API key
shown in [.env.example](.env.example). `pack` sends candidate trajectories to
that configured endpoint and saves pending review atoms. It fills remaining
slots and reuses saved queue state. Retain failed staging directories if a run
is interrupted; `--resume-decisions` can reuse their saved decision JSONL.

Review each pending atom for a consequential unconfirmed decision, direct
evidence, accurate consequences, and the pre-feedback cutoff. Save the reviewed
atom under the queue `accepted/` or `rejected/` directory, retaining its
`sample_id`, `source_session_id`, `manifest`, `model_inputs`, and `annotation`.
Use `review` metadata to record the reviewer and any changes. Keep the pending
atom as the original proposal. Rerun `pack` to fill remaining slots.
The default queue is `benchmarks/feedbacktrace/.tmp/feedbacktrace_review_queue_tool_exchange/`.
`validate` aggregates accepted atoms and checks dataset consistency. Generated
queues, source trajectories, and reports are ignored by Git.
The current CLI targets 100 accepted samples; it does not expose a 1–3-sample
pack option. Final native artifacts are under `benchmarks/feedbacktrace/feedbacktrace/`.

The current desktop construction collection and the EBG evaluation collection
are different revisions. Reproducing the paper uses the frozen EBG download,
not freshly regenerated annotations.

Run `python -m pytest benchmarks/feedbacktrace/tests -q` for offline checks.
