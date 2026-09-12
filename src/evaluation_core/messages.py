"""Shared benchmark prompt loading and message assembly."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .contracts import EvaluationCoreError


PROMPT_ROOT = Path(__file__).resolve().parents[2] / "prompts"
PromptVariant = Literal["BEG", "baseline"]
Benchmark = Literal["specgap", "silentswap", "feedbacktrace"]

_PROMPT_FILES: dict[tuple[str, str], Path] = {
    (variant, benchmark): Path(variant) / f"{benchmark}.txt"
    for variant in ("BEG", "baseline")
    for benchmark in ("specgap", "silentswap", "feedbacktrace")
}
_PROMPT_FILES[("BEG", "silentswap")] = Path("BEG/silentswap1.txt")

def load_task_prompt(
    variant: PromptVariant,
    benchmark: Benchmark,
    *,
    prompt_root: str | Path = PROMPT_ROOT,
) -> str:
    """Load one allowlisted module-seven prompt asset."""

    relative = _PROMPT_FILES.get((variant, benchmark))
    if relative is None:
        raise EvaluationCoreError(
            f"unsupported prompt selection: {variant!r}/{benchmark!r}"
        )
    path = Path(prompt_root) / relative
    try:
        prompt = path.read_text(encoding="utf-8")
    except OSError as error:
        raise EvaluationCoreError(f"cannot load task prompt: {path}") from error
    if not prompt.strip():
        raise EvaluationCoreError(f"task prompt is empty: {path}")
    return prompt.rstrip()


@dataclass(frozen=True, slots=True)
class BaselinePromptTemplate:
    """One baseline protocol, benchmark task, and immutable-input template."""

    system: str
    benchmark: str
    user: str


def load_baseline_prompt(
    benchmark: Benchmark,
    *,
    prompt_root: str | Path = PROMPT_ROOT,
) -> BaselinePromptTemplate:
    """Bind a Raw benchmark task to the shared Repo protocol or Trace protocol."""

    content = load_task_prompt("baseline", benchmark, prompt_root=prompt_root)
    if benchmark == "feedbacktrace":
        return BaselinePromptTemplate(
            system=content,
            benchmark="",
            user="{{TRACE_PAYLOAD_JSON}}",
        )
    from agentloop.config import DEFAULT_CONFIG
    from agentloop.prompts import system_prompt

    return BaselinePromptTemplate(
        system=system_prompt(DEFAULT_CONFIG),
        benchmark=content,
        user=(
            "[[TASK DOCUMENT]]\n{{TASK_DOCUMENT}}\n[[END TASK DOCUMENT]]\n\n"
            "[[INITIAL REPOSITORY INDEX]]\n{{REPOSITORY_MANIFEST}}\n"
            "[[END INITIAL REPOSITORY INDEX]]"
        ),
    )


def build_baseline_messages(
    benchmark: Benchmark,
    *,
    task_document: str | None = None,
    repository_manifest: str | None = None,
    trace_payload: dict[str, Any] | None = None,
    prompt_root: str | Path = PROMPT_ROOT,
) -> list[dict[str, str]]:
    """Render a Raw baseline preview with separated protocol, task, and input."""

    template = load_baseline_prompt(benchmark, prompt_root=prompt_root)
    if benchmark == "feedbacktrace":
        if task_document is not None or repository_manifest is not None:
            raise EvaluationCoreError(
                "FeedbackTrace baseline does not accept repository inputs"
            )
        payload = _validated_trace_payload(trace_payload)
        user = _replace_placeholder(
            template.user,
            "TRACE_PAYLOAD_JSON",
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
        )
    else:
        if trace_payload is not None:
            raise EvaluationCoreError(
                "repository baseline does not accept a trace payload"
            )
        _require_text(task_document, "task document")
        _require_text(repository_manifest, "repository manifest")
        immutable_input = _replace_placeholder(
            template.user,
            "TASK_DOCUMENT",
            task_document.rstrip(),
        )
        immutable_input = _replace_placeholder(
            immutable_input,
            "REPOSITORY_MANIFEST",
            repository_manifest.rstrip(),
        )
        user = (
            "[[BENCHMARK TASK]]\n"
            + template.benchmark.rstrip()
            + "\n[[END BENCHMARK TASK]]\n\n"
            + "[[IMMUTABLE SAMPLE INPUT]]\n"
            + immutable_input.rstrip()
            + "\n[[END IMMUTABLE SAMPLE INPUT]]"
        )
    return [
        {"role": "system", "content": template.system},
        {"role": "user", "content": user},
    ]


def build_repo_initial_user_prompt(
    *,
    input_id: str,
    benchmark: Literal["specgap", "silentswap"],
    task_prompt: str,
    task_document: str,
    initial_index: str,
) -> str:
    """Build the initial message for either repository benchmark."""

    _require_text(input_id, "input ID")
    if benchmark not in {"specgap", "silentswap"}:
        raise EvaluationCoreError("repository benchmark is invalid")
    _require_text(task_prompt, "task prompt")
    _require_text(task_document, "task document")
    _require_text(initial_index, "initial repository index")
    return (
        "[[RUN INPUT]]\n"
        + f"Input ID: {input_id}\nBenchmark: {benchmark}\n"
        + "[[END RUN INPUT]]\n\n"
        + "[[BENCHMARK TASK]]\n"
        + task_prompt.rstrip()
        + "\n[[END BENCHMARK TASK]]\n\n"
        + "[[TASK DOCUMENT]]\n"
        + task_document.rstrip()
        + "\n[[END TASK DOCUMENT]]\n\n"
        + "[[INITIAL REPOSITORY INDEX]]\n"
        + initial_index.rstrip()
        + "\n[[END INITIAL REPOSITORY INDEX]]"
    )


def build_trace_review_messages(
    *,
    input_id: str,
    task_prompt: str,
    trace_view: str,
) -> list[dict[str, str]]:
    """Build the complete one-shot TraceReview request."""

    _require_text(input_id, "input ID")
    _require_text(task_prompt, "task prompt")
    _require_text(trace_view, "trace view")
    return [
        {"role": "system", "content": task_prompt.rstrip()},
        {
            "role": "user",
            "content": (
                "[[RUN INPUT]]\n"
                + f"Input ID: {input_id}\nBenchmark: feedbacktrace\n"
                + "[[END RUN INPUT]]\n\n"
                + "[[COMPLETE TRACE VIEW]]\n"
                + trace_view.rstrip()
                + "\n[[END COMPLETE TRACE VIEW]]"
            ),
        },
    ]


def _require_text(value: str, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise EvaluationCoreError(f"{label} must be non-empty")


def _replace_placeholder(template: str, name: str, value: str) -> str:
    placeholder = "{{" + name + "}}"
    if template.count(placeholder) != 1:
        raise EvaluationCoreError(
            f"baseline prompt requires one {placeholder} placeholder"
        )
    return template.replace(placeholder, value, 1)


def _validated_trace_payload(value: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvaluationCoreError("FeedbackTrace baseline requires a trace payload")
    required = {"input_id", "track", "selectable_evidence_ids", "events"}
    allowed = required | {"early_history_summary"}
    if set(value) - allowed or not required <= set(value):
        raise EvaluationCoreError("FeedbackTrace baseline payload fields are invalid")
    if not isinstance(value["input_id"], str) or not value["input_id"]:
        raise EvaluationCoreError("FeedbackTrace input_id must be non-empty")
    if not isinstance(value["track"], str) or not value["track"]:
        raise EvaluationCoreError("FeedbackTrace track must be non-empty")
    evidence_ids = value["selectable_evidence_ids"]
    if (
        not isinstance(evidence_ids, list)
        or any(not isinstance(item, str) or not item for item in evidence_ids)
        or evidence_ids != sorted(set(evidence_ids))
    ):
        raise EvaluationCoreError(
            "FeedbackTrace selectable_evidence_ids must be sorted and unique"
        )
    if not isinstance(value["events"], list):
        raise EvaluationCoreError("FeedbackTrace events must be an array")
    return value
