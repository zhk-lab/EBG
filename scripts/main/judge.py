"""Judge saved main experiment predictions and summarize results."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import Any
from urllib.parse import urlsplit


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agentloop.errors import RetryableModelError
from agentloop.provider import (
    ModelClient,
    ModelCompletion,
    OpenAICompatibleJsonClient,
)
from scripts.main.predict import ARMS, BENCHMARKS, PHASES, load_split
from scripts.main.layout import sample_directory, update_summary
from scripts.model_config import (
    add_model_arguments,
    apply_model_settings,
    public_settings,
)

JUDGE_MANIFEST_VERSION = 4

SCORER_VERSION = 2

JUDGE_PROFILE = "official-desktop-v1"

DEFAULT_JUDGE_MODEL = "glm-5-2"

DEFAULT_JUDGE_BASE_URL = "http://127.0.0.1:28080/v1"

DEFAULT_MAX_OUTPUT_TOKENS = 32_768

DEFAULT_NETWORK_RETRIES = 2

DEFAULT_FORMAT_REPAIRS = 1

METRIC_NAMES = {
    "specgap": (
        "gold_precision",
        "recall",
        "f1",
        "question_quality",
        "location_precision",
        "location_recall",
        "location_f1",
    ),
    "silentswap": (
        "localization_score",
        "location_correct",
        "code_change_correct",
    ),
    "feedbacktrace": (
        "verification_point_alignment",
        "evidence_location_score",
        "evidence_hit_rate",
    ),
}


class BatchJudgeError(ValueError):
    """Raised when a Judge batch is invalid or cannot resume safely."""


class JudgeResponseFormatError(BatchJudgeError):
    """Raised when Judge output is not exactly one JSON object."""


class JudgeContentValidationError(BatchJudgeError):
    """Raised when a parsed Judge object violates the scoring contract."""


class JudgeNetworkRetriesExhausted(BatchJudgeError):
    """Raised after the initial HTTP call and two network retries fail."""


@dataclass(frozen=True, slots=True)
class JudgeBatchConfig:
    experiment_name: str
    experiment_root: Path
    phase: str
    benchmarks: tuple[str, ...]
    arms: tuple[str, ...]
    base_url: str = DEFAULT_JUDGE_BASE_URL
    judge_model: str = DEFAULT_JUDGE_MODEL
    api_key_env: str = "GLM_API_KEY"
    workers: int = 3
    timeout: float = 600.0
    network_retries: int = DEFAULT_NETWORK_RETRIES
    format_repairs: int = DEFAULT_FORMAT_REPAIRS
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    artifact_root: Path = PROJECT_ROOT / "evaluation"
    request_options: dict[str, Any] | None = None
    deferred_retries: int = 1

    @property
    def output_root(self) -> Path:
        return self.experiment_root / self.experiment_name

    @property
    def judge_root(self) -> Path:
        return self.output_root / "judges" / self.judge_model

    def validate(self) -> None:
        if not self.experiment_name.strip():
            raise BatchJudgeError("experiment_name must be non-empty")
        if self.phase not in PHASES:
            raise BatchJudgeError(f"unsupported phase: {self.phase}")
        _validate_choices(self.benchmarks, BENCHMARKS, "benchmarks")
        _validate_choices(self.arms, ARMS, "arms")
        if not self.judge_model.strip():
            raise BatchJudgeError("judge_model must be non-empty")
        if not self.base_url.startswith(("http://", "https://")):
            raise BatchJudgeError("base_url must be HTTP(S)")
        if not self.api_key_env.strip():
            raise BatchJudgeError("api_key_env must be non-empty")
        if self.workers <= 0:
            raise BatchJudgeError("workers must be positive")
        if self.network_retries < 0:
            raise BatchJudgeError("network_retries must be non-negative")
        if self.format_repairs not in {0, 1}:
            raise BatchJudgeError("format_repairs must be zero or one")
        if self.deferred_retries not in {0, 1}:
            raise BatchJudgeError("deferred_retries must be zero or one")
        if self.timeout <= 0 or self.max_output_tokens <= 0:
            raise BatchJudgeError("timeout and max_output_tokens must be positive")


def run_batch_judges(
    config: JudgeBatchConfig,
    *,
    client_factory: Callable[[], ModelClient] | None = None,
    report: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Judge every selected saved prediction and write a zero-filled summary."""

    config.validate()
    saved_manifest = config.judge_root / "manifest.json"
    if saved_manifest.is_file():
        # Existing experiments retain their frozen retry policy.
        policy = _read_json(saved_manifest)["retry_policy"]
        config = replace(config, deferred_retries=policy["complete_sample_reruns"])
        config.validate()
    prediction_manifest = _load_prediction_manifest(config)
    selected_ids = _selected_ids(config, prediction_manifest)
    judge_manifest = _build_manifest(config, prediction_manifest, selected_ids)
    _freeze_json(config.judge_root / "manifest.json", judge_manifest)
    make_client = client_factory or _default_client_factory(config)
    modules = {benchmark: _load_judge_module(benchmark) for benchmark in config.benchmarks}
    jobs = [
        (benchmark, arm, input_id)
        for benchmark in config.benchmarks
        for arm in config.arms
        for input_id in selected_ids[benchmark]
    ]
    emit = report or _print_event
    results: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=min(config.workers, len(jobs))) as executor:
        futures = {
            executor.submit(
                _run_job,
                config,
                benchmark,
                arm,
                input_id,
                modules[benchmark],
                make_client,
            ): (benchmark, arm, input_id)
            for benchmark, arm, input_id in jobs
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            emit({"event": "judge_sample_finished", **result})

        if config.deferred_retries:
            retry_results = [item for item in results if _needs_deferred_retry(config, item)]
            if retry_results:
                emit({"event": "judge_retry_batch_started", "samples": len(retry_results)})
                futures = {
                    executor.submit(
                        _run_deferred_retry, config, item,
                        modules[item["benchmark"]], make_client,
                    ): (item["benchmark"], item["arm"], item["input_id"])
                    for item in retry_results
                }
                replacements = {}
                for future in as_completed(futures):
                    result = future.result()
                    replacements[futures[future]] = result
                    emit({"event": "judge_sample_retried", **result})
                results = [
                    replacements.get((r["benchmark"], r["arm"], r["input_id"]), r)
                    for r in results
                ]

    results.sort(key=lambda item: (item["benchmark"], item["arm"], item["input_id"]))
    summary = {
        "schema_version": 1,
        "experiment_name": config.experiment_name,
        "split_id": prediction_manifest["split_id"],
        "phase": config.phase,
        "judge_model": config.judge_model,
        "judge_profile": JUDGE_PROFILE,
        "request_options": config.request_options or {},
        "samples": results,
        "totals": _aggregate_totals(results),
        "groups": _aggregate_groups(results),
    }
    update_summary(config.output_root, judge_model=config.judge_model, judgment=summary)
    emit({"event": "judge_batch_finished", **summary["totals"]})
    return summary


def _needs_deferred_retry(config: JudgeBatchConfig, result: dict[str, Any]) -> bool:
    root = sample_directory(config.output_root, f"judges/{config.judge_model}", result["arm"], result["input_id"])
    if result["status"] == "complete":
        retry_status = root / "retry_1/status.json"
        # A later validated correction must not be replaced by an older failed retry.
        return retry_status.is_file() and _read_json(retry_status).get("status") == "complete"
    # Missing predictions and invalid local inputs require a fix, not a model rerun.
    return str(result.get("failure", "")).startswith((
        "JudgeNetworkRetriesExhausted:", "JudgeResponseFormatError:",
        "JudgeContentValidationError:",
    ))


def _run_deferred_retry(
    config: JudgeBatchConfig, initial: dict[str, Any],
    judge: ModuleType, client_factory: Callable[[], ModelClient],
) -> dict[str, Any]:
    benchmark, arm, input_id = initial["benchmark"], initial["arm"], initial["input_id"]
    root = sample_directory(config.output_root, f"judges/{config.judge_model}", arm, input_id)
    retry_root = root / "retry_1"
    # Freeze the first failure before attempting the one permitted rerun.
    first_path = retry_root / "initial_failure.json"
    if not first_path.is_file():
        _write_json(first_path, initial)
    first = _read_json(first_path)
    status_path = retry_root / "status.json"
    if status_path.is_file():
        result = {**_read_json(status_path), "resumed": True}
    else:
        result = _run_job(
            config, benchmark, arm, input_id, judge, client_factory,
            sample_root=retry_root,
        )
    if result["status"] == "complete":
        _write_json(root / "result.json", _read_json(retry_root / "result.json"))
    result.update(
        auto_retried=True, initial_failure=first["failure"], usage=_attempt_usage(root),
    )
    _write_json(root / "status.json", result)
    return result


def _run_job(
    config: JudgeBatchConfig,
    benchmark: str,
    arm: str,
    input_id: str,
    judge: ModuleType,
    client_factory: Callable[[], ModelClient],
    *,
    sample_root: Path | None = None,
) -> dict[str, Any]:
    sample_root = sample_root or sample_directory(config.output_root, f"judges/{config.judge_model}", arm, input_id)
    prediction_path = sample_directory(config.output_root, "runs", arm, input_id) / "prediction.json"
    try:
        prediction = _read_json(prediction_path)
        if prediction.get("input_id") != input_id:
            raise BatchJudgeError(f"prediction input_id differs: {prediction_path}")
        if prediction.get("benchmark") != benchmark:
            raise BatchJudgeError(f"prediction benchmark differs: {prediction_path}")
        gold_path = (
            config.artifact_root
            / benchmark
            / "artifacts"
            / "hidden_gold"
            / f"{input_id}.json"
        )
        gold = _normalize_gold(benchmark, _read_json(gold_path))
        kwargs = _judge_kwargs(config, benchmark, input_id)
        messages = judge.build_llm_messages(gold, prediction, **kwargs)
        judge_input = {
            "schema_version": 1,
            "input_id": input_id,
            "benchmark": benchmark,
            "arm": arm,
            "prediction_path": str(prediction_path.resolve()),
            "gold_path": str(gold_path.resolve()),
            "prediction": prediction,
            "model_profile": _expected_model_profile(config),
            "messages": messages,
        }
        _freeze_json(sample_root / "judge_input.json", judge_input)

        result_path = sample_root / "result.json"
        if result_path.is_file():
            result = _read_json(result_path)
            _validate_saved_result(judge, benchmark, gold, prediction, result)
            record = _sample_record(
                benchmark,
                arm,
                input_id,
                result=result,
                gold=gold,
                prediction=prediction,
                sample_root=sample_root,
                resumed=True,
            )
            _write_json(sample_root / "status.json", record)
            return record

        result, reused_attempt = _judge_with_retry_policy(
            config,
            judge,
            benchmark,
            gold,
            prediction,
            kwargs,
            messages,
            sample_root,
            client_factory,
        )
        _write_json(result_path, result)
        record = _sample_record(
            benchmark,
            arm,
            input_id,
            result=result,
            gold=gold,
            prediction=prediction,
            sample_root=sample_root,
            resumed=reused_attempt,
        )
    except Exception as error:  # One failed sample must not stop paired groups.
        record = _failed_record(
            benchmark,
            arm,
            input_id,
            sample_root,
            f"{type(error).__name__}: {error}",
        )
    _write_json(sample_root / "status.json", record)
    return record


def _judge_with_retry_policy(
    config: JudgeBatchConfig,
    judge: ModuleType,
    benchmark: str,
    gold: Mapping[str, Any],
    prediction: Mapping[str, Any],
    kwargs: dict[str, Any],
    messages: list[dict[str, str]],
    sample_root: Path,
    client_factory: Callable[[], ModelClient],
) -> tuple[dict[str, Any], bool]:
    content, path, resumed = _call_judge_stage(
        config,
        stage="initial",
        messages=messages,
        sample_root=sample_root,
        client_factory=client_factory,
    )
    try:
        response = _json_object(content)
    except JudgeResponseFormatError as error:
        _mark_attempt(
            path,
            status="json_invalid",
            failure_class="json_format",
            error=error,
        )
        if config.format_repairs == 0:
            raise
        repair_messages = _format_repair_messages(messages, content)
        content, path, repair_resumed = _call_judge_stage(
            config,
            stage="format_repair_1",
            messages=repair_messages,
            sample_root=sample_root,
            client_factory=client_factory,
        )
        resumed = resumed or repair_resumed
        try:
            response = _json_object(content)
        except JudgeResponseFormatError as repair_error:
            _mark_attempt(
                path,
                status="json_invalid",
                failure_class="json_format_repair_exhausted",
                error=repair_error,
            )
            raise JudgeResponseFormatError(
                "Judge JSON format repair was invalid"
            ) from repair_error

    try:
        result = _score_response(judge, benchmark, gold, prediction, response, kwargs)
    except Exception as error:
        _mark_attempt(
            path,
            status="content_invalid",
            failure_class="content_validation",
            error=error,
        )
        raise JudgeContentValidationError(
            f"Judge content validation failed: {type(error).__name__}: {error}"
        ) from error
    _mark_attempt(path, status="valid")
    return result, resumed


def _call_judge_stage(
    config: JudgeBatchConfig,
    *,
    stage: str,
    messages: list[dict[str, str]],
    sample_root: Path,
    client_factory: Callable[[], ModelClient],
) -> tuple[str, Path, bool]:
    resumed = False
    for network_attempt in range(1, config.network_retries + 2):
        path = _attempt_path(sample_root, stage, network_attempt)
        if path.is_file():
            resumed = True
            record = _read_json(path)
            _validate_attempt_identity(record, stage, network_attempt, path)
            if record.get("status") == "network_error":
                continue
            if record.get("status") == "fatal_error":
                raise BatchJudgeError(
                    f"saved Judge provider failure: {record.get('error', 'unknown')}"
                )
            content = record.get("content")
            if not isinstance(content, str):
                raise BatchJudgeError(f"saved Judge attempt lacks content: {path}")
            return content, path, resumed

        record: dict[str, Any] = {
            "schema_version": 2,
            "stage": stage,
            "network_attempt": network_attempt,
            "provider_called": False,
            "status": "pending",
            "content": None,
            "raw_response": None,
            "usage": {},
        }
        try:
            client = client_factory()
            _validate_client_profile(client, config)
            record["provider_called"] = True
            completion = client.complete(
                messages, max_output_tokens=config.max_output_tokens
            )
            if not isinstance(completion, ModelCompletion):
                raise BatchJudgeError("Judge client returned an invalid completion")
            record.update(
                {
                    "status": "received",
                    "content": completion.content,
                    "raw_response": completion.raw_response,
                    "usage": completion.usage,
                }
            )
            _write_json(path, record)
            return completion.content, path, resumed
        except RetryableModelError as error:
            record.update(
                {
                    "status": "network_error",
                    "failure_class": "network",
                    "error": f"{type(error).__name__}: {error}",
                    "raw_response": error.raw_response,
                    "usage": error.usage,
                }
            )
            _write_json(path, record)
            continue
        except Exception as error:
            record.update(
                {
                    "status": "fatal_error",
                    "failure_class": "provider_or_configuration",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            _write_json(path, record)
            raise
    raise JudgeNetworkRetriesExhausted(
        f"Judge network retries exhausted during {stage}: "
        f"{config.network_retries + 1} HTTP attempts"
    )


def _format_repair_messages(
    messages: list[dict[str, str]], content: str
) -> list[dict[str, str]]:
    return [
        *messages,
        {"role": "assistant", "content": content},
        {
            "role": "user",
            "content": (
                "[[JSON FORMAT REPAIR]]\n"
                "Your previous answer was not exactly one valid JSON object. "
                "Return the same judgment as one JSON object that follows the "
                "required schema. Do not add evidence, change scores or revise "
                "the substantive judgment. Output JSON only."
            ),
        },
    ]


def _mark_attempt(
    path: Path,
    *,
    status: str,
    failure_class: str | None = None,
    error: Exception | None = None,
) -> None:
    record = _read_json(path)
    record["status"] = status
    if failure_class is None:
        record.pop("failure_class", None)
    else:
        record["failure_class"] = failure_class
    if error is None:
        record.pop("error", None)
    else:
        record["error"] = f"{type(error).__name__}: {error}"
    _write_json(path, record)


def _validate_attempt_identity(
    record: Mapping[str, Any], stage: str, network_attempt: int, path: Path
) -> None:
    if (
        record.get("schema_version") != 2
        or record.get("stage") != stage
        or record.get("network_attempt") != network_attempt
    ):
        raise BatchJudgeError(f"saved Judge attempt identity differs: {path}")


class _StaticJudgeClient:
    def __init__(self, response: Mapping[str, Any]) -> None:
        self.response = response

    def complete(self, messages: list[dict[str, str]]) -> Mapping[str, Any]:
        return self.response


def _score_response(
    judge: ModuleType,
    benchmark: str,
    gold: Mapping[str, Any],
    prediction: Mapping[str, Any],
    response: Mapping[str, Any],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    result = judge.judge_prediction(
        gold, prediction, _StaticJudgeClient(response), **kwargs
    )
    if not isinstance(result, dict):
        raise BatchJudgeError("Judge scorer returned a non-object")
    return result


def _normalize_specgap_response(
    response: Mapping[str, Any], *, finding_ids: set[str] | None = None
) -> dict[str, Any]:
    """Remove only contradictory zero or duplicate SpecGap assignments.

    Positive semantic scores remain unchanged. Invalid positive scores and missing
    assignments are deliberately left for the benchmark validator to reject.
    """

    matches = response.get("matches")
    unmatched = response.get("unmatched_findings")
    if not isinstance(matches, list) or not isinstance(unmatched, list):
        return dict(response)

    retained_indices: set[int] = set()
    positive: list[tuple[float, int, str, str]] = []
    for index, item in enumerate(matches):
        if not isinstance(item, Mapping):
            retained_indices.add(index)
            continue
        score = item.get("match_score")
        if isinstance(score, (int, float)) and score <= 0:
            continue
        if score not in {0.5, 1.0}:
            retained_indices.add(index)
            continue
        positive.append(
            (
                float(score),
                index,
                str(item.get("finding_id") or ""),
                str(item.get("condition_id") or ""),
            )
        )

    matched_findings: set[str] = set()
    matched_conditions: set[str] = set()
    for _score, index, finding_id, condition_id in sorted(
        positive, key=lambda item: (-item[0], item[1])
    ):
        if finding_id in matched_findings or condition_id in matched_conditions:
            continue
        retained_indices.add(index)
        matched_findings.add(finding_id)
        matched_conditions.add(condition_id)

    normalized = dict(response)
    normalized["matches"] = [
        item for index, item in enumerate(matches) if index in retained_indices
    ]
    normalized["unmatched_findings"] = [
        item
        for item in unmatched
        if not (
            isinstance(item, Mapping)
            and (
                str(item.get("finding_id") or "") in matched_findings
                or (
                    finding_ids is not None
                    and str(item.get("finding_id") or "") not in finding_ids
                )
            )
        )
    ]
    return normalized


def _validate_saved_result(
    judge: ModuleType,
    benchmark: str,
    gold: Mapping[str, Any],
    prediction: Mapping[str, Any],
    result: Mapping[str, Any],
) -> None:
    if result.get("input_id") != prediction.get("input_id"):
        raise BatchJudgeError("saved Judge result has a different input_id")
    if result.get("benchmark") != benchmark:
        raise BatchJudgeError("saved Judge result has a different benchmark")
    if benchmark == "specgap":
        findings = judge.candidate_findings(prediction)
        judgment = judge.validate_llm_response(
            result["llm_judge"]["judgment"],
            finding_ids={item["finding_id"] for item in findings},
            condition_ids={str(item["condition_id"]) for item in gold["conditions"]},
        )
        expected = {
            "input_id": prediction["input_id"],
            "benchmark": benchmark,
            "rule_based": judge.rule_based_score(gold, prediction),
            "llm_judge": {
                "judgment": judgment,
                "metrics": judge.llm_metrics(gold, prediction, judgment),
            },
        }
    elif benchmark == "silentswap":
        expected = {
            "input_id": prediction["input_id"],
            "benchmark": benchmark,
            "rule_based": judge.rule_based_score(gold, prediction),
            "llm_judge": judge.validate_llm_response(
                result["llm_judge"], candidate_count=len(prediction["swaps"])
            ),
        }
    else:
        expected = _validate_feedbacktrace_result(judge, prediction, result)
    if result != expected:
        raise BatchJudgeError("saved Judge result no longer matches its scorer")


def _sample_record(
    benchmark: str,
    arm: str,
    input_id: str,
    *,
    result: Mapping[str, Any],
    gold: Mapping[str, Any],
    prediction: Mapping[str, Any],
    sample_root: Path,
    resumed: bool,
) -> dict[str, Any]:
    return {
        "benchmark": benchmark,
        "arm": arm,
        "input_id": input_id,
        "status": "complete",
        "resumed": resumed,
        "failure": None,
        "metrics": _extract_metrics(benchmark, result, gold, prediction),
        "usage": _attempt_usage(sample_root),
    }


def _failed_record(
    benchmark: str,
    arm: str,
    input_id: str,
    sample_root: Path,
    failure: str,
) -> dict[str, Any]:
    return {
        "benchmark": benchmark,
        "arm": arm,
        "input_id": input_id,
        "status": "failed",
        "resumed": False,
        "failure": failure,
        "metrics": {name: 0.0 for name in METRIC_NAMES[benchmark]},
        "usage": _attempt_usage(sample_root),
    }


def _extract_metrics(
    benchmark: str,
    result: Mapping[str, Any],
    gold: Mapping[str, Any],
    prediction: Mapping[str, Any],
) -> dict[str, float]:
    if benchmark == "specgap":
        llm = result["llm_judge"]["metrics"]
        rule = result["rule_based"]
        values = {
            "gold_precision": llm["gold_precision"],
            "recall": llm["recall"],
            "f1": llm["f1"],
            "question_quality": llm["question_quality"],
            "location_precision": rule["location_precision"],
            "location_recall": rule["location_recall"],
            "location_f1": rule["location_f1"],
        }
    elif benchmark == "silentswap":
        values = {
            "localization_score": result["rule_based"]["localization_score"],
            "location_correct": result["llm_judge"]["scores"][
                "location_correct"
            ],
            "code_change_correct": result["llm_judge"]["scores"][
                "code_change_correct"
            ],
        }
    else:
        values = {
            "verification_point_alignment": result[
                "verification_point_alignment"
            ],
            "evidence_location_score": result["evidence_location_score"],
            "evidence_hit_rate": _feedbacktrace_evidence_hit(gold, prediction),
        }
    normalized = {name: float(value) for name, value in values.items()}
    if any(value < 0 or value > 1 for value in normalized.values()):
        raise BatchJudgeError("Judge metric is outside [0, 1]")
    return normalized


def _feedbacktrace_evidence_hit(
    gold: Mapping[str, Any], prediction: Mapping[str, Any]
) -> float:
    predicted = prediction.get("supporting_evidence_ids")
    expected = gold.get("gold_evidence_ids")
    if prediction.get("verdict") != "KEY":
        return 0.0
    if not isinstance(predicted, list) or not isinstance(expected, list):
        return 0.0
    return float(bool(set(predicted) & set(expected)))


def _aggregate_groups(results: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        groups.setdefault(f"{result['benchmark']}/{result['arm']}", []).append(result)
    return {
        key: _aggregate_group(values)
        for key, values in sorted(groups.items())
    }


def _aggregate_group(results: list[dict[str, Any]]) -> dict[str, Any]:
    selected = len(results)
    completed = [item for item in results if item["status"] == "complete"]
    names = METRIC_NAMES[results[0]["benchmark"]]
    return {
        "selected_samples": selected,
        "completed_samples": len(completed),
        "failed_samples": selected - len(completed),
        "completion_rate": round(len(completed) / selected, 6) if selected else 0.0,
        "resumed_samples": sum(bool(item["resumed"]) for item in completed),
        "score_means": {
            name: _mean(item["metrics"][name] for item in results)
            for name in names
        },
        "score_means_on_completed": {
            name: _mean(item["metrics"][name] for item in completed)
            for name in names
        },
        "usage": _sum_usage(results),
    }


def _aggregate_totals(results: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [item for item in results if item["status"] == "complete"]
    return {
        "selected_samples": len(results),
        "completed_samples": len(completed),
        "failed_samples": len(results) - len(completed),
        "completion_rate": round(len(completed) / len(results), 6) if results else 0.0,
        "usage": _sum_usage(results),
        "automatic_retries": {
            "initial_failures": sum(r["status"] == "failed" or bool(r.get("auto_retried")) for r in results),
            "retried": sum(bool(r.get("auto_retried")) for r in results),
            "recovered": sum(bool(r.get("auto_retried")) and r["status"] == "complete" for r in results),
            "failed_after_retry": sum(bool(r.get("auto_retried")) and r["status"] == "failed" for r in results),
        },
    }


def _sum_usage(results: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    names = ("calls", "input_tokens", "output_tokens", "total_tokens")
    return {
        name: sum(int(item["usage"][name]) for item in results)
        for name in names
    }


def _attempt_usage(sample_root: Path) -> dict[str, int]:
    totals = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    paths = list((sample_root / "attempts").glob("*.json"))
    paths.extend((sample_root / "retry_1/attempts").glob("*.json"))
    paths.extend(sample_root.glob("alignment_correction*/attempts/*.json"))
    for path in sorted(paths):
        value = _read_json(path)
        if value.get("provider_called") is not True:
            continue
        usage = value.get("usage")
        if not isinstance(usage, Mapping):
            usage = {}
        input_tokens = _usage_integer(usage, "input_tokens", "prompt_tokens")
        output_tokens = _usage_integer(usage, "output_tokens", "completion_tokens")
        total_tokens = _usage_integer(usage, "total_tokens")
        if "total_tokens" not in usage:
            total_tokens = input_tokens + output_tokens
        totals["calls"] += 1
        totals["input_tokens"] += input_tokens
        totals["output_tokens"] += output_tokens
        totals["total_tokens"] += total_tokens
    return totals


def _load_prediction_manifest(config: JudgeBatchConfig) -> dict[str, Any]:
    path = config.output_root / "manifest.json"
    value = _read_json(path)
    if value.get("experiment_name") != config.experiment_name:
        raise BatchJudgeError("prediction manifest experiment name differs")
    if value.get("phase") != config.phase:
        raise BatchJudgeError("prediction manifest phase differs")
    if value.get("prepare_only") is True:
        raise BatchJudgeError("cannot judge a prepare-only prediction experiment")
    if config.phase == "full":
        selected = value.get("selected_ids")
        if not isinstance(selected, dict):
            raise BatchJudgeError("prediction manifest lacks selected_ids")
        for benchmark in config.benchmarks:
            ids = selected.get(benchmark)
            if (
                not isinstance(ids, list) or not ids
                or any(not isinstance(item, str) or not item for item in ids)
            ):
                raise BatchJudgeError(f"prediction manifest lacks samples for {benchmark}")
            if len(ids) != len(set(ids)):
                raise BatchJudgeError(f"duplicate prediction samples for {benchmark}")
        return value
    split_file = value.get("split_file")
    if not isinstance(split_file, str):
        raise BatchJudgeError("prediction manifest lacks split_file")
    split = load_split(Path(split_file))
    if split.get("split_id") != value.get("split_id"):
        raise BatchJudgeError("prediction manifest split_id differs from split file")
    selected = value.get("selected_ids")
    if not isinstance(selected, dict):
        raise BatchJudgeError("prediction manifest lacks selected_ids")
    for benchmark in value.get("benchmarks", []):
        if selected.get(benchmark) != split["benchmarks"][benchmark][config.phase]:
            raise BatchJudgeError("prediction manifest selected IDs drifted from split")
    return value


def _selected_ids(
    config: JudgeBatchConfig, manifest: Mapping[str, Any]
) -> dict[str, list[str]]:
    manifest_benchmarks = manifest.get("benchmarks")
    manifest_arms = manifest.get("arms")
    if not isinstance(manifest_benchmarks, list) or not set(config.benchmarks) <= set(
        manifest_benchmarks
    ):
        raise BatchJudgeError("requested benchmark was not in the prediction batch")
    if not isinstance(manifest_arms, list) or not set(config.arms) <= set(manifest_arms):
        raise BatchJudgeError("requested arm was not in the prediction batch")
    selected = manifest["selected_ids"]
    return {benchmark: list(selected[benchmark]) for benchmark in config.benchmarks}


def _build_manifest(
    config: JudgeBatchConfig,
    prediction_manifest: Mapping[str, Any],
    selected_ids: Mapping[str, list[str]],
) -> dict[str, Any]:
    return {
        "schema_version": JUDGE_MANIFEST_VERSION,
        "scorer_version": SCORER_VERSION,
        "experiment_name": config.experiment_name,
        "phase": config.phase,
        "split_id": prediction_manifest["split_id"],
        "selected_ids": selected_ids,
        "benchmarks": list(config.benchmarks),
        "arms": list(config.arms),
        "source_prediction_manifest": dict(prediction_manifest),
        "artifact_root": str(config.artifact_root.resolve()),
        "judge_model": config.judge_model,
        "judge_profile": JUDGE_PROFILE,
        "base_url": config.base_url.rstrip("/"),
        "request_options": config.request_options or {},
        "model_profile": _expected_model_profile(config),
        "json_mode": True,
        "tools": False,
        "api_key_env": config.api_key_env,
        "timeout": config.timeout,
        "retry_policy": {
            "network_retries": config.network_retries,
            "max_http_attempts_per_request": config.network_retries + 1,
            "json_format_repairs": config.format_repairs,
            "content_validation_retries": 0,
            "complete_sample_reruns": config.deferred_retries,
        },
        "max_output_tokens": config.max_output_tokens,
        "workers": config.workers,
    }


def _judge_kwargs(
    config: JudgeBatchConfig, benchmark: str, input_id: str
) -> dict[str, Any]:
    if benchmark == "feedbacktrace":
        path = (
            config.artifact_root
            / benchmark
            / "artifacts"
            / "visible_bundles"
            / input_id
            / "trace"
            / "model_input.json"
        )
        return {"model_input": _read_json(path)}
    if benchmark != "specgap":
        return {}
    path = (
        config.artifact_root
        / benchmark
        / "artifacts"
        / "visible_bundles"
        / input_id
        / "documents"
        / "3_document_after.md"
    )
    return {"document_after": path.read_text(encoding="utf-8")}


def _normalize_gold(benchmark: str, gold: dict[str, Any]) -> dict[str, Any]:
    """Adapt the bundled FeedbackTrace annotation to the latest Judge names."""

    if benchmark != "feedbacktrace":
        return gold
    normalized = dict(gold)
    normalized["gold_verification_point"] = gold.get("verification_point")
    normalized["gold_evidence_ids"] = gold.get("evidence_ids")
    return normalized


def _validate_feedbacktrace_result(
    judge: ModuleType,
    prediction: Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    expected_keys = {
        "input_id",
        "benchmark",
        "verification_point_relation",
        "verification_point_alignment",
        "evidence_evaluation",
        "evidence_location_score",
        "judge_rubric",
    }
    if set(result) != expected_keys:
        raise BatchJudgeError("saved FeedbackTrace Judge fields differ")
    relation = result["verification_point_relation"]
    alignment = judge.verification_point_alignment_from_relation(relation)
    evaluation = result["evidence_evaluation"]
    location = judge.evidence_location_score_from_evaluation(evaluation)
    expected = {
        "input_id": prediction["input_id"],
        "benchmark": "feedbacktrace",
        "verification_point_relation": relation,
        "verification_point_alignment": alignment,
        "evidence_evaluation": evaluation,
        "evidence_location_score": location,
        "judge_rubric": judge.JUDGE_RUBRIC_ID,
    }
    return expected


def _load_judge_module(benchmark: str) -> ModuleType:
    path = PROJECT_ROOT / "evaluation" / benchmark / "judge" / "judge.py"
    spec = importlib.util.spec_from_file_location(f"beg_{benchmark}_judge", path)
    if spec is None or spec.loader is None:
        raise BatchJudgeError(f"cannot load Judge module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    return module


def _default_client_factory(config: JudgeBatchConfig) -> Callable[[], ModelClient]:
    api_key = os.environ.get(config.api_key_env, "")
    if not api_key and urlsplit(config.base_url).hostname == "127.0.0.1":
        api_key = "unused-placeholder"
    if not api_key:
        raise BatchJudgeError(f"environment variable {config.api_key_env} is required")

    def create() -> ModelClient:
        return OpenAICompatibleJsonClient(
            base_url=config.base_url,
            api_key=api_key,
            model=config.judge_model,
            timeout=config.timeout,
            request_options=config.request_options or {},
        )

    return create


def _expected_model_profile(config: JudgeBatchConfig) -> dict[str, Any]:
    return OpenAICompatibleJsonClient(
        base_url=config.base_url,
        api_key="profile-only",
        model=config.judge_model,
        timeout=config.timeout,
        request_options=config.request_options or {},
    ).profile


def _validate_client_profile(client: ModelClient, config: JudgeBatchConfig) -> None:
    if client.profile != _expected_model_profile(config):
        raise BatchJudgeError("Judge client profile differs from the frozen manifest")


def _attempt_path(sample_root: Path, stage: str, network_attempt: int) -> Path:
    return sample_root / "attempts" / f"{stage}_{network_attempt:03d}.json"


def _json_object(content: str) -> dict[str, Any]:
    try:
        value = json.loads(content)
    except json.JSONDecodeError as error:
        raise JudgeResponseFormatError("Judge response is not valid JSON") from error
    if not isinstance(value, dict):
        raise JudgeResponseFormatError("Judge response must be one JSON object")
    return value


def _usage_integer(usage: Mapping[str, Any], *names: str) -> int:
    for name in names:
        if name not in usage:
            continue
        value = usage[name]
        if type(value) is not int or value < 0:
            raise BatchJudgeError(f"usage.{name} must be a non-negative integer")
        return value
    return 0


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return round(sum(items) / len(items), 6) if items else 0.0


def _validate_choices(values: tuple[str, ...], allowed: tuple[str, ...], label: str) -> None:
    if not values or len(values) != len(set(values)):
        raise BatchJudgeError(f"{label} must be non-empty and unique")
    invalid = [value for value in values if value not in allowed]
    if invalid:
        raise BatchJudgeError(f"unsupported {label}: {invalid}")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BatchJudgeError(f"cannot load JSON object: {path}") from error
    if not isinstance(value, dict):
        raise BatchJudgeError(f"JSON input must be an object: {path}")
    return value


def _freeze_json(path: Path, expected: dict[str, Any]) -> None:
    if path.is_file():
        if _read_json(path) != expected:
            raise BatchJudgeError(f"frozen JSON differs: {path}")
        return
    _write_json(path, expected)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="",
    )
    temporary.replace(path)


def _print_event(value: dict[str, Any]) -> None:
    print(json.dumps(value, ensure_ascii=False, separators=(",", ":")), flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        settings = apply_model_settings(args, judge=True)
    except (OSError, ValueError) as error:
        print(json.dumps({"status": "failed", "failure": str(error)}, ensure_ascii=False))
        return 1
    if args.show_config:
        print(json.dumps(public_settings(settings), ensure_ascii=False, indent=2))
        return 0
    if not args.experiment_name:
        print("--experiment-name is required unless --show-config is used")
        return 1
    config = JudgeBatchConfig(
        experiment_name=args.experiment_name,
        experiment_root=args.experiment_root,
        phase=args.phase,
        benchmarks=tuple(args.benchmark or BENCHMARKS),
        arms=tuple(args.arm or ARMS),
        base_url=args.base_url,
        judge_model=args.judge_model,
        api_key_env=args.api_key_env,
        workers=args.workers,
        timeout=args.timeout,
        network_retries=args.network_retries,
        format_repairs=args.format_repairs,
        max_output_tokens=args.max_output_tokens,
        artifact_root=args.artifact_root,
        request_options=args.request_options,
    )
    try:
        summary = run_batch_judges(config)
    except (BatchJudgeError, OSError, ValueError) as error:
        print(json.dumps({"status": "failed", "failure": str(error)}, ensure_ascii=False))
        return 1
    return 0 if summary["totals"]["failed_samples"] == 0 else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-name")
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=PROJECT_ROOT / "experiments",
    )
    parser.add_argument("--phase", default="full", choices=PHASES)
    parser.add_argument("--benchmark", action="append", choices=BENCHMARKS)
    parser.add_argument("--arm", action="append", choices=ARMS)
    add_model_arguments(parser, judge=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument(
        "--network-retries", type=int, default=DEFAULT_NETWORK_RETRIES
    )
    parser.add_argument(
        "--format-repairs", type=int, default=DEFAULT_FORMAT_REPAIRS
    )
    parser.add_argument(
        "--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS
    )
    parser.add_argument(
        "--artifact-root", type=Path, default=PROJECT_ROOT / "evaluation"
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
