"""Run single-sample or batch baseline/EBG predictions."""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Sequence
from urllib.parse import urlsplit


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agentloop import (
    AgentLoop,
    DEFAULT_CONFIG as AGENTLOOP_CONFIG,
    OpenAICompatibleJsonClient,
    RawBackend,
    RunStore,
    prepare_initial_request,
)
from agentloop.benchmark_configs import repo_benchmark_config
from agentloop.errors import AgentLoopError
from ebg.behavior_directory import DIRECTORY_ENCODING
from ebg.evidence_intake import load_visible_bundle
from evaluation_core.contracts import load_prediction_schema
from evaluation_core.messages import (
    build_baseline_messages,
    build_trace_review_messages,
    load_task_prompt,
)
from scripts.main.prepare import token_counter, validate_directory_artifact
from scripts.main.layout import sample_directory, update_summary
from scripts.model_config import (
    add_model_arguments,
    apply_model_settings,
    public_settings,
)
from analysis_experiment.performance_vs_input_size.scripts.tracereview import (
    DEFAULT_CONFIG as TRACE_REVIEW_CONFIG,
    TraceReview,
    prepare_trace_request,
    render_raw_trace_payload,
    render_trace_view,
)


def _run(
    args: argparse.Namespace,
    *,
    backend_factory: Callable[..., Any] | None = None,
    trace_renderer: Callable[..., Any] | None = None,
    task_prompt: str | None = None,
    max_rounds: int | None = None,
    tool_result_budget: int | None = None,
) -> dict[str, Any]:
    artifact_root = (
        Path(args.artifact_root)
        if args.artifact_root
        else PROJECT_ROOT / "evaluation" / args.benchmark / "artifacts"
    )
    store = RunStore(args.output)

    if args.benchmark == "feedbacktrace":
        if args.arm == "graph":
            graph = _load_graph(artifact_root, args.input_id)
            view = (trace_renderer or render_trace_view)(graph)
            messages = build_trace_review_messages(
                input_id=view.input_id,
                task_prompt=(task_prompt if task_prompt is not None
                             else load_task_prompt("EBG", "feedbacktrace")),
                trace_view=view.text,
            )
            prompt_variant = "EBG"
        else:
            payload = _load_raw_trace_payload(artifact_root, args.input_id)
            view = render_raw_trace_payload(payload)
            messages = build_baseline_messages(
                "feedbacktrace",
                trace_payload=payload,
            )
            prompt_variant = "baseline"
        schema = load_prediction_schema(args.schema_root, "feedbacktrace")
        token_count, request_policy = prepare_trace_request(
            messages,
            config=TRACE_REVIEW_CONFIG,
        )
        store.save_trace_view(view.text)
        store.save_request_if_unchanged(
            1,
            messages=messages,
            token_count=token_count,
            compression=request_policy,
            provider_retry=0,
        )
        if args.prepare_only:
            return {
                "status": "prepared",
                "input_id": args.input_id,
                "benchmark": args.benchmark,
                "arm": args.arm,
                "prompt_variant": prompt_variant,
                "estimated_input_tokens": token_count,
            }
        client = _client(args)
        prediction = TraceReview(
            view=view,
            messages=messages,
            prediction_schema=schema,
            client=client,
            store=store,
            config=TRACE_REVIEW_CONFIG,
        ).run()
        return {
            "status": "complete",
            "input_id": args.input_id,
            "benchmark": args.benchmark,
            "arm": args.arm,
            "prompt_variant": prompt_variant,
            "turns": 1,
            "prediction": prediction,
        }

    bundle = load_visible_bundle(
        artifact_root / "visible_bundles" / args.input_id
    )
    if bundle.task_document is None:
        raise AgentLoopError("Repo benchmark bundle lacks its task document")
    benchmark_config = repo_benchmark_config(args.benchmark)
    if max_rounds is not None:
        benchmark_config = replace(benchmark_config, max_rounds=max_rounds)
    loop_config = (
        replace(AGENTLOOP_CONFIG, format_repair_attempts=1)
        if args.benchmark == "silentswap"
        else AGENTLOOP_CONFIG
    )
    if tool_result_budget is not None:
        loop_config = replace(loop_config, tool_result_budget=tool_result_budget)
    benchmark_config.validate_for(bundle.benchmark)
    count_directory_tokens = token_counter(DIRECTORY_ENCODING)
    prompt_variant = "baseline" if args.arm == "raw" else "EBG"
    if args.arm == "raw":
        backend = RawBackend(
            bundle,
            index_budget=AGENTLOOP_CONFIG.index_budget,
            count_tokens=count_directory_tokens,
        )
    else:
        from agentloop.graph_backend import GraphBackend

        graph = _load_graph(artifact_root, args.input_id)
        directory_root = (
            artifact_root
            / "behavior_directories"
            / args.input_id
        )
        graph_root = artifact_root / "behavior_graphs" / args.input_id
        directory_summary = validate_directory_artifact(
            bundle.root,
            graph_root,
            directory_root,
            encoding_name=DIRECTORY_ENCODING,
        )
        directory = _load_json(directory_root / "ranked_directory.json")
        backend = (backend_factory or GraphBackend)(
            bundle,
            graph,
            directory,
            count_tokens=count_directory_tokens,
        )
        if (backend_factory is None
                and backend.initial_index_tokens != int(directory_summary["token_count"])):
            raise AgentLoopError("Ranked Directory token count changed during loading")
    module7 = benchmark_config.bind_module7(
        args.schema_root,
        input_id=bundle.input_id,
        task_document_name=bundle.task_document.path,
        task_document=bundle.task_document.content,
        initial_index=backend.initial_index,
        prompt_variant=prompt_variant,
        source_texts={artifact.path: artifact.content for artifact in bundle.repo_artifacts},
        task_prompt=task_prompt,
    )
    prepared = prepare_initial_request(
        module7.initial_user_prompt,
        module7.max_rounds,
        config=loop_config,
        priority_groups=backend.priority_groups,
    )
    store.save_request_if_unchanged(
        1,
        messages=list(prepared.messages),
        token_count=prepared.token_count,
        compression=prepared.manifest.to_dict(),
        provider_retry=0,
    )
    if args.prepare_only:
        return {
            "status": "prepared",
            "input_id": args.input_id,
            "benchmark": args.benchmark,
            "arm": args.arm,
            "prompt_variant": prompt_variant,
            "estimated_input_tokens": prepared.token_count,
        }
    client = _client(args)
    outcome = AgentLoop(
        backend=backend,
        initial_user_prompt=module7.initial_user_prompt,
        max_rounds=module7.max_rounds,
        finish_contract=module7.finish_contract,
        client=client,
        store=store,
        config=loop_config,
    ).run()
    return {
        "status": outcome.status,
        "input_id": args.input_id,
        "benchmark": args.benchmark,
        "arm": args.arm,
        "prompt_variant": prompt_variant,
        "turns": outcome.turns,
        "prediction": outcome.prediction,
        "failure": outcome.failure,
    }


def _client(args: argparse.Namespace) -> OpenAICompatibleJsonClient:
    base_url = args.base_url or os.environ.get("EBG_API_BASE_URL", "")
    model = args.model or os.environ.get("EBG_MODEL", "")
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key and urlsplit(base_url).hostname == "127.0.0.1":
        api_key = "unused-placeholder"
    if not base_url or not model or not api_key:
        raise AgentLoopError(
            "set --base-url/EBG_API_BASE_URL, --model/EBG_MODEL, and "
            f"the {args.api_key_env} environment variable"
        )
    return OpenAICompatibleJsonClient(
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout=args.timeout,
        request_options=getattr(args, "request_options", {}),
    )


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AgentLoopError(f"expected a JSON object: {path}")
    return value


def _load_graph(artifact_root: Path, input_id: str) -> dict[str, Any]:
    return _load_json(
        artifact_root / "behavior_graphs" / input_id / "behavior_graph.json"
    )


def _load_raw_trace_payload(
    artifact_root: Path, input_id: str
) -> dict[str, Any]:
    bundle_root = artifact_root / "visible_bundles" / input_id
    load_visible_bundle(bundle_root)
    payload = _load_json(bundle_root / "trace" / "model_input.json")
    events = payload.get("events")
    if payload.get("input_id") != input_id or not isinstance(events, list):
        raise AgentLoopError("FeedbackTrace visible input is invalid")
    selectable = sorted(
        str(event["evidence_id"])
        for event in events
        if isinstance(event, dict) and event.get("evidence_id") is not None
    )
    return {
        "input_id": input_id,
        "track": "long",
        "selectable_evidence_ids": selectable,
        "events": events,
    }


run_single_sample = _run

run_sample = _run

BENCHMARKS = ("specgap", "silentswap", "feedbacktrace")

ARMS = ("graph", "raw")

PHASES = ("development", "formal", "full")

MANIFEST_VERSION = 4

FORMAT_FAILURES = {"finish_format_retries_exhausted", "action_format_retries_exhausted"}
RETRY_BENCHMARKS = ("specgap", "silentswap")


class BatchExperimentError(ValueError):
    """Raised when a batch experiment is invalid or cannot be resumed safely."""


@dataclass(frozen=True, slots=True)
class BatchConfig:
    experiment_name: str
    experiment_root: Path
    split_file: Path | None
    phase: str
    benchmarks: tuple[str, ...]
    arms: tuple[str, ...]
    model: str
    base_url: str
    workers: int = 2
    api_key_env: str = "EBG_API_KEY"
    timeout: float = 600.0
    prepare_only: bool = False
    schema_root: Path = PROJECT_ROOT / "schemas"
    request_options: dict[str, Any] | None = None
    artifact_root: Path = PROJECT_ROOT / "evaluation"

    @property
    def output_root(self) -> Path:
        return self.experiment_root / self.experiment_name

    def validate(self) -> None:
        if not self.experiment_name.strip():
            raise BatchExperimentError("experiment_name must be non-empty")
        if self.phase not in PHASES:
            raise BatchExperimentError(f"unsupported phase: {self.phase}")
        _validate_choices(self.benchmarks, BENCHMARKS, "benchmarks")
        _validate_choices(self.arms, ARMS, "arms")
        if not self.model.strip() or not self.base_url.startswith(("http://", "https://")):
            raise BatchExperimentError("model and an HTTP(S) base_url are required")
        if self.workers <= 0:
            raise BatchExperimentError("workers must be positive")
        if self.timeout <= 0:
            raise BatchExperimentError("timeout must be positive")
        if not self.api_key_env.strip():
            raise BatchExperimentError("api_key_env must be non-empty")
        if self.phase == "full" and self.split_file is not None:
            raise BatchExperimentError("full evaluation does not use a split file")
        if self.phase != "full" and self.split_file is None:
            raise BatchExperimentError("development/formal evaluation requires --split-file")


def load_split(path: Path) -> dict[str, Any]:
    """Load and validate a development/formal split without reading Gold."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BatchExperimentError(f"cannot load split: {path}") from error
    if (
        not isinstance(value, dict)
        or not isinstance(value.get("split_id"), str)
        or not value["split_id"].strip()
    ):
        raise BatchExperimentError("split must contain a string split_id")
    benchmark_data = value.get("benchmarks")
    if not isinstance(benchmark_data, dict):
        raise BatchExperimentError("split must contain a benchmarks object")

    expected_counts = value.get("selection", {})
    if not isinstance(expected_counts, dict):
        raise BatchExperimentError("split selection must be an object")
    for benchmark in BENCHMARKS:
        entry = benchmark_data.get(benchmark)
        if not isinstance(entry, dict):
            raise BatchExperimentError(f"split lacks benchmark {benchmark}")
        phases: dict[str, list[str]] = {}
        for phase in ("development", "formal"):
            ids = entry.get(phase)
            if not isinstance(ids, list) or not ids:
                raise BatchExperimentError(
                    f"split {benchmark}.{phase} must be a non-empty list"
                )
            if any(not isinstance(input_id, str) or not input_id.strip() for input_id in ids):
                raise BatchExperimentError(
                    f"split {benchmark}.{phase} contains an invalid input ID"
                )
            if len(ids) != len(set(ids)):
                raise BatchExperimentError(
                    f"split {benchmark}.{phase} contains duplicate input IDs"
                )
            prefix = {
                "specgap": "sg_",
                "silentswap": "ss_",
                "feedbacktrace": "ft_",
            }[benchmark]
            if any(not input_id.startswith(prefix) for input_id in ids):
                raise BatchExperimentError(
                    f"split {benchmark}.{phase} contains a wrong-prefix input ID"
                )
            expected = expected_counts.get(f"{phase}_count")
            if expected is not None and (type(expected) is not int or len(ids) != expected):
                raise BatchExperimentError(
                    f"split {benchmark}.{phase} count differs from selection metadata"
                )
            phases[phase] = ids
        overlap = set(phases["development"]) & set(phases["formal"])
        if overlap:
            raise BatchExperimentError(
                f"split {benchmark} development/formal sets overlap"
            )
    return value


def run_batch(
    config: BatchConfig,
    *,
    run_one: Callable[[argparse.Namespace], dict[str, Any]] = run_single_sample,
    report: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run or resume every requested sample and write an aggregate summary."""

    config.validate()
    split = load_selection(config)
    manifest = _build_manifest(config, split)
    _freeze_manifest(config.output_root / "manifest.json", manifest)
    jobs = _jobs(config, split)
    emit = report or _print_event
    results: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=min(config.workers, len(jobs))) as executor:
        pending = {
            executor.submit(_run_job, config, job, run_one, emit): job
            for job in jobs
        }
        for future in as_completed(pending):
            result = future.result()
            results.append(result)
            emit({"event": "sample_finished", **result})

        retry_jobs = [
            (result["benchmark"], result["arm"], result["input_id"])
            for result in results
            if not config.prepare_only
            and result["status"] == "failed"
            and result["benchmark"] in RETRY_BENCHMARKS
            and not result.get("auto_retried")
        ]
        if retry_jobs:
            emit({"event": "prediction_retry_batch_started", "samples": len(retry_jobs)})
            retries = {
                executor.submit(_run_job, config, job, run_one, emit, attempt=2): job
                for job in retry_jobs
            }
            retried = {}
            for future in as_completed(retries):
                result = future.result()
                retried[retries[future]] = result
                emit({"event": "sample_retried", **result})
            results = [
                retried.get((r["benchmark"], r["arm"], r["input_id"]), r)
                for r in results
            ]

    results.sort(key=lambda item: (item["benchmark"], item["arm"], item["input_id"]))
    summary = {
        "schema_version": 1,
        "experiment_name": config.experiment_name,
        "split_id": split["split_id"],
        "phase": config.phase,
        "samples": results,
        "totals": _aggregate_results(results),
        "groups": _aggregate_groups(results),
    }
    update_summary(config.output_root, prediction=summary)
    finished = {"event": "batch_finished", **summary["totals"]}
    if "automatic_retries" in summary["totals"]:
        counts = summary["totals"]["automatic_retries"]
        finished["message"] = (
            f"首次预测失败 {counts['initial_failures']} 条，"
            f"已自动重测 {counts['retried']} 条，"
            f"重测成功 {counts['recovered']} 条，"
            f"最终仍失败 {summary['totals']['status_counts'].get('failed', 0)} 条。"
        )
    emit(finished)
    return summary


def collect_response_usage(run_root: Path) -> dict[str, int]:
    """Sum provider usage from every persisted successful or failed call."""

    totals = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    response_root = run_root / "responses"
    failure_root = run_root / "provider_failures"
    paths = (
        list(response_root.glob("turn_*.json"))
        + list(failure_root.glob("turn_*_retry_*.json"))
    )
    for path in sorted(paths):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise BatchExperimentError(f"cannot summarize response: {path}") from error
        if not isinstance(value, dict):
            raise BatchExperimentError(f"response metadata is not an object: {path}")
        usage = value.get("usage")
        if not isinstance(usage, dict):
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


def _run_job(
    config: BatchConfig,
    job: tuple[str, str, str],
    run_one: Callable[[argparse.Namespace], dict[str, Any]],
    emit: Callable[[dict[str, Any]], None],
    *,
    attempt: int = 1,
) -> dict[str, Any]:
    benchmark, arm, input_id = job
    run_root = sample_directory(config.output_root, "runs", arm, input_id)
    attempt_root = (
        run_root / f"attempt_{attempt}" if benchmark in RETRY_BENCHMARKS else run_root
    )
    previous = (
        _load_json(run_root / "batch_result.json")
        if benchmark in RETRY_BENCHMARKS and (run_root / "batch_result.json").is_file()
        else {}
    )
    if benchmark in RETRY_BENCHMARKS and attempt == 1 and previous.get("auto_retried"):
        return previous
    emit(
        {
            "event": "sample_started",
            "benchmark": benchmark,
            "arm": arm,
            "input_id": input_id,
            "attempt": attempt,
        }
    )
    args = argparse.Namespace(
        benchmark=benchmark,
        input_id=input_id,
        arm=arm,
        artifact_root=config.artifact_root / benchmark / "artifacts",
        output=attempt_root,
        schema_root=config.schema_root,
        base_url=config.base_url,
        model=config.model,
        api_key_env=config.api_key_env,
        timeout=config.timeout,
        prepare_only=config.prepare_only,
        request_options=config.request_options or {},
    )
    try:
        raw_result = run_one(args)
        if not isinstance(raw_result, dict):
            raise BatchExperimentError("single-sample runner returned a non-object")
        usage = collect_response_usage(attempt_root)
        result = {
            "benchmark": benchmark,
            "arm": arm,
            "input_id": input_id,
            "status": str(raw_result.get("status") or "failed"),
            "turns": raw_result.get("turns"),
            "failure": raw_result.get("failure"),
            "usage": usage,
        }
    except Exception as error:  # Keep unrelated samples resumable after one failure.
        try:
            usage = collect_response_usage(attempt_root)
        except Exception as usage_error:
            usage = {
                "calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            }
            error = BatchExperimentError(f"{error}; usage summary failed: {usage_error}")
        result = {
            "benchmark": benchmark,
            "arm": arm,
            "input_id": input_id,
            "status": "failed",
            "turns": None,
            "failure": f"{type(error).__name__}: {error}",
            "usage": usage,
        }
    if benchmark in RETRY_BENCHMARKS:
        result["usage_scope"] = "successful_attempt_only"
        result["actual_usage"] = dict(result["usage"])
        result["auto_retried"] = attempt == 2
        result["initial_failure"] = (
            result["failure"] or "prediction_failed" if result["status"] == "failed" else None
        )
        result["initial_format_failure"] = (
            result["failure"] if result["failure"] in FORMAT_FAILURES else None
        )
        if attempt == 2:
            result["initial_failure"] = previous.get(
                "initial_failure", previous.get("initial_format_failure")
            )
            result["initial_format_failure"] = previous["initial_format_failure"]
            first_usage = previous["actual_usage"]
            result["actual_usage"] = {
                key: value + first_usage[key]
                for key, value in result["actual_usage"].items()
            }
        if result["status"] != "complete":
            result["usage"] = {key: 0 for key in result["usage"]}
        else:
            prediction_path = attempt_root / "prediction.json"
            if prediction_path.is_file():
                _write_json(run_root / "prediction.json", _load_json(prediction_path))
    _write_json(run_root / "batch_result.json", result)
    return result


def _jobs(config: BatchConfig, split: dict[str, Any]) -> list[tuple[str, str, str]]:
    jobs: list[tuple[str, str, str]] = []
    for benchmark in config.benchmarks:
        ids = split["benchmarks"][benchmark][config.phase]
        for arm in config.arms:
            jobs.extend((benchmark, arm, input_id) for input_id in ids)
    return jobs


def _build_manifest(config: BatchConfig, split: dict[str, Any]) -> dict[str, Any]:
    selected = {
        benchmark: list(split["benchmarks"][benchmark][config.phase])
        for benchmark in config.benchmarks
    }
    manifest = {
        "schema_version": MANIFEST_VERSION,
        "experiment_name": config.experiment_name,
        "split_file": str(config.split_file.resolve()) if config.split_file else None,
        "split_id": split["split_id"],
        "phase": config.phase,
        "selected_ids": selected,
        "benchmarks": list(config.benchmarks),
        "arms": list(config.arms),
        "model": config.model,
        "base_url": config.base_url.rstrip("/"),
        "prediction_requests": {
            benchmark: config.request_options or {}
            for benchmark in config.benchmarks
        },
        "artifact_root": str(config.artifact_root.resolve()),
        "prompt_variants": {"raw": "baseline", "graph": "EBG"},
        "api_key_env": config.api_key_env,
        "timeout": config.timeout,
        "workers": config.workers,
        "prepare_only": config.prepare_only,
        "schema_root": str(config.schema_root.resolve()),
        "single_sample_runner": "scripts.main.predict._run",
    }
    if "silentswap" in config.benchmarks:
        manifest["silentswap_retry_policy"] = {
            "format_repairs_per_attempt": 1,
            "deferred_prediction_failure_reruns": 1,
            "usage_scope": "successful_attempt_only",
        }
    if "specgap" in config.benchmarks:
        manifest["specgap_retry_policy"] = {
            "format_repairs_per_attempt": AGENTLOOP_CONFIG.format_repair_attempts,
            "deferred_prediction_failure_reruns": 1,
            "usage_scope": "successful_attempt_only",
        }
    return manifest


def load_selection(config: BatchConfig) -> dict[str, Any]:
    if config.phase != "full":
        return load_split(config.split_file)
    selected: dict[str, Any] = {}
    for benchmark in config.benchmarks:
        root = config.artifact_root / benchmark / "artifacts" / "visible_bundles"
        ids = sorted(path.name for path in root.iterdir() if path.is_dir())
        if not ids:
            raise BatchExperimentError(f"no visible samples for {benchmark}")
        selected[benchmark] = {"full": ids}
    return {"split_id": "full", "benchmarks": selected}


def _freeze_manifest(path: Path, expected: dict[str, Any]) -> None:
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise BatchExperimentError(f"cannot load experiment manifest: {path}") from error
        legacy_policy = {
            "format_repairs_per_attempt": 1,
            "deferred_format_failure_reruns": 1,
            "usage_scope": "successful_attempt_only",
        }
        upgraded = dict(existing)
        if existing.get("silentswap_retry_policy") == legacy_policy:
            upgraded["silentswap_retry_policy"] = {
                "format_repairs_per_attempt": 1,
                "deferred_prediction_failure_reruns": 1,
                "usage_scope": "successful_attempt_only",
            }
        if upgraded == expected and existing != expected:
            _write_json(path, expected)
            return
        if existing != expected:
            raise BatchExperimentError(
                "experiment manifest differs; use a new experiment name"
            )
        return
    _write_json(path, expected)


def _aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    turns = 0
    for result in results:
        status = result["status"]
        status_counts[status] = status_counts.get(status, 0) + 1
        value = result.get("turns")
        if type(value) is int:
            turns += value
        for key in usage:
            usage[key] += int(result["usage"][key])
    totals = {
        "samples": len(results),
        "status_counts": status_counts,
        "turns": turns,
        "usage": usage,
    }
    if any(r["benchmark"] in RETRY_BENCHMARKS for r in results):
        totals["usage_scope"] = "successful_attempt_only_for_repo"
        totals["automatic_retries"] = {
            "initial_failures": sum(
                bool(r.get("initial_failure", r.get("initial_format_failure")))
                for r in results
            ),
            "initial_format_failures": sum(
                bool(r.get("initial_format_failure")) for r in results
            ),
            "retried": sum(bool(r.get("auto_retried")) for r in results),
            "recovered": sum(
                bool(r.get("auto_retried"))
                and r.get("automatic_retry_status", r["status"]) == "complete"
                for r in results
            ),
            "failed_after_retry": sum(
                bool(r.get("auto_retried"))
                and r.get("automatic_retry_status", r["status"]) == "failed"
                for r in results
            ),
        }
    return totals


def _aggregate_groups(results: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        key = f"{result['benchmark']}/{result['arm']}"
        grouped.setdefault(key, []).append(result)
    return {key: _aggregate_results(values) for key, values in sorted(grouped.items())}


def _usage_integer(usage: dict[str, Any], *names: str) -> int:
    for name in names:
        if name not in usage:
            continue
        value = usage[name]
        if type(value) is not int or value < 0:
            raise BatchExperimentError(f"usage.{name} must be a non-negative integer")
        return value
    return 0


def _validate_choices(values: tuple[str, ...], allowed: tuple[str, ...], label: str) -> None:
    if not values or len(values) != len(set(values)):
        raise BatchExperimentError(f"{label} must be non-empty and unique")
    invalid = [value for value in values if value not in allowed]
    if invalid:
        raise BatchExperimentError(f"unsupported {label}: {invalid}")


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
        settings = apply_model_settings(args)
    except (OSError, ValueError) as error:
        print(json.dumps({"status": "failed", "failure": str(error)}, ensure_ascii=False))
        return 1
    if args.show_config:
        print(json.dumps(public_settings(settings), ensure_ascii=False, indent=2))
        return 0
    if args.input_id:
        if not args.benchmark or len(args.benchmark) != 1:
            raise SystemExit("--input-id requires exactly one --benchmark")
        if args.arm and len(args.arm) != 1:
            raise SystemExit("--input-id accepts exactly one --arm")
        if args.output is None:
            raise SystemExit("--input-id requires --output")
        args.benchmark = args.benchmark[0]
        args.arm = args.arm[0] if args.arm else "graph"
        args.artifact_root = args.artifact_root / args.benchmark / "artifacts"
        try:
            result = run_sample(args)
        except (OSError, ValueError) as error:
            result = {"status": "failed", "failure": str(error)}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] in {"complete", "prepared"} else 1
    if args.output is not None:
        raise SystemExit("--output is only used with --input-id")
    if not args.experiment_name:
        print("--experiment-name is required unless --show-config is used")
        return 1
    benchmarks = tuple(args.benchmark or BENCHMARKS)
    arms = tuple(args.arm or ARMS)
    config = BatchConfig(
        experiment_name=args.experiment_name,
        experiment_root=args.experiment_root,
        split_file=args.split_file,
        phase=args.phase,
        benchmarks=benchmarks,
        arms=arms,
        model=args.model,
        base_url=args.base_url,
        workers=args.workers,
        api_key_env=args.api_key_env,
        timeout=args.timeout,
        prepare_only=args.prepare_only,
        schema_root=args.schema_root,
        request_options=args.request_options,
        artifact_root=args.artifact_root,
    )
    try:
        summary = run_batch(config)
    except (BatchExperimentError, OSError, ValueError) as error:
        print(json.dumps({"status": "failed", "failure": str(error)}, ensure_ascii=False))
        return 1
    return 0 if summary["totals"]["status_counts"].get("failed", 0) == 0 else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a frozen paired prediction batch.")
    parser.add_argument("--experiment-name")
    parser.add_argument("--input-id", help="Run one sample instead of a batch")
    parser.add_argument("--output", type=Path, help="Output directory for --input-id")
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=PROJECT_ROOT / "experiments",
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        default=None,
    )
    parser.add_argument("--phase", default="full", choices=PHASES)
    parser.add_argument("--benchmark", action="append", choices=BENCHMARKS)
    parser.add_argument("--arm", action="append", choices=ARMS)
    add_model_arguments(parser)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--schema-root", type=Path, default=PROJECT_ROOT / "schemas")
    parser.add_argument("--artifact-root", type=Path, default=PROJECT_ROOT / "evaluation")
    parser.add_argument("--prepare-only", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
