"""Resumable batch runner for paired Repo and BEG predictions."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from agentloop.benchmark_configs import PredictionRequestConfig, repo_benchmark_config
from scripts.agentloop_cli import PROJECT_ROOT, _run as run_single_sample


BENCHMARKS = ("specgap", "silentswap", "feedbacktrace")
ARMS = ("graph", "raw")
PHASES = ("development", "formal")
MANIFEST_VERSION = 3


class BatchExperimentError(ValueError):
    """Raised when a batch experiment is invalid or cannot be resumed safely."""


@dataclass(frozen=True, slots=True)
class BatchConfig:
    experiment_name: str
    experiment_root: Path
    split_file: Path
    phase: str
    benchmarks: tuple[str, ...]
    arms: tuple[str, ...]
    model: str
    base_url: str
    workers: int = 2
    api_key_env: str = "BEG_API_KEY"
    timeout: float = 600.0
    prepare_only: bool = False
    schema_root: Path = PROJECT_ROOT / "schemas"

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
        for phase in PHASES:
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
    split = load_split(config.split_file)
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
    _write_json(config.output_root / "summary.json", summary)
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
) -> dict[str, Any]:
    benchmark, arm, input_id = job
    run_root = (
        config.output_root / "runs" / config.phase / benchmark / arm / input_id
    )
    emit(
        {
            "event": "sample_started",
            "benchmark": benchmark,
            "arm": arm,
            "input_id": input_id,
        }
    )
    args = argparse.Namespace(
        benchmark=benchmark,
        input_id=input_id,
        arm=arm,
        artifact_root=None,
        output=run_root,
        schema_root=config.schema_root,
        base_url=config.base_url,
        model=config.model,
        api_key_env=config.api_key_env,
        timeout=config.timeout,
        prepare_only=config.prepare_only,
    )
    try:
        raw_result = run_one(args)
        if not isinstance(raw_result, dict):
            raise BatchExperimentError("single-sample runner returned a non-object")
        usage = collect_response_usage(run_root)
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
            usage = collect_response_usage(run_root)
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
    return {
        "schema_version": MANIFEST_VERSION,
        "experiment_name": config.experiment_name,
        "split_file": str(config.split_file.resolve()),
        "split_id": split["split_id"],
        "phase": config.phase,
        "selected_ids": selected,
        "benchmarks": list(config.benchmarks),
        "arms": list(config.arms),
        "model": config.model,
        "base_url": config.base_url.rstrip("/"),
        "prediction_requests": {
            benchmark: _prediction_request(benchmark).public_dict()
            for benchmark in config.benchmarks
        },
        "prompt_variants": {"raw": "baseline", "graph": "BEG"},
        "api_key_env": config.api_key_env,
        "timeout": config.timeout,
        "workers": config.workers,
        "prepare_only": config.prepare_only,
        "schema_root": str(config.schema_root.resolve()),
        "single_sample_runner": "scripts.agentloop_cli._run",
    }


def _prediction_request(benchmark: str) -> PredictionRequestConfig:
    if benchmark == "feedbacktrace":
        return PredictionRequestConfig(
            thinking=None,
            reasoning_effort="none",
        )
    return repo_benchmark_config(benchmark).prediction_request


def _freeze_manifest(path: Path, expected: dict[str, Any]) -> None:
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise BatchExperimentError(f"cannot load experiment manifest: {path}") from error
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
    return {
        "samples": len(results),
        "status_counts": status_counts,
        "turns": turns,
        "usage": usage,
    }


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
    )
    try:
        summary = run_batch(config)
    except (BatchExperimentError, OSError, ValueError) as error:
        print(json.dumps({"status": "failed", "failure": str(error)}, ensure_ascii=False))
        return 1
    return 0 if summary["totals"]["status_counts"].get("failed", 0) == 0 else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a frozen paired prediction batch.")
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "experiments",
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "splits" / "luna_glm_20260826.json",
    )
    parser.add_argument("--phase", required=True, choices=PHASES)
    parser.add_argument("--benchmark", action="append", choices=BENCHMARKS)
    parser.add_argument("--arm", action="append", choices=ARMS)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--api-key-env", default="BEG_API_KEY")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--schema-root", type=Path, default=PROJECT_ROOT / "schemas")
    parser.add_argument("--prepare-only", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
