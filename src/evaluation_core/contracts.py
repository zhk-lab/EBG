"""Shared benchmark schemas and formal prediction validation."""

from __future__ import annotations

import copy
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePath, PurePosixPath
from typing import Any, Literal

from jsonschema import Draft202012Validator


class EvaluationCoreError(ValueError):
    """Base error for invalid evaluation inputs or outputs."""


class PredictionError(EvaluationCoreError):
    """A final prediction violates its schema or evidence boundary."""


class PredictionFormatError(PredictionError):
    """A final prediction has a repairable JSON/schema shape error."""


class PredictionGroundingError(PredictionError):
    """A final prediction makes an ungrounded or invalid content claim."""


@dataclass(frozen=True, slots=True)
class EvidenceSpan:
    """One source range the model has actually read."""

    path: str
    symbol: str
    start: int
    end: int

    def contains(self, path: str, start: int, end: int) -> bool:
        return self.path == path and self.start <= start <= end <= self.end


RepoBenchmark = Literal["specgap", "silentswap"]

SCHEMA_FILES = {
    "specgap": "specgap_prediction.schema.json",
    "silentswap": "silentswap_prediction.schema.json",
    "feedbacktrace": "feedbacktrace_prediction.schema.json",
}

GROUNDING_PROFILES = {
    "specgap": "specgap_code_evidence_v1",
    "silentswap": "silentswap_target_ranges_v1",
}


def load_prediction_schema(
    schema_root: str | Path,
    benchmark_or_filename: str,
) -> dict[str, Any]:
    """Load and validate one formal schema.

    ``benchmark_or_filename`` accepts a benchmark name for the public CLI and a
    plain filename for declarative benchmark configuration.  Paths are rejected
    so callers cannot accidentally escape the supplied schema root.
    """

    filename = SCHEMA_FILES.get(benchmark_or_filename, benchmark_or_filename)
    if (
        not isinstance(filename, str)
        or not filename
        or PurePath(filename).name != filename
        or not filename.endswith(".json")
    ):
        raise PredictionError(
            f"unsupported prediction schema: {benchmark_or_filename!r}"
        )
    path = Path(schema_root) / filename
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PredictionError(f"cannot load prediction schema: {path}") from error
    if not isinstance(value, dict):
        raise PredictionError("prediction schema must contain an object")
    Draft202012Validator.check_schema(value)
    return value


def model_prediction_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Remove runner-owned fields from the model-visible contract."""

    result = copy.deepcopy(schema)
    properties = result.get("properties")
    required = result.get("required")
    if not isinstance(properties, dict) or not isinstance(required, list):
        raise PredictionError("prediction schema has an invalid top-level contract")
    runner_owned = {"input_id", "benchmark"}
    benchmark = schema.get("properties", {}).get("benchmark", {})
    if isinstance(benchmark, dict) and benchmark.get("const") == "feedbacktrace":
        runner_owned.add("verdict")
    for field in runner_owned:
        properties.pop(field, None)
    result["required"] = [item for item in required if item not in runner_owned]
    return result


class RepoPredictionContract:
    """A schema-bound module-seven contract for one repository benchmark."""

    def __init__(
        self,
        benchmark: RepoBenchmark,
        schema: dict[str, Any],
        *,
        grounding_profile: str | None = None,
        source_texts: dict[str, str] | None = None,
    ) -> None:
        if benchmark not in GROUNDING_PROFILES:
            raise PredictionError(f"unsupported Repo benchmark: {benchmark!r}")
        expected_profile = GROUNDING_PROFILES[benchmark]
        profile = grounding_profile or expected_profile
        if profile != expected_profile:
            raise PredictionError(
                f"grounding profile {profile!r} does not belong to {benchmark}"
            )
        normalized_schema = _json_copy(schema, "prediction schema")
        Draft202012Validator.check_schema(normalized_schema)
        _validate_schema_benchmark(normalized_schema, benchmark)
        self._benchmark = benchmark
        self._schema = normalized_schema
        self._grounding_profile = profile
        self._source_texts = dict(source_texts or {})

    @property
    def prediction_schema(self) -> dict[str, Any]:
        return copy.deepcopy(self._schema)

    @property
    def resume_identity(self) -> dict[str, Any]:
        return {
            "kind": "repo_prediction_contract_v1",
            "benchmark": self._benchmark,
            "grounding_profile": self._grounding_profile,
            "prediction_schema": copy.deepcopy(self._schema),
        }

    def validate(
        self,
        prediction: dict[str, Any],
        *,
        input_id: str,
        observed_spans: Sequence[EvidenceSpan],
    ) -> dict[str, Any]:
        return validate_repo_prediction(
            prediction,
            input_id=input_id,
            benchmark=self._benchmark,
            schema=self._schema,
            observed_spans=observed_spans,
            grounding_profile=self._grounding_profile,
            source_texts=self._source_texts,
        )


def validate_repo_prediction(
    prediction: dict[str, Any],
    *,
    input_id: str,
    benchmark: str,
    schema: dict[str, Any],
    observed_spans: Iterable[EvidenceSpan],
    grounding_profile: str | None = None,
    source_texts: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Validate the formal result and ground every reported source range."""

    if benchmark not in GROUNDING_PROFILES:
        raise PredictionError("repo prediction requires SpecGap or SilentSwap")
    expected_profile = GROUNDING_PROFILES[benchmark]
    profile = grounding_profile or expected_profile
    if profile != expected_profile:
        raise PredictionError(
            f"grounding profile {profile!r} does not belong to {benchmark}"
        )
    normalized_prediction = _normalize_repo_prediction(prediction, benchmark)
    _reject_runner_owned_fields(normalized_prediction)
    wrapped = {
        **normalized_prediction,
        "input_id": input_id,
        "benchmark": benchmark,
    }
    _validate_schema(wrapped, schema)
    spans = tuple(observed_spans)

    if profile == GROUNDING_PROFILES["specgap"]:
        for finding in wrapped["findings"]:
            for evidence in finding["code_evidence"]:
                _validate_location(
                    path=evidence["path"],
                    ranges=(
                        {
                            "start": evidence["start_line"],
                            "end": evidence["end_line"],
                        },
                    ),
                    observed_spans=spans,
                    source_texts=source_texts,
                )
    else:
        for swap in wrapped["swaps"]:
            target = swap["target"]
            _validate_location(
                path=target["file"],
                ranges=target["line_ranges"],
                observed_spans=spans,
                source_texts=source_texts,
            )
    return wrapped


def _normalize_repo_prediction(
    prediction: dict[str, Any], benchmark: str
) -> dict[str, Any]:
    """Normalize a small whitelist of representational aliases before schema checks."""

    result = copy.deepcopy(prediction)
    if benchmark == "specgap":
        findings = result.get("findings")
        if isinstance(findings, list):
            for finding in findings:
                evidence = finding.get("code_evidence") if isinstance(finding, dict) else None
                if not isinstance(evidence, list):
                    continue
                for location in evidence:
                    if isinstance(location, dict) and isinstance(location.get("path"), str):
                        location["path"] = location["path"].replace("\\", "/")
        return result

    swaps = result.get("swaps")
    if not isinstance(swaps, list):
        return result
    for swap in swaps:
        target = swap.get("target") if isinstance(swap, dict) else None
        if not isinstance(target, dict):
            continue
        if isinstance(target.get("file"), str):
            target["file"] = target["file"].replace("\\", "/")
        symbol = target.get("symbol")
        if not isinstance(symbol, dict):
            continue
        if symbol.get("qualified_name") == "":
            symbol["qualified_name"] = []
        if symbol.get("kind") == "property":
            symbol["kind"] = "method"
    return result


def validate_trace_prediction(
    prediction: dict[str, Any],
    *,
    input_id: str,
    schema: dict[str, Any],
    selectable_evidence_ids: set[str],
) -> dict[str, Any]:
    """Validate the one-shot FeedbackTrace result and evidence references."""

    _reject_runner_owned_fields(prediction)
    wrapped = {
        **prediction,
        "input_id": input_id,
        "benchmark": "feedbacktrace",
    }
    _validate_schema(wrapped, schema)
    unknown = [
        item
        for item in wrapped["supporting_evidence_ids"]
        if item not in selectable_evidence_ids
    ]
    if unknown:
        raise PredictionGroundingError(
            f"FeedbackTrace evidence ID was not present in the input: {unknown[0]}"
        )
    return wrapped


def _validate_schema_benchmark(schema: dict[str, Any], benchmark: str) -> None:
    properties = schema.get("properties")
    benchmark_schema = (
        properties.get("benchmark") if isinstance(properties, dict) else None
    )
    if (
        not isinstance(benchmark_schema, dict)
        or benchmark_schema.get("const") != benchmark
    ):
        raise PredictionError(f"prediction schema does not belong to {benchmark}")


def _reject_runner_owned_fields(prediction: dict[str, Any]) -> None:
    owned = {"input_id", "benchmark"} & set(prediction)
    if owned:
        raise PredictionFormatError(
            f"model prediction contains runner-owned field: {sorted(owned)[0]}"
        )


def _validate_schema(value: dict[str, Any], schema: dict[str, Any]) -> None:
    errors = sorted(
        Draft202012Validator(schema).iter_errors(value),
        key=lambda item: tuple(str(part) for part in item.absolute_path),
    )
    if errors:
        first = errors[0]
        location = ".".join(str(item) for item in first.absolute_path) or "prediction"
        raise PredictionFormatError(f"{location}: {first.message}")


def _validate_location(
    *,
    path: str,
    ranges: Iterable[dict[str, Any]],
    observed_spans: tuple[EvidenceSpan, ...],
    source_texts: dict[str, str] | None = None,
) -> None:
    """Enforce shared path/range exposure; the scorer judges symbol accuracy."""

    normalized = _normalize_path(path)
    for item in ranges:
        start = int(item["start"])
        end = int(item["end"])
        if end < start:
            raise PredictionFormatError(
                f"invalid line range {normalized}:{start}-{end}"
            )
        source_text = (source_texts or {}).get(normalized)
        if not _observed_range_contains(observed_spans, normalized, start, end, source_text):
            raise PredictionGroundingError(
                "prediction location was not shown by a successful read: "
                f"{normalized}:{start}-{end}"
            )


def _observed_range_contains(
    observed_spans: tuple[EvidenceSpan, ...],
    path: str,
    start: int,
    end: int,
    source_text: str | None = None,
) -> bool:
    """Cover the citation with read chunks, allowing verified blank gaps between them."""

    cursor = start
    for span in sorted(
        (item for item in observed_spans if item.path == path),
        key=lambda item: (item.start, item.end),
    ):
        if span.end < cursor:
            continue
        if span.start > cursor:
            if cursor == start or source_text is None:
                return False
            lines = source_text.splitlines()
            if span.start - 1 > len(lines) or any(
                line.strip() for line in lines[cursor - 1 : span.start - 1]
            ):
                return False
        cursor = max(cursor, span.end + 1)
        if cursor > end:
            return True
    return False


def _normalize_path(path: str) -> str:
    normalized = str(path).replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    pure = PurePosixPath(normalized)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise PredictionGroundingError(f"invalid repository path: {path!r}")
    return pure.as_posix()


def _json_copy(value: Any, label: str) -> dict[str, Any]:
    try:
        result = json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))
    except (TypeError, ValueError) as error:
        raise PredictionError(f"{label} must be JSON-serializable") from error
    if not isinstance(result, dict):
        raise PredictionError(f"{label} must contain an object")
    return result
