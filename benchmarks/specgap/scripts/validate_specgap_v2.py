#!/usr/bin/env python3
"""Strict, offline validation for SpecGAP 2.2 sample directories.

The validator intentionally does not repair or normalize samples.  A valid
sample must be reproducible from the files in its directory alone:

* ``3_document_after.md`` is the exact result of removing the declared spans
  from ``1_document_before.md``;
* each JSON artifact is authoritative for its own payload; no aggregate
  sample or metadata file is required;
* every document, repository, patch, condition-selection, and evidence digest
  matches;
* code evidence points at real, immutable lines in the pristine repository.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import stat
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Sequence
from urllib.parse import urlsplit

if __package__:
    from .repo_evidence import (
        classify_evidence_path,
        extract_repository_snippets,
    )
else:
    from repo_evidence import (  # type: ignore
        classify_evidence_path,
        extract_repository_snippets,
    )


SCHEMA_VERSION = "specgap-2.2"
COLLECTION_SCHEMA_VERSION = "specgap-collection-1.0"
BENCHMARK_TYPE = "spec_gap"

COLLECTION_MANIFEST_FILE = "manifest.json"
DOCUMENT_BEFORE_FILE = "1_document_before.md"
DELETED_PARTS_FILE = "2_deleted_parts.json"
DOCUMENT_AFTER_FILE = "3_document_after.md"
CODE_MAPPING_FILE = "4_code_mapping.json"
ORIGINAL_REPO_DIR = "5_original_repo"
CONDITION_SELECTION_FILE = "condition_selection.json"

COLLECTION_MANIFEST_KEYS = {
    "schema_version",
    "collection",
    "release_eligible",
    "updated_at",
    "samples",
}
COLLECTION_SAMPLE_KEYS = {
    "sample_id",
    "path",
    "annotation_status",
    "mechanical_validation",
    "github_url",
    "parent_commit",
    "generator",
    "model",
    "generated_at",
}

METADATA_KEYS = {
    "schema_version",
    "sample_id",
    "benchmark_type",
    "source",
    "document_before",
    "document_after",
    "original_repo",
    "derivation",
    "annotation",
}

SOURCE_KEYS = {
    "dataset",
    "dataset_file",
    "instance_id",
    "difficulty",
    "pypi_name",
}

ARTIFACT_KEYS = {
    "document_before",
    "deleted_parts",
    "document_after",
    "code_mapping",
    "original_repo",
}

DOCUMENT_KEYS = {"path", "sha256", "chars"}

DELETED_PART_KEYS = {
    "condition_id",
    "type",
    "importance",
    "source_text",
    "normalized_condition",
    "deleted_spans",
    "why_important",
    "sa_gap",
    "expected_verification_question",
    "downstream_impact",
}

DELETED_SPAN_KEYS = {
    "span_id",
    "role",
    "exact_text",
    "char_start",
    "char_end",
    "line_start",
    "line_end",
}

CODE_MAPPING_KEYS = {
    "condition_id",
    "mapping_status",
    "coverage",
    "evidence_basis",
    "mapping_explanation",
    "no_direct_mapping_reason",
    "evidence",
    "related_tests",
}

CODE_MAPPINGS_BUNDLE_REQUIRED_KEYS = {"methods", "comparison_items"}
CODE_MAPPINGS_BUNDLE_OPTIONAL_KEYS = {"gold_labels"}
CODE_MAPPING_METHOD_KEYS = {
    "condition_by_condition",
    "holistic_alignment",
}
CONDITION_BY_CONDITION_METHOD_KEYS = {
    "mode",
    "input_scope",
    "repository_scan",
}
HOLISTIC_ALIGNMENT_METHOD_KEYS = {
    "mode",
    "input_scope",
    "alignment_summary",
    "repository_scan",
}
REPOSITORY_SCAN_KEYS = {
    "indexed_files",
    "indexed_snippets",
    "shards",
}
COMPARISON_ITEM_KEYS = {
    "condition_id",
    "condition_by_condition",
    "holistic_alignment",
    "manual_comparison",
}
MANUAL_COMPARISON_KEYS = {"agreement", "notes"}
MANUAL_AGREEMENT_VALUES = {
    "consistent",
    "partially_consistent",
    "inconsistent",
}
GOLD_LABEL_KEYS = {
    "condition_id",
    "mapping_explanation",
    "gold_source",
}
GOLD_SOURCE_VALUES = {
    "condition_by_condition",
    "holistic_alignment",
    "manually_rewritten",
}

ORIGINAL_REPO_KEYS = {
    "github_url",
    "owner",
    "name",
    "revision",
    "snapshot",
    "license_spdx_id",
    "evaluation",
}

REVISION_KEYS = {"kind", "commit"}
SNAPSHOT_KEYS = {"path", "tree_sha256", "pristine"}
EVALUATION_KEYS = {
    "image_url",
    "workdir",
    "test_patch",
    "test_files",
    "passed_test_ids",
}
TEST_PATCH_KEYS = {"path", "sha256"}

DERIVATION_KEYS = {
    "operation",
    "transform_version",
    "document_truncated",
    "condition_selection_path",
    "condition_selection_sha256",
}

ANNOTATION_KEYS = {
    "status",
    "generator",
    "model",
    "generated_at",
    "notes",
}

PROJECTION_KEYS = {"schema_version", "sample_id"}

CONDITION_TYPES = {
    "api_constraint",
    "data_flow",
    "boundary_behavior",
    "dependency_choice",
    "test_related",
    "metric_or_eval",
    "evidence_boundary",
}
IMPORTANCE_VALUES = {"must_ask", "worth_ask", "can_ignore"}
SA_GAP_VALUES = {"perception", "comprehension", "projection"}
MAPPING_STATUSES = {"direct", "no_direct_mapping"}
REVIEW_STATUSES = {"approved"}
EVIDENCE_TYPES = {
    "implementation",
    "test",
    "configuration",
    "data_flow",
    "fixture",
    "build_metadata",
    "documentation",
}
EVIDENCE_RELATIONS = {
    "implements",
    "enforces",
    "validates",
    "configures",
    "routes",
    "constrains",
    "demonstrates",
    "contradicts",
}
EVIDENCE_STRENGTHS = {"direct", "indirect"}
EVIDENCE_PROVENANCE = {"original_repo"}
EVIDENCE_CONFIDENCE = {"high", "medium", "low"}
RELATED_TEST_STATUSES = {"found", "none_found"}
SYMBOL_KINDS = {
    "function",
    "method",
    "class",
    "module",
    "config_key",
    "data",
    "none",
}
COVERAGE_VALUES = {"full", "partial", "none"}
NO_DIRECT_REASON_CODES = {
    "test_only",
    "configuration_only",
    "data_flow_only",
    "documentation_only",
    "cross_cutting_behavior",
    "external_behavior",
    "evidence_not_found",
    "other",
}

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
SAMPLE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
CONDITION_ID_RE = re.compile(r"^kc_[0-9]{3}$")


@dataclass
class ValidationContext:
    """Accumulate all errors for one sample instead of failing at the first."""

    sample_dir: Path
    errors: list[str] = field(default_factory=list)

    def error(self, location: str, message: str) -> None:
        self.errors.append(f"{location}: {message}")


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(root: Path) -> str:
    """Hash a repository tree using the canonical SpecGAP 2.2 algorithm.

    ``.git`` entries are excluded.  Every other entry must be a regular file
    or directory; symbolic links and special files make the snapshot invalid.
    Paths are sorted by their POSIX UTF-8 byte representation.  For each file,
    the digest receives an eight-byte path length, the path bytes, an
    eight-byte content length, and the raw content.
    """

    if not root.is_dir() or root.is_symlink():
        raise ValueError("repository snapshot is not a real directory")

    entries: list[tuple[bytes, Path]] = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if ".git" in relative.parts:
            continue
        if path.is_symlink():
            raise ValueError(f"symbolic link is not allowed: {relative.as_posix()}")
        mode = path.stat().st_mode
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise ValueError(f"special file is not allowed: {relative.as_posix()}")
        path_bytes = relative.as_posix().encode("utf-8")
        entries.append((path_bytes, path))

    entries.sort(key=lambda item: item[0])
    digest = hashlib.sha256()
    for path_bytes, path in entries:
        content = path.read_bytes()
        digest.update(len(path_bytes).to_bytes(8, "big"))
        digest.update(path_bytes)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_object(
    ctx: ValidationContext, value: Any, location: str
) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        ctx.error(location, "must be an object")
        return None
    return value


def _require_list(
    ctx: ValidationContext, value: Any, location: str
) -> list[Any] | None:
    if not isinstance(value, list):
        ctx.error(location, "must be an array")
        return None
    return value


def _require_exact_keys(
    ctx: ValidationContext,
    value: dict[str, Any],
    expected: set[str],
    location: str,
) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing:
        ctx.error(location, f"missing keys: {', '.join(missing)}")
    if unexpected:
        ctx.error(location, f"unexpected keys: {', '.join(unexpected)}")


def _require_string(
    ctx: ValidationContext,
    value: Any,
    location: str,
    *,
    allow_empty: bool = False,
) -> str | None:
    if not isinstance(value, str):
        ctx.error(location, "must be a string")
        return None
    if not allow_empty and not value.strip():
        ctx.error(location, "must not be empty")
        return None
    return value


def _require_nullable_string(
    ctx: ValidationContext,
    value: Any,
    location: str,
    *,
    allow_empty: bool = False,
) -> str | None:
    if value is None:
        return None
    return _require_string(ctx, value, location, allow_empty=allow_empty)


def _require_enum(
    ctx: ValidationContext,
    value: Any,
    choices: set[str],
    location: str,
) -> str | None:
    result = _require_string(ctx, value, location)
    if result is not None and result not in choices:
        ctx.error(location, f"must be one of {sorted(choices)}, got {result!r}")
        return None
    return result


def _require_sha256(
    ctx: ValidationContext, value: Any, location: str
) -> str | None:
    result = _require_string(ctx, value, location)
    if result is not None and not SHA256_RE.fullmatch(result):
        ctx.error(location, "must be a lowercase 64-character SHA-256 digest")
        return None
    return result


def _validate_unique_strings(
    ctx: ValidationContext,
    value: Any,
    location: str,
    *,
    allowed: set[str] | None = None,
    allow_empty: bool = True,
) -> list[str] | None:
    items = _require_list(ctx, value, location)
    if items is None:
        return None
    if not allow_empty and not items:
        ctx.error(location, "must not be empty")
    strings: list[str] = []
    for index, item in enumerate(items):
        string = _require_string(ctx, item, f"{location}[{index}]")
        if string is None:
            continue
        if allowed is not None and string not in allowed:
            ctx.error(
                f"{location}[{index}]",
                f"must be one of {sorted(allowed)}, got {string!r}",
            )
        strings.append(string)
    if len(strings) != len(set(strings)):
        ctx.error(location, "must not contain duplicate values")
    return strings


def _safe_relative_path(
    ctx: ValidationContext, value: Any, location: str
) -> PurePosixPath | None:
    string = _require_string(ctx, value, location)
    if string is None:
        return None
    if "\\" in string:
        ctx.error(location, "must use POSIX separators, not backslashes")
        return None
    if "\x00" in string:
        ctx.error(location, "must not contain NUL")
        return None
    pure = PurePosixPath(string)
    if pure.is_absolute():
        ctx.error(location, "must be relative")
        return None
    if any(part in {"", ".", ".."} for part in string.split("/")):
        ctx.error(location, "must be a canonical relative path without . or ..")
        return None
    if pure.as_posix() != string:
        ctx.error(location, "must be a canonical POSIX relative path")
        return None
    return pure


def _resolve_under(
    ctx: ValidationContext,
    root: Path,
    relative: PurePosixPath,
    location: str,
) -> Path | None:
    """Resolve a safe relative path and reject symlink traversal."""

    root_resolved = root.resolve()
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            ctx.error(location, f"must not traverse symbolic link {current}")
            return None
    try:
        resolved = current.resolve()
        resolved.relative_to(root_resolved)
    except (OSError, ValueError):
        ctx.error(location, f"path escapes {root}")
        return None
    return current


def _read_bytes(
    ctx: ValidationContext, path: Path | None, location: str
) -> bytes | None:
    if path is None:
        return None
    if not path.is_file() or path.is_symlink():
        ctx.error(location, f"file does not exist or is not regular: {path}")
        return None
    try:
        return path.read_bytes()
    except OSError as exc:
        ctx.error(location, f"cannot read file: {exc}")
        return None


def _decode_utf8(
    ctx: ValidationContext, content: bytes | None, location: str
) -> str | None:
    if content is None:
        return None
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError as exc:
        ctx.error(location, f"must be valid UTF-8: {exc}")
        return None


def _read_json_object(
    ctx: ValidationContext, path: Path, location: str
) -> dict[str, Any] | None:
    content = _read_bytes(ctx, path, location)
    text = _decode_utf8(ctx, content, location)
    if text is None:
        return None

    def reject_duplicate_keys(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate object key {key!r}")
            result[key] = value
        return result

    def reject_non_finite(value: str) -> Any:
        raise ValueError(f"non-finite JSON number {value}")

    try:
        value = json.loads(
            text,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_non_finite,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        ctx.error(location, f"invalid JSON: {exc}")
        return None
    return _require_object(ctx, value, location)


def _validate_document_descriptor(
    ctx: ValidationContext,
    descriptor: Any,
    location: str,
    expected_path: str,
) -> tuple[bytes | None, str | None]:
    obj = _require_object(ctx, descriptor, location)
    if obj is None:
        return None, None
    _require_exact_keys(ctx, obj, DOCUMENT_KEYS, location)

    relative = _safe_relative_path(ctx, obj.get("path"), f"{location}.path")
    if relative is not None and relative.as_posix() != expected_path:
        ctx.error(f"{location}.path", f"must equal {expected_path!r}")
    path = (
        _resolve_under(ctx, ctx.sample_dir, relative, f"{location}.path")
        if relative is not None
        else None
    )
    content = _read_bytes(ctx, path, f"{location}.path")
    text = _decode_utf8(ctx, content, f"{location}.path")

    declared_hash = _require_sha256(
        ctx, obj.get("sha256"), f"{location}.sha256"
    )
    if content is not None and declared_hash is not None:
        actual_hash = sha256_bytes(content)
        if declared_hash != actual_hash:
            ctx.error(
                f"{location}.sha256",
                f"digest mismatch: declared {declared_hash}, actual {actual_hash}",
            )

    chars = obj.get("chars")
    if not _is_int(chars) or chars < 0:
        ctx.error(f"{location}.chars", "must be a non-negative integer")
    elif text is not None and chars != len(text):
        ctx.error(
            f"{location}.chars",
            f"character count mismatch: declared {chars}, actual {len(text)}",
        )
    return content, text


def _line_number_at(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _line_number_for_span_end(text: str, start: int, end: int) -> int:
    # ``char_end`` is exclusive.  The final character, including a trailing
    # newline when one is explicitly deleted, belongs to the preceding line.
    return _line_number_at(text, max(start, end - 1))


def _validate_deleted_parts(
    ctx: ValidationContext,
    value: Any,
    document_before: str | None,
) -> tuple[list[str], list[tuple[int, int, str, str]]]:
    parts = _require_list(ctx, value, "deleted_parts")
    if parts is None:
        return [], []
    if not 3 <= len(parts) <= 5:
        ctx.error("deleted_parts", "must contain exactly 3 to 5 entries")

    condition_ids: list[str] = []
    all_spans: list[tuple[int, int, str, str]] = []
    span_ids: list[str] = []

    for part_index, raw_part in enumerate(parts):
        location = f"deleted_parts[{part_index}]"
        part = _require_object(ctx, raw_part, location)
        if part is None:
            continue
        _require_exact_keys(ctx, part, DELETED_PART_KEYS, location)

        condition_id = _require_string(
            ctx, part.get("condition_id"), f"{location}.condition_id"
        )
        if condition_id is not None:
            if not CONDITION_ID_RE.fullmatch(condition_id):
                ctx.error(
                    f"{location}.condition_id",
                    "must match 'kc_NNN'",
                )
            condition_ids.append(condition_id)

        _require_enum(
            ctx, part.get("type"), CONDITION_TYPES, f"{location}.type"
        )
        _require_enum(
            ctx,
            part.get("importance"),
            IMPORTANCE_VALUES,
            f"{location}.importance",
        )
        for key in (
            "source_text",
            "normalized_condition",
            "why_important",
            "expected_verification_question",
            "downstream_impact",
        ):
            _require_string(ctx, part.get(key), f"{location}.{key}")
        source_text = part.get("source_text")
        if (
            document_before is not None
            and isinstance(source_text, str)
            and source_text
        ):
            occurrence_count = document_before.count(source_text)
            if occurrence_count != 1:
                ctx.error(
                    f"{location}.source_text",
                    "must occur exactly once in document_before; "
                    f"found {occurrence_count} occurrences",
                )
        if part.get("importance") == "can_ignore":
            ctx.error(
                f"{location}.importance",
                "a selected/deleted condition cannot be can_ignore",
            )
        _validate_unique_strings(
            ctx,
            part.get("sa_gap"),
            f"{location}.sa_gap",
            allowed=SA_GAP_VALUES,
            allow_empty=False,
        )

        spans = _require_list(
            ctx, part.get("deleted_spans"), f"{location}.deleted_spans"
        )
        if spans is None:
            continue
        if not spans:
            ctx.error(f"{location}.deleted_spans", "must not be empty")
        for span_index, raw_span in enumerate(spans):
            span_location = f"{location}.deleted_spans[{span_index}]"
            span = _require_object(ctx, raw_span, span_location)
            if span is None:
                continue
            _require_exact_keys(ctx, span, DELETED_SPAN_KEYS, span_location)

            span_id = _require_string(
                ctx, span.get("span_id"), f"{span_location}.span_id"
            )
            if span_id is not None:
                expected_span_id = (
                    f"{condition_id}_ds{span_index + 1:03d}"
                    if condition_id is not None
                    else None
                )
                if span_id != expected_span_id:
                    ctx.error(
                        f"{span_location}.span_id",
                        f"must equal {expected_span_id!r}",
                    )
                span_ids.append(span_id)

            if span.get("role") != "semantic_requirement":
                ctx.error(
                    f"{span_location}.role",
                    "must equal 'semantic_requirement'",
                )
            exact_text = _require_string(
                ctx, span.get("exact_text"), f"{span_location}.exact_text"
            )

            start = span.get("char_start")
            end = span.get("char_end")
            if not _is_int(start):
                ctx.error(f"{span_location}.char_start", "must be an integer")
                start = None
            if not _is_int(end):
                ctx.error(f"{span_location}.char_end", "must be an integer")
                end = None

            if start is not None and end is not None:
                if start < 0 or end <= start:
                    ctx.error(
                        span_location,
                        "must satisfy 0 <= char_start < char_end",
                    )
                if (
                    document_before is not None
                    and 0 <= start < end <= len(document_before)
                ):
                    actual_text = document_before[start:end]
                    if exact_text is not None and exact_text != actual_text:
                        ctx.error(
                            f"{span_location}.exact_text",
                            "does not equal document_before[char_start:char_end]",
                        )
                    line_start = span.get("line_start")
                    line_end = span.get("line_end")
                    expected_line_start = _line_number_at(
                        document_before, start
                    )
                    expected_line_end = _line_number_for_span_end(
                        document_before, start, end
                    )
                    if (
                        not _is_int(line_start)
                        or line_start != expected_line_start
                    ):
                        ctx.error(
                            f"{span_location}.line_start",
                            f"must equal {expected_line_start}",
                        )
                    if not _is_int(line_end) or line_end != expected_line_end:
                        ctx.error(
                            f"{span_location}.line_end",
                            f"must equal {expected_line_end}",
                        )
                    if exact_text is not None and condition_id is not None:
                        all_spans.append(
                            (start, end, exact_text, condition_id)
                        )
                elif document_before is not None:
                    ctx.error(
                        span_location,
                        f"character range is outside document of length "
                        f"{len(document_before)}",
                    )

            if document_before is None or start is None or end is None:
                line_start = span.get("line_start")
                line_end = span.get("line_end")
                if not _is_int(line_start) or line_start < 1:
                    ctx.error(
                        f"{span_location}.line_start",
                        "must be a positive integer",
                    )
                if (
                    not _is_int(line_end)
                    or (
                        _is_int(line_start)
                        and line_end < line_start
                    )
                ):
                    ctx.error(
                        f"{span_location}.line_end",
                        "must be an integer >= line_start",
                    )

    if len(condition_ids) != len(set(condition_ids)):
        ctx.error("deleted_parts", "condition_id values must be unique")
    if len(span_ids) != len(set(span_ids)):
        ctx.error("deleted_parts", "span_id values must be globally unique")

    sorted_spans = sorted(all_spans, key=lambda item: (item[0], item[1]))
    for previous, current in zip(sorted_spans, sorted_spans[1:]):
        if current[0] < previous[1]:
            ctx.error(
                "deleted_parts",
                "deleted spans overlap: "
                f"{previous[3]} [{previous[0]}, {previous[1]}) and "
                f"{current[3]} [{current[0]}, {current[1]})",
            )
    return condition_ids, sorted_spans


def _rebuild_after(
    document_before: str, spans: Sequence[tuple[int, int, str, str]]
) -> str:
    chunks: list[str] = []
    cursor = 0
    for start, end, _exact_text, _condition_id in spans:
        chunks.append(document_before[cursor:start])
        cursor = end
    chunks.append(document_before[cursor:])
    return "".join(chunks)


def _read_projection(
    ctx: ValidationContext,
    filename: str,
    list_key: str,
    expected_sample_id: str | None,
) -> Any:
    path = ctx.sample_dir / filename
    projection = _read_json_object(ctx, path, filename)
    if projection is None:
        return None
    expected_keys = PROJECTION_KEYS | {list_key}
    _require_exact_keys(ctx, projection, expected_keys, filename)
    if projection.get("schema_version") != SCHEMA_VERSION:
        ctx.error(
            f"{filename}.schema_version",
            f"must equal {SCHEMA_VERSION!r}",
        )
    if projection.get("sample_id") != expected_sample_id:
        ctx.error(
            f"{filename}.sample_id",
            "must exactly match the containing directory name",
        )
    return projection.get(list_key)


def _read_condition_selection(
    ctx: ValidationContext,
    expected_sample_id: str | None,
) -> list[str] | None:
    filename = CONDITION_SELECTION_FILE
    selection = _read_json_object(ctx, ctx.sample_dir / filename, filename)
    if selection is None:
        return None
    _require_exact_keys(
        ctx,
        selection,
        {
            "schema_version",
            "sample_id",
            "considered_conditions",
            "selected_condition_ids",
        },
        filename,
    )
    if selection.get("schema_version") != SCHEMA_VERSION:
        ctx.error(
            f"{filename}.schema_version",
            f"must equal {SCHEMA_VERSION!r}",
        )
    if selection.get("sample_id") != expected_sample_id:
        ctx.error(
            f"{filename}.sample_id",
            "must exactly match the containing directory name",
        )
    _require_list(
        ctx,
        selection.get("considered_conditions"),
        f"{filename}.considered_conditions",
    )
    return _validate_unique_strings(
        ctx,
        selection.get("selected_condition_ids"),
        f"{filename}.selected_condition_ids",
        allow_empty=False,
    )


def _canonical_excerpt(
    text: str, ranges: Sequence[tuple[int, int]]
) -> str | None:
    lines = text.splitlines(keepends=True)
    snippets: list[str] = []
    for start, end in ranges:
        if start < 1 or end < start or end > len(lines):
            return None
        snippets.append("".join(lines[start - 1 : end]))
    return "\n".join(snippets)


def _validate_symbol(
    ctx: ValidationContext, value: Any, location: str
) -> tuple[str | None, str | None]:
    symbol = _require_object(ctx, value, location)
    if symbol is None:
        return None, None
    _require_exact_keys(ctx, symbol, {"kind", "qualified_name"}, location)
    kind = _require_enum(
        ctx, symbol.get("kind"), SYMBOL_KINDS, f"{location}.kind"
    )
    qualified_name = _require_nullable_string(
        ctx, symbol.get("qualified_name"), f"{location}.qualified_name"
    )
    return kind, qualified_name


def _validate_line_ranges(
    ctx: ValidationContext, value: Any, location: str
) -> list[tuple[int, int]]:
    raw_ranges = _require_list(ctx, value, location)
    if raw_ranges is None:
        return []
    if not raw_ranges:
        ctx.error(location, "must not be empty")
    result: list[tuple[int, int]] = []
    for index, raw_range in enumerate(raw_ranges):
        item_location = f"{location}[{index}]"
        line_range = _require_object(ctx, raw_range, item_location)
        if line_range is None:
            continue
        _require_exact_keys(ctx, line_range, {"start", "end"}, item_location)
        start = line_range.get("start")
        end = line_range.get("end")
        if not _is_int(start) or start < 1:
            ctx.error(f"{item_location}.start", "must be a positive integer")
            continue
        if not _is_int(end) or end < start:
            ctx.error(
                f"{item_location}.end",
                "must be an integer >= start",
            )
            continue
        result.append((start, end))
    return result


def _python_symbol_ranges(
    text: str,
) -> list[tuple[str, str, int, int]]:
    """Return Python symbols as ``(kind, qualified_name, start, end)``."""

    parse_text = text[1:] if text.startswith("\ufeff") else text
    tree = ast.parse(parse_text)
    result: list[tuple[str, str, int, int]] = []

    class Collector(ast.NodeVisitor):
        def __init__(self) -> None:
            self.stack: list[tuple[str, str]] = []

        @staticmethod
        def node_start(node: ast.AST) -> int:
            starts = [int(getattr(node, "lineno"))]
            for decorator in getattr(node, "decorator_list", []):
                line = getattr(decorator, "lineno", None)
                if line is not None:
                    starts.append(int(line))
            return min(starts)

        def record(self, node: ast.AST, kind: str, name: str) -> None:
            start = self.node_start(node)
            end = int(getattr(node, "end_lineno", start) or start)
            qualified = ".".join(
                [entry[0] for entry in self.stack] + [name]
            )
            result.append((kind, qualified, start, end))

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self.record(node, "class", node.name)
            self.stack.append((node.name, "class"))
            self.generic_visit(node)
            self.stack.pop()

        def visit_function(
            self, node: ast.FunctionDef | ast.AsyncFunctionDef
        ) -> None:
            kind = (
                "method"
                if self.stack and self.stack[-1][1] == "class"
                else "function"
            )
            self.record(node, kind, node.name)
            self.stack.append((node.name, "function"))
            self.generic_visit(node)
            self.stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self.visit_function(node)

        def visit_AsyncFunctionDef(
            self, node: ast.AsyncFunctionDef
        ) -> None:
            self.visit_function(node)

    Collector().visit(tree)
    return result


def _qualified_symbol_matches(declared: str, actual: str) -> bool:
    return declared == actual or declared.endswith(f".{actual}")


def _validate_python_symbol_location(
    ctx: ValidationContext,
    *,
    file_text: str,
    kind: str,
    qualified_name: str,
    ranges: Sequence[tuple[int, int]],
    location: str,
) -> None:
    try:
        symbols = _python_symbol_ranges(file_text)
    except (SyntaxError, TypeError, ValueError) as exc:
        ctx.error(location, f"cannot parse Python file to verify symbol: {exc}")
        return

    matches = [
        (start, end)
        for actual_kind, actual_name, start, end in symbols
        if actual_kind == kind
        and _qualified_symbol_matches(qualified_name, actual_name)
    ]
    if not matches:
        ctx.error(
            location,
            f"declared {kind} {qualified_name!r} does not exist in the file",
        )
        return
    if ranges and not any(
        all(symbol_start <= start and end <= symbol_end for start, end in ranges)
        for symbol_start, symbol_end in matches
    ):
        ctx.error(
            location,
            "declared line ranges are not contained in the declared symbol",
        )


def _evidence_type_matches_path(
    evidence_type: str, indexed_type: str
) -> bool:
    compatible = {
        # Executable doctests legitimately live inside implementation files.
        "implementation": {"implementation", "test", "data_flow"},
        "test": {"test", "fixture", "data_flow"},
        "fixture": {"fixture", "test", "data_flow"},
        "configuration": {"configuration", "data_flow"},
        "build_metadata": {
            "build_metadata",
            "configuration",
            "data_flow",
        },
        "documentation": {"documentation", "data_flow"},
    }
    return evidence_type in compatible.get(indexed_type, {indexed_type})


def _validate_evidence_location(
    ctx: ValidationContext,
    evidence_type: str | None,
    location_value: Any,
    location: str,
    repo_root: Path | None,
) -> int | None:
    evidence_location = _require_object(ctx, location_value, location)
    if evidence_location is None:
        return None
    evidence_location_keys = {
        "file_path",
        "file_sha256",
        "symbol",
        "line_ranges",
        "excerpt",
        "excerpt_sha256",
        "flow_step",
    }
    _require_exact_keys(
        ctx, evidence_location, evidence_location_keys, location
    )

    relative = _safe_relative_path(
        ctx,
        evidence_location.get("file_path"),
        f"{location}.file_path",
    )
    if relative is not None and ".git" in relative.parts:
        ctx.error(
            f"{location}.file_path",
            "must not reference repository metadata under .git",
        )
        relative = None
    if relative is not None and evidence_type is not None:
        indexed_type = classify_evidence_path(relative)
        if indexed_type is None:
            ctx.error(
                f"{location}.file_path",
                "is not an eligible repository-index evidence file",
            )
        elif not _evidence_type_matches_path(evidence_type, indexed_type):
            ctx.error(
                f"{location}.evidence_type",
                f"{evidence_type!r} is incompatible with indexed path type "
                f"{indexed_type!r}",
            )
    file_path = (
        _resolve_under(ctx, repo_root, relative, f"{location}.file_path")
        if repo_root is not None and relative is not None
        else None
    )
    file_content = _read_bytes(ctx, file_path, f"{location}.file_path")
    file_text = _decode_utf8(ctx, file_content, f"{location}.file_path")
    declared_file_hash = _require_sha256(
        ctx,
        evidence_location.get("file_sha256"),
        f"{location}.file_sha256",
    )
    if file_content is not None and declared_file_hash is not None:
        actual_file_hash = sha256_bytes(file_content)
        if declared_file_hash != actual_file_hash:
            ctx.error(
                f"{location}.file_sha256",
                f"digest mismatch: declared {declared_file_hash}, "
                f"actual {actual_file_hash}",
            )

    kind, qualified_name = _validate_symbol(
        ctx, evidence_location.get("symbol"), f"{location}.symbol"
    )
    if (
        evidence_type == "implementation"
        and kind in {"function", "method", "class"}
        and qualified_name is None
    ):
        ctx.error(
            f"{location}.symbol.qualified_name",
            "implementation function/class/method evidence requires a symbol",
        )

    ranges = _validate_line_ranges(
        ctx,
        evidence_location.get("line_ranges"),
        f"{location}.line_ranges",
    )
    excerpt = _require_string(
        ctx, evidence_location.get("excerpt"), f"{location}.excerpt"
    )
    excerpt_hash = _require_sha256(
        ctx,
        evidence_location.get("excerpt_sha256"),
        f"{location}.excerpt_sha256",
    )
    if excerpt is not None and excerpt_hash is not None:
        actual_excerpt_hash = sha256_bytes(excerpt.encode("utf-8"))
        if excerpt_hash != actual_excerpt_hash:
            ctx.error(
                f"{location}.excerpt_sha256",
                f"digest mismatch: declared {excerpt_hash}, "
                f"actual {actual_excerpt_hash}",
            )
    if file_text is not None and ranges:
        canonical = _canonical_excerpt(file_text, ranges)
        if canonical is None:
            ctx.error(
                f"{location}.line_ranges",
                "a line range is outside the referenced file",
            )
        elif excerpt is not None and excerpt != canonical:
            ctx.error(
                f"{location}.excerpt",
                "must exactly equal the referenced lines; multiple ranges "
                "must be joined with '\\n...\\n'",
            )
    if (
        file_text is not None
        and relative is not None
        and relative.suffix.lower() in {".py", ".pyi"}
        and kind in {"function", "method", "class"}
        and qualified_name is not None
    ):
        _validate_python_symbol_location(
            ctx,
            file_text=file_text,
            kind=kind,
            qualified_name=qualified_name,
            ranges=ranges,
            location=f"{location}.symbol",
        )

    flow_step = evidence_location.get("flow_step")
    if flow_step is not None and (
        not _is_int(flow_step) or flow_step < 1
    ):
        ctx.error(
            f"{location}.flow_step",
            "must be null or a positive integer",
        )
        return None
    return flow_step


def _validate_evidence(
    ctx: ValidationContext,
    evidence: dict[str, Any],
    location: str,
    repo_root: Path | None,
    expected_evidence_id: str,
) -> tuple[str | None, str | None, str | None, str | None]:
    """Return ``(evidence_id, type, strength, relation)`` after validation."""

    evidence_keys = {
        "evidence_id",
        "evidence_type",
        "relation",
        "strength",
        "provenance",
        "locations",
        "explanation",
        "confidence",
    }
    _require_exact_keys(ctx, evidence, evidence_keys, location)

    evidence_id = _require_string(
        ctx, evidence.get("evidence_id"), f"{location}.evidence_id"
    )
    if evidence_id is not None and evidence_id != expected_evidence_id:
        ctx.error(
            f"{location}.evidence_id",
            f"must equal {expected_evidence_id!r}",
        )
    evidence_type = _require_enum(
        ctx,
        evidence.get("evidence_type"),
        EVIDENCE_TYPES,
        f"{location}.evidence_type",
    )
    relation = _require_enum(
        ctx,
        evidence.get("relation"),
        EVIDENCE_RELATIONS,
        f"{location}.relation",
    )
    strength = _require_enum(
        ctx,
        evidence.get("strength"),
        EVIDENCE_STRENGTHS,
        f"{location}.strength",
    )
    _require_enum(
        ctx,
        evidence.get("provenance"),
        EVIDENCE_PROVENANCE,
        f"{location}.provenance",
    )
    locations = _require_list(
        ctx, evidence.get("locations"), f"{location}.locations"
    )
    flow_steps: list[int | None] = []
    if locations is not None:
        if not locations:
            ctx.error(f"{location}.locations", "must not be empty")
        for index, location_value in enumerate(locations):
            flow_steps.append(
                _validate_evidence_location(
                    ctx,
                    evidence_type,
                    location_value,
                    f"{location}.locations[{index}]",
                    repo_root,
                )
            )
    if evidence_type == "data_flow":
        if any(step is None for step in flow_steps):
            ctx.error(
                f"{location}.locations",
                "data_flow evidence requires a positive flow_step on every "
                "location",
            )
        concrete_steps = [step for step in flow_steps if step is not None]
        if (
            len(concrete_steps) != len(set(concrete_steps))
            or sorted(concrete_steps)
            != list(range(1, len(concrete_steps) + 1))
        ):
            ctx.error(
                f"{location}.locations",
                "data_flow flow_step values must be unique and contiguous "
                "starting at 1",
            )
    elif any(step is not None for step in flow_steps):
        ctx.error(
            f"{location}.locations",
            "flow_step must be null unless evidence_type is 'data_flow'",
        )
    _require_string(
        ctx, evidence.get("explanation"), f"{location}.explanation"
    )
    _require_enum(
        ctx,
        evidence.get("confidence"),
        EVIDENCE_CONFIDENCE,
        f"{location}.confidence",
    )
    return evidence_id, evidence_type, strength, relation


def _validate_related_tests(
    ctx: ValidationContext,
    value: Any,
    location: str,
    required_test_evidence_ids: set[str],
    eligible_test_related_evidence_ids: set[str],
) -> tuple[str | None, list[str]]:
    tests = _require_object(ctx, value, location)
    if tests is None:
        return None, []
    _require_exact_keys(
        ctx, tests, {"status", "evidence_ids", "search_notes"}, location
    )
    status_value = _require_enum(
        ctx,
        tests.get("status"),
        RELATED_TEST_STATUSES,
        f"{location}.status",
    )
    evidence_ids = _validate_unique_strings(
        ctx,
        tests.get("evidence_ids"),
        f"{location}.evidence_ids",
    )
    _require_string(
        ctx, tests.get("search_notes"), f"{location}.search_notes"
    )
    evidence_ids = evidence_ids or []
    unknown = sorted(
        set(evidence_ids) - eligible_test_related_evidence_ids
    )
    if unknown:
        ctx.error(
            f"{location}.evidence_ids",
            f"references non-test-related or unknown evidence IDs: {unknown}",
        )
    missing = sorted(required_test_evidence_ids - set(evidence_ids))
    if missing:
        ctx.error(
            f"{location}.evidence_ids",
            f"omits test evidence IDs: {missing}",
        )
    if status_value == "found" and not evidence_ids:
        ctx.error(
            location,
            "status 'found' requires at least one referenced test evidence",
        )
    if status_value == "none_found" and (
        evidence_ids or required_test_evidence_ids
    ):
        ctx.error(
            location,
            "status 'none_found' cannot coexist with test evidence",
        )
    return status_value, evidence_ids


def _validate_no_direct_reason(
    ctx: ValidationContext, value: Any, location: str
) -> None:
    reason = _require_object(ctx, value, location)
    if reason is None:
        return
    _require_exact_keys(ctx, reason, {"code", "detail"}, location)
    _require_enum(
        ctx,
        reason.get("code"),
        NO_DIRECT_REASON_CODES,
        f"{location}.code",
    )
    _require_string(ctx, reason.get("detail"), f"{location}.detail")


def _validate_repository_scan(
    ctx: ValidationContext,
    value: Any,
    location: str,
    actual_index: dict[str, int] | None,
) -> dict[str, int] | None:
    repository_scan = _require_object(ctx, value, location)
    if repository_scan is None:
        return None
    _require_exact_keys(
        ctx, repository_scan, REPOSITORY_SCAN_KEYS, location
    )
    result: dict[str, int] = {}
    for key in ("indexed_files", "indexed_snippets", "shards"):
        count = repository_scan.get(key)
        if not _is_int(count) or count < 1:
            ctx.error(
                f"{location}.{key}",
                "must be a positive integer",
            )
        else:
            result[key] = count
    if actual_index is not None:
        for key in ("indexed_files", "indexed_snippets"):
            if result.get(key) != actual_index.get(key):
                ctx.error(
                    f"{location}.{key}",
                    f"must equal the deterministic repository index value "
                    f"{actual_index.get(key)!r}",
                )
    return result


def _validate_code_mapping_methods(
    ctx: ValidationContext,
    value: Any,
    actual_index: dict[str, int] | None,
) -> None:
    location = "code_mappings.methods"
    methods = _require_object(ctx, value, location)
    if methods is None:
        return
    _require_exact_keys(ctx, methods, CODE_MAPPING_METHOD_KEYS, location)

    cbc_location = f"{location}.condition_by_condition"
    condition_by_condition = _require_object(
        ctx, methods.get("condition_by_condition"), cbc_location
    )
    cbc_scan: dict[str, int] | None = None
    if condition_by_condition is not None:
        _require_exact_keys(
            ctx,
            condition_by_condition,
            CONDITION_BY_CONDITION_METHOD_KEYS,
            cbc_location,
        )
        if (
            condition_by_condition.get("mode")
            != "one_condition_full_repository_sharded_alignment"
        ):
            ctx.error(
                f"{cbc_location}.mode",
                "must equal "
                "'one_condition_full_repository_sharded_alignment'",
            )
        if (
            condition_by_condition.get("input_scope")
            != "single_deleted_condition_complete_repository"
        ):
            ctx.error(
                f"{cbc_location}.input_scope",
                "must equal "
                "'single_deleted_condition_complete_repository'",
            )
        cbc_scan = _validate_repository_scan(
            ctx,
            condition_by_condition.get("repository_scan"),
            f"{cbc_location}.repository_scan",
            actual_index,
        )

    holistic_location = f"{location}.holistic_alignment"
    holistic_alignment = _require_object(
        ctx, methods.get("holistic_alignment"), holistic_location
    )
    holistic_scan: dict[str, int] | None = None
    if holistic_alignment is not None:
        _require_exact_keys(
            ctx,
            holistic_alignment,
            HOLISTIC_ALIGNMENT_METHOD_KEYS,
            holistic_location,
        )
        if (
            holistic_alignment.get("mode")
            != "complete_document_repo_wide_alignment"
        ):
            ctx.error(
                f"{holistic_location}.mode",
                "must equal 'complete_document_repo_wide_alignment'",
            )
        if (
            holistic_alignment.get("input_scope")
            != "complete_document_all_selected_conditions_and_repository"
        ):
            ctx.error(
                f"{holistic_location}.input_scope",
                "must equal "
                "'complete_document_all_selected_conditions_and_repository'",
            )
        _require_string(
            ctx,
            holistic_alignment.get("alignment_summary"),
            f"{holistic_location}.alignment_summary",
        )
        holistic_scan = _validate_repository_scan(
            ctx,
            holistic_alignment.get("repository_scan"),
            f"{holistic_location}.repository_scan",
            actual_index,
        )
    if (
        cbc_scan is not None
        and holistic_scan is not None
        and cbc_scan != holistic_scan
    ):
        ctx.error(
            "code_mappings.methods",
            "both routes must report the same deterministic repository scan",
        )


def _validate_manual_comparison(
    ctx: ValidationContext, value: Any, location: str
) -> None:
    comparison = _require_object(ctx, value, location)
    if comparison is None:
        return
    _require_exact_keys(ctx, comparison, MANUAL_COMPARISON_KEYS, location)

    agreement = comparison.get("agreement")
    if agreement is not None:
        _require_enum(
            ctx,
            agreement,
            MANUAL_AGREEMENT_VALUES,
            f"{location}.agreement",
        )
    _require_string(
        ctx,
        comparison.get("notes"),
        f"{location}.notes",
        allow_empty=agreement is None,
    )


def _validate_gold_labels(
    ctx: ValidationContext,
    value: Any,
    comparison_ids: Sequence[str],
) -> None:
    location = "code_mappings.gold_labels"
    labels = _require_list(ctx, value, location)
    if labels is None:
        return

    label_ids: list[str] = []
    for label_index, raw_label in enumerate(labels):
        label_location = f"{location}[{label_index}]"
        label = _require_object(ctx, raw_label, label_location)
        if label is None:
            continue
        _require_exact_keys(ctx, label, GOLD_LABEL_KEYS, label_location)
        condition_id = _require_string(
            ctx,
            label.get("condition_id"),
            f"{label_location}.condition_id",
        )
        if condition_id is not None:
            label_ids.append(condition_id)
        _require_string(
            ctx,
            label.get("mapping_explanation"),
            f"{label_location}.mapping_explanation",
        )
        _require_enum(
            ctx,
            label.get("gold_source"),
            GOLD_SOURCE_VALUES,
            f"{label_location}.gold_source",
        )

    if len(label_ids) != len(set(label_ids)):
        ctx.error(location, "condition_id values must be unique")
    if label_ids != list(comparison_ids):
        ctx.error(
            location,
            "must contain exactly one gold label per comparison item, in "
            "comparison_items condition_id order",
        )


def _validate_code_mapping_branch(
    ctx: ValidationContext,
    value: Any,
    parent_condition_id: str | None,
    location: str,
    repo_root: Path | None,
    evidence_id_infix: str,
    global_evidence_ids: list[str],
) -> None:
    mapping = _require_object(ctx, value, location)
    if mapping is None:
        return
    _require_exact_keys(ctx, mapping, CODE_MAPPING_KEYS, location)

    condition_id = _require_string(
        ctx, mapping.get("condition_id"), f"{location}.condition_id"
    )
    if condition_id is not None and condition_id != parent_condition_id:
        ctx.error(
            f"{location}.condition_id",
            "must exactly match the parent comparison item condition_id",
        )
    status_value = _require_enum(
        ctx,
        mapping.get("mapping_status"),
        MAPPING_STATUSES,
        f"{location}.mapping_status",
    )
    coverage = _require_enum(
        ctx,
        mapping.get("coverage"),
        COVERAGE_VALUES,
        f"{location}.coverage",
    )
    evidence_basis = _validate_unique_strings(
        ctx,
        mapping.get("evidence_basis"),
        f"{location}.evidence_basis",
        allowed=EVIDENCE_TYPES,
    )
    _require_string(
        ctx,
        mapping.get("mapping_explanation"),
        f"{location}.mapping_explanation",
    )
    reason = mapping.get("no_direct_mapping_reason")

    raw_evidence = _require_list(
        ctx, mapping.get("evidence"), f"{location}.evidence"
    )
    evidence_types: list[str] = []
    positive_implementation_direct = False
    has_contradiction = False
    required_test_evidence_ids: set[str] = set()
    eligible_test_related_evidence_ids: set[str] = set()
    if raw_evidence is not None:
        for evidence_index, raw_item in enumerate(raw_evidence):
            evidence_location = f"{location}.evidence[{evidence_index}]"
            item = _require_object(ctx, raw_item, evidence_location)
            if item is None:
                continue
            id_condition = parent_condition_id or condition_id
            evidence_id, evidence_type, strength, relation = _validate_evidence(
                ctx,
                item,
                evidence_location,
                repo_root,
                (
                    f"{id_condition}_{evidence_id_infix}"
                    f"_ev{evidence_index + 1:03d}"
                    if id_condition is not None
                    else ""
                ),
            )
            if evidence_id is not None:
                global_evidence_ids.append(evidence_id)
            if evidence_type is not None:
                evidence_types.append(evidence_type)
            if evidence_id is not None:
                if evidence_type in {"test", "fixture"}:
                    required_test_evidence_ids.add(evidence_id)
                if evidence_type in {
                    "test",
                    "fixture",
                    "build_metadata",
                    "configuration",
                    "documentation",
                }:
                    eligible_test_related_evidence_ids.add(evidence_id)
            if (
                evidence_type == "implementation"
                and strength == "direct"
                and relation != "contradicts"
            ):
                positive_implementation_direct = True
            if relation == "contradicts":
                has_contradiction = True

    expected_basis = list(dict.fromkeys(evidence_types))
    if evidence_basis is not None and evidence_basis != expected_basis:
        ctx.error(
            f"{location}.evidence_basis",
            "must exactly list first-occurrence evidence_type values "
            "present in evidence",
        )

    if status_value == "direct":
        if coverage not in {"full", "partial"}:
            ctx.error(
                f"{location}.coverage",
                "direct mapping requires full or partial coverage",
            )
        if reason is not None:
            ctx.error(
                f"{location}.no_direct_mapping_reason",
                "direct mapping requires null",
            )
        if not positive_implementation_direct:
            ctx.error(
                location,
                "direct mapping requires positive implementation evidence "
                "whose strength is 'direct'",
            )
        if has_contradiction and coverage == "full":
            ctx.error(
                f"{location}.coverage",
                "a mapping containing contradicting evidence cannot claim "
                "full coverage",
            )
    elif status_value == "no_direct_mapping":
        reason_location = f"{location}.no_direct_mapping_reason"
        if reason is None:
            ctx.error(
                reason_location,
                "no_direct_mapping requires a non-null reason object",
            )
        else:
            _validate_no_direct_reason(ctx, reason, reason_location)
        if positive_implementation_direct:
            ctx.error(
                location,
                "no_direct_mapping cannot contain direct implementation "
                "evidence",
            )
        if not evidence_types and coverage != "none":
            ctx.error(
                f"{location}.coverage",
                "a mapping without evidence requires coverage 'none'",
            )

    # Test references are intentionally resolved against this branch only.
    _validate_related_tests(
        ctx,
        mapping.get("related_tests"),
        f"{location}.related_tests",
        required_test_evidence_ids,
        eligible_test_related_evidence_ids,
    )


def _validate_code_mappings(
    ctx: ValidationContext,
    value: Any,
    condition_ids: Sequence[str],
    repo_root: Path | None,
) -> None:
    bundle = _require_object(ctx, value, "code_mappings")
    if bundle is None:
        return
    expected_bundle_keys = CODE_MAPPINGS_BUNDLE_REQUIRED_KEYS | (
        CODE_MAPPINGS_BUNDLE_OPTIONAL_KEYS if "gold_labels" in bundle else set()
    )
    _require_exact_keys(ctx, bundle, expected_bundle_keys, "code_mappings")
    actual_index: dict[str, int] | None = None
    if repo_root is not None:
        try:
            snippets = extract_repository_snippets(repo_root)
        except Exception as exc:  # noqa: BLE001 - report deterministic index failure.
            ctx.error(
                "code_mappings.methods",
                f"cannot rebuild deterministic repository index: {exc}",
            )
        else:
            actual_index = {
                "indexed_files": len(
                    {snippet.file_path for snippet in snippets}
                ),
                "indexed_snippets": len(snippets),
            }
    _validate_code_mapping_methods(
        ctx, bundle.get("methods"), actual_index
    )

    comparisons = _require_list(
        ctx,
        bundle.get("comparison_items"),
        "code_mappings.comparison_items",
    )
    if comparisons is None:
        if "gold_labels" in bundle:
            _validate_gold_labels(ctx, bundle.get("gold_labels"), [])
        return

    comparison_ids: list[str] = []
    global_evidence_ids: list[str] = []
    for comparison_index, raw_comparison in enumerate(comparisons):
        location = f"code_mappings.comparison_items[{comparison_index}]"
        comparison = _require_object(ctx, raw_comparison, location)
        if comparison is None:
            continue
        _require_exact_keys(
            ctx, comparison, COMPARISON_ITEM_KEYS, location
        )
        condition_id = _require_string(
            ctx,
            comparison.get("condition_id"),
            f"{location}.condition_id",
        )
        if condition_id is not None:
            comparison_ids.append(condition_id)
        _validate_code_mapping_branch(
            ctx,
            comparison.get("condition_by_condition"),
            condition_id,
            f"{location}.condition_by_condition",
            repo_root,
            "cbc",
            global_evidence_ids,
        )
        _validate_code_mapping_branch(
            ctx,
            comparison.get("holistic_alignment"),
            condition_id,
            f"{location}.holistic_alignment",
            repo_root,
            "ha",
            global_evidence_ids,
        )
        _validate_manual_comparison(
            ctx,
            comparison.get("manual_comparison"),
            f"{location}.manual_comparison",
        )

    if len(comparison_ids) != len(set(comparison_ids)):
        ctx.error(
            "code_mappings.comparison_items",
            "condition_id values must be unique",
        )
    if comparison_ids != list(condition_ids):
        ctx.error(
            "code_mappings.comparison_items",
            "must contain exactly one comparison per selected condition, in "
            "selected_condition_ids order",
        )
    if "gold_labels" in bundle:
        _validate_gold_labels(
            ctx,
            bundle.get("gold_labels"),
            comparison_ids,
        )
    if len(global_evidence_ids) != len(set(global_evidence_ids)):
        ctx.error(
            "code_mappings",
            "evidence_id values must be globally unique across both methods",
        )


def _validate_source(ctx: ValidationContext, value: Any) -> None:
    source = _require_object(ctx, value, "source")
    if source is None:
        return
    _require_exact_keys(ctx, source, SOURCE_KEYS, "source")
    _require_string(ctx, source.get("dataset"), "source.dataset")
    _require_nullable_string(
        ctx, source.get("dataset_file"), "source.dataset_file"
    )
    _require_string(ctx, source.get("instance_id"), "source.instance_id")
    difficulty = source.get("difficulty")
    if difficulty is not None and not isinstance(difficulty, (str, int, float)):
        ctx.error(
            "source.difficulty",
            "must be a string, number, or null",
        )
    if isinstance(difficulty, bool):
        ctx.error(
            "source.difficulty",
            "must not be a boolean",
        )
    _require_nullable_string(ctx, source.get("pypi_name"), "source.pypi_name")


def _validate_artifacts(ctx: ValidationContext, value: Any) -> None:
    artifacts = _require_object(ctx, value, "artifacts")
    if artifacts is None:
        return
    _require_exact_keys(ctx, artifacts, ARTIFACT_KEYS, "artifacts")
    expected = {
        "document_before": DOCUMENT_BEFORE_FILE,
        "deleted_parts": DELETED_PARTS_FILE,
        "document_after": DOCUMENT_AFTER_FILE,
        "code_mapping": CODE_MAPPING_FILE,
        "original_repo": ORIGINAL_REPO_DIR,
    }
    for key, expected_path in expected.items():
        relative = _safe_relative_path(
            ctx, artifacts.get(key), f"artifacts.{key}"
        )
        if relative is not None and relative.as_posix() != expected_path:
            ctx.error(
                f"artifacts.{key}", f"must equal {expected_path!r}"
            )


def _validate_github_url(
    ctx: ValidationContext, value: Any, owner: Any, name: Any
) -> None:
    url = _require_string(ctx, value, "original_repo.github_url")
    owner_string = _require_string(ctx, owner, "original_repo.owner")
    name_string = _require_string(ctx, name, "original_repo.name")
    if url is None:
        return
    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        ctx.error("original_repo.github_url", f"invalid URL: {exc}")
        return
    parts = [part for part in parsed.path.split("/") if part]
    if (
        parsed.scheme != "https"
        or hostname != "github.com"
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
        or parsed.query
        or parsed.fragment
        or len(parts) != 2
    ):
        ctx.error(
            "original_repo.github_url",
            "must be a canonical https://github.com/<owner>/<repo> URL",
        )
        return
    url_name = parts[1][:-4] if parts[1].endswith(".git") else parts[1]
    if owner_string is not None and parts[0] != owner_string:
        ctx.error(
            "original_repo.owner",
            "does not match github_url owner",
        )
    if name_string is not None and url_name != name_string:
        ctx.error(
            "original_repo.name",
            "does not match github_url repository name",
        )


def _validate_evaluation(
    ctx: ValidationContext, value: Any, repo_root: Path | None
) -> None:
    evaluation = _require_object(ctx, value, "original_repo.evaluation")
    if evaluation is None:
        return
    _require_exact_keys(
        ctx, evaluation, EVALUATION_KEYS, "original_repo.evaluation"
    )
    _require_nullable_string(
        ctx,
        evaluation.get("image_url"),
        "original_repo.evaluation.image_url",
    )
    _require_nullable_string(
        ctx,
        evaluation.get("workdir"),
        "original_repo.evaluation.workdir",
    )

    test_patch = evaluation.get("test_patch")
    if test_patch is not None:
        patch = _require_object(
            ctx, test_patch, "original_repo.evaluation.test_patch"
        )
        if patch is not None:
            _require_exact_keys(
                ctx,
                patch,
                TEST_PATCH_KEYS,
                "original_repo.evaluation.test_patch",
            )
            relative = _safe_relative_path(
                ctx,
                patch.get("path"),
                "original_repo.evaluation.test_patch.path",
            )
            if (
                relative is not None
                and relative.as_posix() != "evaluation_tests.patch"
            ):
                ctx.error(
                    "original_repo.evaluation.test_patch.path",
                    "must equal 'evaluation_tests.patch'",
                )
            patch_path = (
                _resolve_under(
                    ctx,
                    ctx.sample_dir,
                    relative,
                    "original_repo.evaluation.test_patch.path",
                )
                if relative is not None
                else None
            )
            content = _read_bytes(
                ctx,
                patch_path,
                "original_repo.evaluation.test_patch.path",
            )
            declared_hash = _require_sha256(
                ctx,
                patch.get("sha256"),
                "original_repo.evaluation.test_patch.sha256",
            )
            if content is not None and declared_hash is not None:
                actual_hash = sha256_bytes(content)
                if declared_hash != actual_hash:
                    ctx.error(
                        "original_repo.evaluation.test_patch.sha256",
                        f"digest mismatch: declared {declared_hash}, "
                        f"actual {actual_hash}",
                    )

    test_files = _validate_unique_strings(
        ctx,
        evaluation.get("test_files"),
        "original_repo.evaluation.test_files",
    )
    for index, path_string in enumerate(test_files or []):
        relative = _safe_relative_path(
            ctx,
            path_string,
            f"original_repo.evaluation.test_files[{index}]",
        )
        if relative is not None and ".git" in relative.parts:
            ctx.error(
                f"original_repo.evaluation.test_files[{index}]",
                "must not reference repository metadata under .git",
            )
    _validate_unique_strings(
        ctx,
        evaluation.get("passed_test_ids"),
        "original_repo.evaluation.passed_test_ids",
    )


def _validate_original_repo(
    ctx: ValidationContext, value: Any
) -> Path | None:
    original_repo = _require_object(ctx, value, "original_repo")
    if original_repo is None:
        return None
    _require_exact_keys(
        ctx, original_repo, ORIGINAL_REPO_KEYS, "original_repo"
    )
    _validate_github_url(
        ctx,
        original_repo.get("github_url"),
        original_repo.get("owner"),
        original_repo.get("name"),
    )

    revision = _require_object(
        ctx, original_repo.get("revision"), "original_repo.revision"
    )
    if revision is not None:
        _require_exact_keys(
            ctx, revision, REVISION_KEYS, "original_repo.revision"
        )
        if revision.get("kind") != "commit":
            ctx.error(
                "original_repo.revision.kind", "must equal 'commit'"
            )
        commit = _require_string(
            ctx,
            revision.get("commit"),
            "original_repo.revision.commit",
        )
        if commit is not None and not COMMIT_RE.fullmatch(commit):
            ctx.error(
                "original_repo.revision.commit",
                "must be a full lowercase 40- or 64-character commit hash",
            )

    repo_root: Path | None = None
    snapshot = _require_object(
        ctx, original_repo.get("snapshot"), "original_repo.snapshot"
    )
    if snapshot is not None:
        _require_exact_keys(
            ctx, snapshot, SNAPSHOT_KEYS, "original_repo.snapshot"
        )
        relative = _safe_relative_path(
            ctx, snapshot.get("path"), "original_repo.snapshot.path"
        )
        if (
            relative is not None
            and relative.as_posix() != ORIGINAL_REPO_DIR
        ):
            ctx.error(
                "original_repo.snapshot.path",
                f"must equal {ORIGINAL_REPO_DIR!r}",
            )
        repo_root = (
            _resolve_under(
                ctx,
                ctx.sample_dir,
                relative,
                "original_repo.snapshot.path",
            )
            if relative is not None
            else None
        )
        if (
            repo_root is not None
            and (not repo_root.is_dir() or repo_root.is_symlink())
        ):
            ctx.error(
                "original_repo.snapshot.path",
                "must point to a real directory",
            )
            repo_root = None
        if snapshot.get("pristine") is not True:
            ctx.error(
                "original_repo.snapshot.pristine", "must be true"
            )
        declared_tree_hash = _require_sha256(
            ctx,
            snapshot.get("tree_sha256"),
            "original_repo.snapshot.tree_sha256",
        )
        if repo_root is not None:
            try:
                actual_tree_hash = tree_sha256(repo_root)
            except (OSError, ValueError) as exc:
                ctx.error(
                    "original_repo.snapshot.path",
                    f"invalid repository tree: {exc}",
                )
            else:
                if (
                    declared_tree_hash is not None
                    and declared_tree_hash != actual_tree_hash
                ):
                    ctx.error(
                        "original_repo.snapshot.tree_sha256",
                        f"digest mismatch: declared {declared_tree_hash}, "
                        f"actual {actual_tree_hash}",
                    )

    license_value = original_repo.get("license_spdx_id")
    if license_value is not None:
        _require_string(
            ctx, license_value, "original_repo.license_spdx_id"
        )
    _validate_evaluation(ctx, original_repo.get("evaluation"), repo_root)
    return repo_root


def _validate_derivation(ctx: ValidationContext, value: Any) -> None:
    derivation = _require_object(ctx, value, "derivation")
    if derivation is None:
        return
    _require_exact_keys(ctx, derivation, DERIVATION_KEYS, "derivation")
    if derivation.get("operation") != "delete_only":
        ctx.error("derivation.operation", "must equal 'delete_only'")
    if derivation.get("transform_version") != "2.0":
        ctx.error(
            "derivation.transform_version", "must equal '2.0'"
        )
    if derivation.get("document_truncated") is not False:
        ctx.error(
            "derivation.document_truncated",
            "must be false; truncated source documents are invalid",
        )

    relative = _safe_relative_path(
        ctx,
        derivation.get("condition_selection_path"),
        "derivation.condition_selection_path",
    )
    if (
        relative is not None
        and relative.as_posix() != "condition_selection.json"
    ):
        ctx.error(
            "derivation.condition_selection_path",
            "must equal 'condition_selection.json'",
        )
    selection_path = (
        _resolve_under(
            ctx,
            ctx.sample_dir,
            relative,
            "derivation.condition_selection_path",
        )
        if relative is not None
        else None
    )
    selection_content = _read_bytes(
        ctx, selection_path, "derivation.condition_selection_path"
    )
    declared_hash = _require_sha256(
        ctx,
        derivation.get("condition_selection_sha256"),
        "derivation.condition_selection_sha256",
    )
    if selection_content is not None and declared_hash is not None:
        actual_hash = sha256_bytes(selection_content)
        if declared_hash != actual_hash:
            ctx.error(
                "derivation.condition_selection_sha256",
                f"digest mismatch: declared {declared_hash}, "
                f"actual {actual_hash}",
            )


def _validate_annotation(ctx: ValidationContext, value: Any) -> None:
    annotation = _require_object(ctx, value, "annotation")
    if annotation is None:
        return
    _require_exact_keys(ctx, annotation, ANNOTATION_KEYS, "annotation")
    _require_enum(
        ctx,
        annotation.get("status"),
        REVIEW_STATUSES,
        "annotation.status",
    )
    _require_string(ctx, annotation.get("generator"), "annotation.generator")
    _require_nullable_string(ctx, annotation.get("model"), "annotation.model")
    generated_at = _require_string(
        ctx, annotation.get("generated_at"), "annotation.generated_at"
    )
    if generated_at is not None:
        try:
            parsed = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
        except ValueError:
            ctx.error(
                "annotation.generated_at",
                "must be an ISO-8601 timestamp",
            )
        else:
            if parsed.tzinfo is None:
                ctx.error(
                    "annotation.generated_at",
                    "must include a timezone",
                )
    _require_string(
        ctx, annotation.get("notes"), "annotation.notes", allow_empty=True
    )


def validate_sample(sample_dir: Path | str) -> list[str]:
    """Validate one sample directory and return all human-readable errors."""

    directory = Path(sample_dir)
    ctx = ValidationContext(sample_dir=directory)
    if not directory.is_dir() or directory.is_symlink():
        return [f"sample: not a real directory: {directory}"]

    sample_id = directory.name
    if not SAMPLE_ID_RE.fullmatch(sample_id) or len(sample_id) > 180:
        ctx.error("sample_id", "containing directory is not a portable sample ID")

    before_bytes = _read_bytes(
        ctx,
        directory / DOCUMENT_BEFORE_FILE,
        DOCUMENT_BEFORE_FILE,
    )
    after_bytes = _read_bytes(
        ctx,
        directory / DOCUMENT_AFTER_FILE,
        DOCUMENT_AFTER_FILE,
    )
    document_before = None
    document_after = None
    if before_bytes is not None:
        try:
            document_before = before_bytes.decode("utf-8")
        except UnicodeDecodeError:
            ctx.error(DOCUMENT_BEFORE_FILE, "must be valid UTF-8")
    if after_bytes is not None:
        try:
            document_after = after_bytes.decode("utf-8")
        except UnicodeDecodeError:
            ctx.error(DOCUMENT_AFTER_FILE, "must be valid UTF-8")

    deleted_parts = _read_projection(
        ctx,
        DELETED_PARTS_FILE,
        "deleted_parts",
        sample_id,
    )
    code_mappings = _read_projection(
        ctx,
        CODE_MAPPING_FILE,
        "code_mappings",
        sample_id,
    )
    selected_ids = _read_condition_selection(ctx, sample_id)
    condition_ids, spans = _validate_deleted_parts(
        ctx, deleted_parts, document_before
    )
    if selected_ids is not None:
        if not 3 <= len(selected_ids) <= 5:
            ctx.error(
                "selected_condition_ids",
                "must contain exactly 3 to 5 IDs",
            )
        if selected_ids != condition_ids:
            ctx.error(
                "selected_condition_ids",
                "must exactly match deleted_parts condition_id order",
            )

    if document_before is not None and document_after is not None:
        rebuilt = _rebuild_after(document_before, spans)
        if rebuilt != document_after:
            ctx.error(
                "document_after",
                "is not the exact pure-deletion reconstruction from "
                "document_before and deleted_spans",
            )
        if len(document_after) >= len(document_before):
            ctx.error(
                "document_after",
                "must be shorter than document_before",
            )

    repo_root = directory / ORIGINAL_REPO_DIR
    if not repo_root.is_dir() or repo_root.is_symlink():
        ctx.error(ORIGINAL_REPO_DIR, "must be a real repository directory")
        repo_root = None
    _validate_code_mappings(
        ctx,
        code_mappings,
        selected_ids or condition_ids,
        repo_root,
    )
    return ctx.errors


def discover_samples(data_dir: Path | str) -> list[Path]:
    """Return immediate sample directories in deterministic name order."""

    root = Path(data_dir)
    if (root / DELETED_PARTS_FILE).is_file():
        return [root]
    if not root.is_dir():
        return []
    return sorted(
        (
            child
            for child in root.iterdir()
            if child.is_dir() and not child.is_symlink()
        ),
        key=lambda path: path.name,
    )


def validate_collection_manifest(
    data_dir: Path | str, samples: Sequence[Path]
) -> list[str]:
    """Require the qualified manifest to exactly match its sample directories."""

    root = Path(data_dir)
    ctx = ValidationContext(sample_dir=root)
    manifest = _read_json_object(
        ctx,
        root / COLLECTION_MANIFEST_FILE,
        COLLECTION_MANIFEST_FILE,
    )
    if manifest is None:
        return ctx.errors
    _require_exact_keys(
        ctx,
        manifest,
        COLLECTION_MANIFEST_KEYS,
        COLLECTION_MANIFEST_FILE,
    )
    if manifest.get("schema_version") != COLLECTION_SCHEMA_VERSION:
        ctx.error(
            f"{COLLECTION_MANIFEST_FILE}.schema_version",
            f"must equal {COLLECTION_SCHEMA_VERSION!r}",
        )
    if manifest.get("collection") != root.name:
        ctx.error(
            f"{COLLECTION_MANIFEST_FILE}.collection",
            f"must equal containing directory name {root.name!r}",
        )
    if manifest.get("release_eligible") is not True:
        ctx.error(
            f"{COLLECTION_MANIFEST_FILE}.release_eligible",
            "must be true for the qualified collection",
        )
    updated_at = _require_string(
        ctx,
        manifest.get("updated_at"),
        f"{COLLECTION_MANIFEST_FILE}.updated_at",
    )
    if updated_at is not None:
        try:
            parsed_date = datetime.strptime(updated_at, "%Y-%m-%d")
        except ValueError:
            ctx.error(
                f"{COLLECTION_MANIFEST_FILE}.updated_at",
                "must use YYYY-MM-DD format",
            )
        else:
            if parsed_date.strftime("%Y-%m-%d") != updated_at:
                ctx.error(
                    f"{COLLECTION_MANIFEST_FILE}.updated_at",
                    "must use canonical YYYY-MM-DD format",
                )

    entries = _require_list(
        ctx,
        manifest.get("samples"),
        f"{COLLECTION_MANIFEST_FILE}.samples",
    )
    if entries is None:
        return ctx.errors

    actual_by_name = {sample.name: sample for sample in samples}
    declared_paths: list[str] = []
    declared_ids: list[str] = []
    for index, raw_entry in enumerate(entries):
        location = f"{COLLECTION_MANIFEST_FILE}.samples[{index}]"
        entry = _require_object(ctx, raw_entry, location)
        if entry is None:
            continue
        _require_exact_keys(ctx, entry, COLLECTION_SAMPLE_KEYS, location)
        sample_id = _require_string(
            ctx, entry.get("sample_id"), f"{location}.sample_id"
        )
        relative = _safe_relative_path(
            ctx, entry.get("path"), f"{location}.path"
        )
        path_value = relative.as_posix() if relative is not None else None
        if relative is not None and len(relative.parts) != 1:
            ctx.error(f"{location}.path", "must name an immediate child directory")
        if sample_id is not None:
            declared_ids.append(sample_id)
        if path_value is not None:
            declared_paths.append(path_value)
        if (
            sample_id is not None
            and path_value is not None
            and sample_id != path_value
        ):
            ctx.error(
                location,
                "sample_id and path must be identical",
            )
        if entry.get("annotation_status") != "approved":
            ctx.error(
                f"{location}.annotation_status",
                "must equal 'approved'",
            )
        if entry.get("mechanical_validation") != "passed":
            ctx.error(
                f"{location}.mechanical_validation",
                "must equal 'passed'",
            )
        sample_path = actual_by_name.get(path_value or "")
        if sample_path is None:
            continue
        github_url = _require_string(
            ctx, entry.get("github_url"), f"{location}.github_url"
        )
        if github_url is not None:
            parsed_url = urlsplit(github_url)
            url_parts = [part for part in parsed_url.path.split("/") if part]
            if (
                parsed_url.scheme != "https"
                or parsed_url.hostname != "github.com"
                or parsed_url.query
                or parsed_url.fragment
                or len(url_parts) != 2
            ):
                ctx.error(
                    f"{location}.github_url",
                    "must be a canonical https://github.com/<owner>/<repo> URL",
                )
        commit = _require_string(
            ctx,
            entry.get("parent_commit"),
            f"{location}.parent_commit",
        )
        if commit is not None and not COMMIT_RE.fullmatch(commit):
            ctx.error(
                f"{location}.parent_commit",
                "must be a full lowercase 40- or 64-character commit hash",
            )
        _require_string(ctx, entry.get("generator"), f"{location}.generator")
        _require_nullable_string(ctx, entry.get("model"), f"{location}.model")
        generated_at = _require_string(
            ctx, entry.get("generated_at"), f"{location}.generated_at"
        )
        if generated_at is not None:
            try:
                parsed = datetime.fromisoformat(
                    generated_at.replace("Z", "+00:00")
                )
            except ValueError:
                ctx.error(
                    f"{location}.generated_at",
                    "must be an ISO-8601 timestamp",
                )
            else:
                if parsed.tzinfo is None:
                    ctx.error(
                        f"{location}.generated_at",
                        "must include a timezone",
                    )

    if len(declared_paths) != len(set(declared_paths)):
        ctx.error(f"{COLLECTION_MANIFEST_FILE}.samples", "contains duplicate paths")
    if len(declared_ids) != len(set(declared_ids)):
        ctx.error(
            f"{COLLECTION_MANIFEST_FILE}.samples",
            "contains duplicate sample_ids",
        )
    actual_names = set(actual_by_name)
    declared_names = set(declared_paths)
    missing = sorted(actual_names - declared_names)
    extra = sorted(declared_names - actual_names)
    if missing:
        ctx.error(
            f"{COLLECTION_MANIFEST_FILE}.samples",
            f"omits sample directories: {missing}",
        )
    if extra:
        ctx.error(
            f"{COLLECTION_MANIFEST_FILE}.samples",
            f"references missing sample directories: {extra}",
        )
    return ctx.errors


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strictly validate SpecGAP 2.2 sample artifacts."
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--data-dir",
        type=Path,
        help="Directory whose immediate children are SpecGAP samples.",
    )
    target.add_argument(
        "--sample",
        type=Path,
        help="One sample directory.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.sample is not None:
        samples = [args.sample]
        manifest_errors: list[str] = []
    else:
        samples = discover_samples(args.data_dir)
        if not samples:
            print(
                f"[error] no sample directories found under {args.data_dir}",
                file=sys.stderr,
            )
            return 2
        manifest_errors = validate_collection_manifest(args.data_dir, samples)

    failure_count = 0
    if manifest_errors:
        failure_count += 1
        print(f"[error] {Path(args.data_dir) / COLLECTION_MANIFEST_FILE}")
        for error in manifest_errors:
            print(f"  - {error}")
    for sample_dir in samples:
        errors = validate_sample(sample_dir)
        if errors:
            failure_count += 1
            print(f"[error] {sample_dir}")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"[ok] {sample_dir}")

    if failure_count:
        print(
            f"Validation failed: {failure_count} validation target(s) invalid.",
            file=sys.stderr,
        )
        return 1
    print(f"Validation passed: {len(samples)} sample(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
