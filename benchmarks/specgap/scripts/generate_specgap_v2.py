#!/usr/bin/env python3
"""Generate repository-grounded SpecGAP 2.2 samples.

The legacy generator created a three-file candidate bundle and used fuzzy text
matching plus implicit cleanup.  This pipeline deliberately has a narrower
contract:

* select 4--5 diverse conditions from a complete, untruncated document;
* resolve every deletion to an exact, unique character span;
* derive the after-document by deletion only;
* pin and materialize the upstream reference repository at ``parent_commit``;
* map every deleted condition twice, once in an isolated per-condition pass
  and once in an independent document/repository-wide alignment pass;
* retain both mappings side by side for an owner to compare without allowing
  either AI pass to see the other pass's output;
* publish a sample only after all mechanical checks pass.

Generated samples use ``approved`` as their only valid status after the
semantic audit and mechanical validation succeed.  They are published directly
to the qualified collection, whose manifest is rebuilt after every run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Sequence

if __package__:
    from .llm_source import (
        BatchAbortError,
        FatalAPIError,
        call_chat_completion,
        get_api_key,
        get_base_url,
        get_model,
        iter_remote_jsonl_rows,
        load_env_file,
        parse_json_object,
    )
    from .repo_evidence import (
        download_github_snapshot,
        extract_repository_snippets,
        format_snippets_for_llm,
        parse_github_repo_url,
        sha256_file,
        tree_sha256,
    )
    from .validate_specgap_v2 import _python_symbol_ranges, validate_sample
else:
    from llm_source import (  # type: ignore
        BatchAbortError,
        FatalAPIError,
        call_chat_completion,
        get_api_key,
        get_base_url,
        get_model,
        iter_remote_jsonl_rows,
        load_env_file,
        parse_json_object,
    )
    from repo_evidence import (  # type: ignore
        download_github_snapshot,
        extract_repository_snippets,
        format_snippets_for_llm,
        parse_github_repo_url,
        sha256_file,
        tree_sha256,
    )
    from validate_specgap_v2 import (  # type: ignore
        _python_symbol_ranges,
        validate_sample,
    )


SCHEMA_VERSION = "specgap-2.2"
COLLECTION_SCHEMA_VERSION = "specgap-collection-1.0"
GENERATOR_VERSION = "2.7.3"
DEFAULT_DATASET = "AweAI-Team/DeNovoSWE"
DEFAULT_DATASET_FILE = "denovoswe_public.jsonl"
DEFAULT_SAMPLE_WORKERS = 10

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
SA_GAPS = {"perception", "comprehension", "projection"}
MAPPING_STATUSES = {"direct", "no_direct_mapping"}
COVERAGE_VALUES = {"full", "partial", "none"}
EVIDENCE_TYPES = {
    "implementation",
    "test",
    "configuration",
    "data_flow",
    "fixture",
    "build_metadata",
    "documentation",
}
RELATED_TEST_EVIDENCE_TYPES = {
    "test",
    "fixture",
    "documentation",
    "build_metadata",
    "configuration",
}
RELATIONS = {
    "implements",
    "enforces",
    "validates",
    "configures",
    "routes",
    "constrains",
    "demonstrates",
    "contradicts",
}
STRENGTH_VALUES = {"direct", "indirect"}
CONFIDENCE_VALUES = {"high", "medium", "low"}
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
MIN_GENERATED_SELECTED_CONDITIONS = 4
MIN_GENERATED_CONDITION_TYPES = 3


class CandidateError(ValueError):
    """A model candidate violated the SpecGAP 2.2 contract."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate strict, repository-grounded SpecGAP 2.2 samples."
    )
    parser.add_argument("--source", choices=["hf", "local"], default="hf")
    parser.add_argument("--input", default=None, help="Local JSON or JSONL input.")
    parser.add_argument(
        "--condition-seed-dir",
        default=None,
        help=(
            "Optional directory of preselected <sample_id>.json condition payloads. "
            "A seed replaces only the first LLM condition draft; it still must pass "
            "all exact-span and semantic audits."
        ),
    )
    parser.add_argument("--hf-dataset", default=DEFAULT_DATASET)
    parser.add_argument("--hf-filename", default=DEFAULT_DATASET_FILE)
    parser.add_argument("--hf-jsonl-url", default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument(
        "--target-successes",
        type=int,
        default=None,
        help=(
            "Stop after this many candidates are successfully published. "
            "--limit remains the maximum number of source candidates inspected."
        ),
    )
    parser.add_argument("--output-dir", default="SpecGAP")
    parser.add_argument("--work-dir", default="runs/current")
    parser.add_argument(
        "--provider",
        choices=["deepseek", "glm", "openai_compatible"],
        default="deepseek",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--thinking",
        choices=["enabled", "disabled"],
        default=None,
        help=(
            "Explicit DeepSeek thinking-mode switch. When enabled, temperature "
            "is omitted because DeepSeek ignores sampling controls in thinking mode."
        ),
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["high", "max"],
        default=None,
        help="DeepSeek reasoning effort; use max for complex agent-style tasks.",
    )
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Maximum condition/mapping proposal attempts per sample.",
    )
    parser.add_argument(
        "--source-retries",
        type=int,
        default=10,
        help="Retries for transient Hugging Face range-download failures.",
    )
    parser.add_argument(
        "--source-timeout",
        type=int,
        default=120,
        help=(
            "Timeout in seconds for each source-data request. This is separate "
            "from --timeout, which applies to model API calls."
        ),
    )
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_SAMPLE_WORKERS,
        help=(
            "Independent samples processed concurrently "
            f"(default: {DEFAULT_SAMPLE_WORKERS}; use cautiously with API limits)."
        ),
    )
    parser.add_argument(
        "--holistic-shard-chars",
        type=int,
        default=60_000,
        help=(
            "Maximum formatted repository characters in each independent "
            "holistic scan shard."
        ),
    )
    parser.add_argument(
        "--max-holistic-aggregate-chars",
        type=int,
        default=480_000,
        help=(
            "Hard character limit for the complete holistic aggregation "
            "prompt. Generation fails explicitly instead of truncating."
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--skip-recorded-failures",
        action="store_true",
        help=(
            "With --resume, treat existing run.json failures as already inspected "
            "instead of paying to retry them. Existing validated completions count "
            "toward --target-successes."
        ),
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Replace only the exact colliding sample directory.",
    )
    parser.add_argument(
        "--keep-failed-stage",
        action="store_true",
        help="Keep failed staging bundles under the work directory.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_json_atomic(path: Path, value: Any) -> None:
    """Atomically replace a JSON control file in its destination directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def rebuild_collection_manifest(
    output_dir: Path,
    sample_records: dict[str, dict[str, Any]] | None = None,
) -> Path:
    """Rebuild the qualified collection manifest from validated sample dirs."""

    existing_records: dict[str, dict[str, Any]] = {}
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file():
        try:
            existing_manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            existing_records = {
                str(entry["sample_id"]): entry
                for entry in existing_manifest.get("samples", [])
                if isinstance(entry, dict) and entry.get("sample_id")
            }
        except (OSError, ValueError, TypeError):
            existing_records = {}
    supplied_records = sample_records or {}
    samples: list[dict[str, str]] = []
    for sample_dir in sorted(output_dir.iterdir(), key=lambda path: path.name):
        if sample_dir.name.startswith(".") or not sample_dir.is_dir():
            continue
        if sample_dir.is_symlink():
            raise CandidateError(
                f"collection sample cannot be a symlink: {sample_dir}"
            )
        if not (sample_dir / "2_deleted_parts.json").is_file():
            raise CandidateError(
                f"unexpected non-sample directory in qualified collection: "
                f"{sample_dir}"
            )
        errors = validate_sample(sample_dir)
        if errors:
            raise CandidateError(
                f"cannot register invalid sample {sample_dir.name}:\n- "
                + "\n- ".join(errors)
            )
        record = supplied_records.get(sample_dir.name) or existing_records.get(
            sample_dir.name
        )
        if not isinstance(record, dict):
            raise CandidateError(
                f"missing collection provenance for {sample_dir.name}"
            )
        samples.append(
            {
                "sample_id": sample_dir.name,
                "path": sample_dir.name,
                "annotation_status": "approved",
                "mechanical_validation": "passed",
                "github_url": str(record["github_url"]),
                "parent_commit": str(record["parent_commit"]).lower(),
                "generator": str(
                    record.get("generator")
                    or f"generate_specgap_v2.py/{GENERATOR_VERSION}"
                ),
                "model": record.get("model"),
                "generated_at": str(record.get("generated_at") or utc_now()),
            }
        )

    write_json_atomic(
        manifest_path,
        {
            "schema_version": COLLECTION_SCHEMA_VERSION,
            "collection": output_dir.name,
            "release_eligible": True,
            "updated_at": datetime.now(timezone.utc).date().isoformat(),
            "samples": samples,
        },
    )
    return manifest_path


def safe_sample_id(value: Any) -> str:
    sample_id = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())
    if not sample_id:
        raise CandidateError("instance_id is empty")
    return sample_id[:180]


def iter_local_rows(path: Path, offset: int, limit: int) -> Iterator[dict[str, Any]]:
    if not path.is_file():
        raise CandidateError(f"local input does not exist: {path}")
    if path.suffix.lower() == ".jsonl":
        selected = 0
        with path.open("r", encoding="utf-8") as handle:
            for index, raw_line in enumerate(handle):
                if index < offset or not raw_line.strip():
                    continue
                if selected >= limit:
                    break
                value = json.loads(raw_line)
                if not isinstance(value, dict):
                    raise CandidateError(f"JSONL row {index + 1} is not an object")
                selected += 1
                yield value
        return

    value = json.loads(path.read_text(encoding="utf-8"))
    rows = value if isinstance(value, list) else value.get("data")
    if not isinstance(rows, list):
        raise CandidateError("local JSON must be a list or {'data': [...]}")
    for row in rows[offset : offset + limit]:
        if not isinstance(row, dict):
            raise CandidateError("local JSON contains a non-object row")
        yield row


def iter_source_rows(args: argparse.Namespace) -> Iterator[dict[str, Any]]:
    if args.source == "local":
        if not args.input:
            raise CandidateError("--input is required for --source local")
        yield from iter_local_rows(Path(args.input), args.offset, args.limit)
        return

    url = args.hf_jsonl_url or (
        f"https://huggingface.co/datasets/{args.hf_dataset}/resolve/main/"
        f"{args.hf_filename}"
    )
    headers: dict[str, str] = {}
    if os.environ.get("HF_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
    yield from iter_remote_jsonl_rows(
        url=url,
        headers=headers,
        offset=args.offset,
        limit=args.limit,
        timeout=args.source_timeout,
        retries=args.source_retries,
        # Hugging Face's Xet-backed endpoint occasionally closes 1 MiB range
        # responses early. Smaller ranges are materially more reliable and the
        # iterator still stops as soon as the requested rows are complete.
        chunk_size=256 * 1024,
    )


def source_cache_path(work_dir: Path, args: argparse.Namespace) -> Path:
    if args.source == "hf":
        effective_url = args.hf_jsonl_url or (
            f"https://huggingface.co/datasets/{args.hf_dataset}/resolve/main/"
            f"{args.hf_filename}"
        )
        identity = {
            "source": "hf",
            "dataset": args.hf_dataset,
            "filename": args.hf_filename,
            "url": effective_url,
            "offset": args.offset,
            "limit": args.limit,
        }
    else:
        if not args.input:
            raise CandidateError("--input is required for --source local")
        input_path = Path(args.input).resolve()
        if not input_path.is_file():
            raise CandidateError(f"local input does not exist: {input_path}")
        identity = {
            "source": "local",
            "path": str(input_path),
            "content_sha256": sha256_file(input_path),
            "offset": args.offset,
            "limit": args.limit,
        }
    identity_digest = sha256_text(
        json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )
    source_key = (
        f"{args.source}_offset{args.offset}_limit{args.limit}_{identity_digest[:20]}"
    )
    return work_dir / f"source_rows_{source_key}.jsonl"


def trim_source_row(row: dict[str, Any]) -> dict[str, Any]:
    """Keep only fields needed to build a sample.

    In particular, do not cache ``test_binary_archive_b64``; it can be very
    large and is not part of the five-part SpecGAP artifact.
    """
    keys = (
        "instance_id",
        "document",
        "github_url",
        "parent_commit",
        "repo",
        "user",
        "pypi_name",
        "difficulty",
        "license_spdx_id",
        "image_url",
        "image",
        "workdir",
        "test_patch",
        "test_files",
        "passed_ptp",
        "unit_test",
    )
    return {key: row.get(key) for key in keys}


def load_or_create_source_cache(
    work_dir: Path, args: argparse.Namespace
) -> list[dict[str, Any]]:
    cache_path = source_cache_path(work_dir, args)
    if cache_path.is_file():
        cached = list(iter_local_rows(cache_path, 0, args.limit))
        if len(cached) == args.limit:
            print(f"[source] using {cache_path}", flush=True)
            return cached
        print(
            f"[source] ignoring incomplete cache ({len(cached)}/{args.limit})",
            flush=True,
        )

    rows = [trim_source_row(row) for row in iter_source_rows(args)]
    if len(rows) != args.limit:
        raise CandidateError(
            f"requested {args.limit} rows but source yielded {len(rows)}"
        )
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(cache_path)
    print(f"[source] cached {len(rows)} rows in {cache_path}", flush=True)
    return rows


def validate_source_row(row: dict[str, Any]) -> None:
    required_strings = (
        "instance_id",
        "document",
        "github_url",
        "parent_commit",
        "repo",
        "user",
    )
    missing = [
        key for key in required_strings if not str(row.get(key) or "").strip()
    ]
    if missing:
        raise CandidateError(f"source row missing required fields: {missing}")
    commit = str(row["parent_commit"]).strip()
    if not re.fullmatch(r"[0-9a-fA-F]{40}", commit):
        raise CandidateError(f"parent_commit is not a 40-character SHA: {commit!r}")
    if len(str(row["document"])) < 100:
        raise CandidateError("source document is unexpectedly short")


def condition_prompt(row: dict[str, Any]) -> list[dict[str, str]]:
    document = str(row['document'])
    schema = {'considered_conditions': [{'condition_id': 'kc_001', 'type': 'api_constraint', 'importance': 'must_ask', 'source_text': 'an exact, contiguous quote from the document', 'normalized_condition': 'one atomic contract', 'deletion_texts': ['one exact and uniquely occurring document span to delete'], 'why_important': 'why this needs verification', 'sa_gap': ['perception', 'comprehension', 'projection'], 'expected_verification_question': 'one concrete question', 'downstream_impact': 'specific implementation or validation impact'}], 'selected_condition_ids': ['kc_001', 'kc_002', 'kc_003', 'kc_004', 'kc_005'], 'annotation_notes': ''}
    system = 'You are a meticulous benchmark annotator. Produce JSON only. Do not infer text that is absent from the supplied document.'
    user = f'\nConstruct a SpecGAP 2.2 sample for AgentMonBench using only the complete original document below. Disregard earlier annotations.\n\nRules:\n1. Propose 6-12 independent atomic conditions and preferably select five. Select four only when a fifth fails the quality criteria; never select only three.\n2. Selected conditions must be must_ask or worth_ask. Removing one must leave at least two reasonable implementation choices worth verifying with the user.\n3. Delete original text only. Each deletion_text must be a nonempty contiguous verbatim span occurring exactly once in the complete document.\n4. Delete complete sentences, list items, code lines, or natural short passages. Leave no fragments, dangling punctuation, empty headings, or broken Markdown.\n5. Merge repeated expressions of the same condition and list every occurrence in deletion_texts. Each span must remain individually unique.\n6. Do not mask, rewrite, replace, insert ellipses, or rely on later whitespace or punctuation cleanup.\n7. source_text must also be a contiguous verbatim quotation. normalized_condition must not introduce an unstated contract.\n8. Selected deletion spans must not overlap or destroy the main task.\n9. Prefer conditions grounded in implementation, tests, configuration, or data flow. Do not select background descriptions or generic engineering advice to fill a quota.\n10. Cover at least three distinct types. With comparable evidence quality, diversify across APIs, boundaries, data flow, dependencies, evaluation, and evidence boundaries.\n11. Use unique condition IDs kc_001, kc_002, etc. Allowed types:\n{json.dumps(sorted(CONDITION_TYPES), ensure_ascii=False)}\nAllowed importance values: {json.dumps(sorted(IMPORTANCE_VALUES), ensure_ascii=False)}\nAllowed sa_gap values: {json.dumps(sorted(SA_GAPS), ensure_ascii=False)}\n\nOutput structure:\n{json.dumps(schema, ensure_ascii=False, indent=2)}\n\nSample metadata:\ninstance_id={row.get('instance_id')}\nrepo={row.get('user')}/{row.get('repo')}\nparent_commit={row.get('parent_commit')}\n\nComplete task document, without truncation:\n<document>\n{document}\n</document>\n'.strip()
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]


def repair_prompt(original_messages: list[dict[str, str]], prior_output: str, errors: Sequence[str], stage: str) -> list[dict[str, str]]:
    return [*original_messages, {'role': 'assistant', 'content': prior_output}, {'role': 'user', 'content': f'The previous {stage} JSON failed deterministic validation. Return a complete corrected JSON object without commentary or code fences. Fix actual errors; do not turn direct into no_direct_mapping merely to bypass coordinate or symbol errors. Preserve every valid evidence item, location, test, and prior condition repair. Keep schema field types; no_direct_mapping_reason.code is one enum string, not an array. Errors:\n- ' + '\n- '.join(errors[:30])}]


def exact_occurrence(document: str, text: str) -> tuple[int, int]:
    if not text:
        raise CandidateError("deletion/source text is empty")
    starts: list[int] = []
    cursor = 0
    while True:
        index = document.find(text, cursor)
        if index < 0:
            break
        starts.append(index)
        cursor = index + max(1, len(text))
    if len(starts) != 1:
        raise CandidateError(
            f"text must occur exactly once, found {len(starts)}: {text[:160]!r}"
        )
    return starts[0], starts[0] + len(text)


def line_for_offset(document: str, offset: int) -> int:
    return document.count("\n", 0, offset) + 1


def resolve_recorded_deletion_span(
    document: str, requested_text: str
) -> tuple[int, int, str]:
    """Resolve a unique target and include an otherwise orphaned line marker.

    This is not unrecorded cleanup: when expansion applies, the expanded
    whitespace/bullet/newline is stored verbatim in ``exact_text`` and covered
    by the recorded offsets.  Expansion is limited to lines where everything
    outside the requested text is only indentation or a Markdown list marker.
    """
    start, end = exact_occurrence(document, requested_text)
    if "\n" in requested_text:
        return start, end, requested_text
    line_start = document.rfind("\n", 0, start) + 1
    newline_index = document.find("\n", end)
    line_end = len(document) if newline_index < 0 else newline_index
    prefix = document[line_start:start]
    suffix = document[end:line_end]
    marker_only = re.fullmatch(r"[ \t]*(?:(?:[-*+]|\d+\.)[ \t]+)?", prefix)
    if marker_only and not suffix.strip():
        expanded_end = line_end + (1 if newline_index >= 0 else 0)
        expanded_text = document[line_start:expanded_end]
        if expanded_text.strip():
            return line_start, expanded_end, expanded_text
    return start, end, requested_text


def normalize_condition_candidate(
    raw: dict[str, Any],
    document: str,
    selected: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    condition_id = str(raw.get("condition_id") or raw.get("id") or "").strip()
    if not re.fullmatch(r"kc_\d{3}", condition_id):
        raise CandidateError(f"invalid condition_id: {condition_id!r}")
    condition_type = str(raw.get("type") or "").strip()
    if condition_type not in CONDITION_TYPES:
        raise CandidateError(f"{condition_id}: invalid type {condition_type!r}")
    importance = str(raw.get("importance") or "").strip()
    if importance not in IMPORTANCE_VALUES:
        raise CandidateError(f"{condition_id}: invalid importance {importance!r}")
    if selected and importance not in {"must_ask", "worth_ask"}:
        raise CandidateError(f"{condition_id}: selected condition is can_ignore")

    required_text = (
        "source_text",
        "normalized_condition",
        "why_important",
        "expected_verification_question",
        "downstream_impact",
    )
    strings = {key: str(raw.get(key) or "").strip() for key in required_text}
    empty = [key for key, value in strings.items() if not value]
    if empty:
        raise CandidateError(f"{condition_id}: empty fields {empty}")
    exact_occurrence(document, strings["source_text"])

    raw_gaps = raw.get("sa_gap")
    if not isinstance(raw_gaps, list):
        raise CandidateError(f"{condition_id}: sa_gap must be a list")
    gaps = list(dict.fromkeys(str(value) for value in raw_gaps))
    if not gaps or any(value not in SA_GAPS for value in gaps):
        raise CandidateError(f"{condition_id}: invalid or empty sa_gap")

    raw_texts = raw.get("deletion_texts")
    if not isinstance(raw_texts, list) or not raw_texts:
        raise CandidateError(f"{condition_id}: deletion_texts must be non-empty")
    deletion_texts = [str(value) for value in raw_texts]
    if any(not value for value in deletion_texts):
        raise CandidateError(f"{condition_id}: empty deletion_text")
    if len(set(deletion_texts)) != len(deletion_texts):
        raise CandidateError(f"{condition_id}: duplicate deletion_texts")

    review_condition = {
        "condition_id": condition_id,
        "type": condition_type,
        "importance": importance,
        **strings,
        "sa_gap": gaps,
        "deletion_texts": deletion_texts,
        "selected_for_deletion": selected,
    }
    if not selected:
        return review_condition, None

    spans = []
    recorded_deletion_texts: list[str] = []
    for index, exact_text in enumerate(deletion_texts, start=1):
        try:
            char_start, char_end, recorded_text = resolve_recorded_deletion_span(
                document, exact_text
            )
        except CandidateError as exc:
            # A model occasionally paraphrases deletion_text while still
            # supplying a valid verbatim source_text. Falling back to that
            # already-validated quote remains strict/exact (never fuzzy), and
            # the semantic audit will reject it if it does not remove the
            # complete requirement.
            if (
                "found 0" not in str(exc)
                or strings["source_text"] in recorded_deletion_texts
            ):
                raise
            char_start, char_end, recorded_text = resolve_recorded_deletion_span(
                document, strings["source_text"]
            )
        recorded_deletion_texts.append(recorded_text)
        spans.append(
            {
                "span_id": f"{condition_id}_ds{index:03d}",
                "role": "semantic_requirement",
                "exact_text": recorded_text,
                "char_start": char_start,
                "char_end": char_end,
                "line_start": line_for_offset(document, char_start),
                "line_end": line_for_offset(document, max(char_start, char_end - 1)),
            }
        )
    review_condition["deletion_texts"] = recorded_deletion_texts
    deleted_part = {
        "condition_id": condition_id,
        "type": condition_type,
        "importance": importance,
        "source_text": strings["source_text"],
        "normalized_condition": strings["normalized_condition"],
        "deleted_spans": spans,
        "why_important": strings["why_important"],
        "sa_gap": gaps,
        "expected_verification_question": strings[
            "expected_verification_question"
        ],
        "downstream_impact": strings["downstream_impact"],
    }
    return review_condition, deleted_part


def delete_exact_spans(document: str, deleted_parts: Sequence[dict[str, Any]]) -> str:
    spans: list[tuple[int, int, str, str]] = []
    for part in deleted_parts:
        for span in part["deleted_spans"]:
            start = int(span["char_start"])
            end = int(span["char_end"])
            exact_text = str(span["exact_text"])
            if document[start:end] != exact_text:
                raise CandidateError(
                    f"{span['span_id']}: exact_text does not match document offsets"
                )
            spans.append((start, end, str(span["span_id"]), exact_text))
    spans.sort(key=lambda value: (value[0], value[1]))
    for previous, current in zip(spans, spans[1:]):
        if current[0] < previous[1]:
            raise CandidateError(
                f"overlapping spans: {previous[2]} and {current[2]}"
            )
    result = document
    for start, end, _, _ in reversed(spans):
        result = result[:start] + result[end:]
    if result == document:
        raise CandidateError("deletion produced an unchanged document")
    if len(result) < max(100, int(len(document) * 0.70)):
        raise CandidateError("selected deletions remove more than 30% of the document")
    return result


def normalize_condition_output(
    data: dict[str, Any], document: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], str]:
    raw_conditions = data.get("considered_conditions")
    selected_ids = data.get("selected_condition_ids")
    if not isinstance(raw_conditions, list) or not 6 <= len(raw_conditions) <= 12:
        raise CandidateError("considered_conditions must contain 6 to 12 objects")
    if (
        not isinstance(selected_ids, list)
        or not MIN_GENERATED_SELECTED_CONDITIONS <= len(selected_ids) <= 5
        or len(set(map(str, selected_ids))) != len(selected_ids)
    ):
        raise CandidateError("selected_condition_ids must contain 4 to 5 unique IDs")
    selected = [str(value).strip() for value in selected_ids]

    raw_by_id: dict[str, dict[str, Any]] = {}
    for raw in raw_conditions:
        if not isinstance(raw, dict):
            raise CandidateError("considered condition is not an object")
        condition_id = str(raw.get("condition_id") or raw.get("id") or "").strip()
        if condition_id in raw_by_id:
            raise CandidateError(f"duplicate condition_id: {condition_id}")
        raw_by_id[condition_id] = raw
    unknown = [value for value in selected if value not in raw_by_id]
    if unknown:
        raise CandidateError(f"selected IDs are not considered conditions: {unknown}")
    selected_types = {
        str(raw_by_id[value].get("type") or "").strip()
        for value in selected
    }
    if len(selected_types) < MIN_GENERATED_CONDITION_TYPES:
        raise CandidateError(
            "selected conditions must cover at least 3 distinct condition types"
        )

    review_conditions: list[dict[str, Any]] = []
    deleted_by_id: dict[str, dict[str, Any]] = {}
    for raw in raw_conditions:
        condition_id = str(raw.get("condition_id") or raw.get("id") or "").strip()
        review, deleted = normalize_condition_candidate(
            raw, document, condition_id in selected
        )
        review_conditions.append(review)
        if deleted is not None:
            deleted_by_id[condition_id] = deleted
    deleted_parts = [deleted_by_id[value] for value in selected]
    delete_exact_spans(document, deleted_parts)
    return (
        review_conditions,
        deleted_parts,
        selected,
        str(data.get("annotation_notes") or "").strip(),
    )


def canonical_condition_payload(
    review_conditions: Sequence[dict[str, Any]],
    selected_ids: Sequence[str],
    annotation_notes: str,
) -> dict[str, Any]:
    return {
        "considered_conditions": [
            {
                key: value
                for key, value in condition.items()
                if key != "selected_for_deletion"
            }
            for condition in review_conditions
        ],
        "selected_condition_ids": list(selected_ids),
        "annotation_notes": annotation_notes,
    }


def condition_audit_prompt(deleted_parts: Sequence[dict[str, Any]], document_after: str) -> list[dict[str, str]]:
    conditions = [{'condition_id': part['condition_id'], 'normalized_condition': part['normalized_condition'], 'source_text': part['source_text'], 'expected_verification_question': part['expected_verification_question']} for part in deleted_parts]
    schema = {'sample_valid': False, 'document_coherent': True, 'conditions': [{'condition_id': 'kc_001', 'status': 'clear', 'evidence_quotes': [], 'explanation': 'why the requirement is or is not recoverable', 'recommendation': 'keep'}], 'issues': []}
    system = 'You are an adversarial benchmark reviewer. Your job is to reject SpecGAP deletions whose missing requirement remains recoverable. Return JSON only.'
    user = f'\nReview the SpecGAP document after deletion. Mechanically correct spans do not establish semantic validity.\n\nRequirements:\n1. For each condition, consider only the remaining document. If any sentence, list, signature, example, note, or repeated passage still states the same contract, use status=leaked.\n2. Semantically equivalent statements also leak: reject recoverable constants, algorithms, defaults, boundaries, data flows, prohibitions, or test expectations.\n3. evidence_quotes must be exact contiguous quotations from the remaining document. Include every leaking occurrence.\n4. Use status=coherence_problem and document_coherent=false for sentence fragments, dangling punctuation, empty headings, broken lists/code blocks, or conspicuous editing artifacts.\n5. sample_valid may be true only when every condition is clear, the text reads naturally, and the main task remains intact.\n6. recommendation must be keep, expand_deletion, or replace_condition. Prefer replace_condition when further deletion would damage the main task.\n7. Return conditions in the same order and with exactly the supplied IDs.\n\nOutput structure:\n{json.dumps(schema, ensure_ascii=False, indent=2)}\n\nConditions to review:\n{json.dumps(conditions, ensure_ascii=False, indent=2)}\n\nDocument after deletion:\n<document_after>\n{document_after}\n</document_after>\n'.strip()
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]


def normalize_condition_audit(
    data: dict[str, Any],
    deleted_parts: Sequence[dict[str, Any]],
    document_after: str,
) -> dict[str, Any]:
    expected_ids = [part["condition_id"] for part in deleted_parts]
    raw_conditions = data.get("conditions")
    if not isinstance(raw_conditions, list):
        raise CandidateError("semantic audit conditions must be a list")
    actual_ids = [
        str(item.get("condition_id") or "").strip()
        for item in raw_conditions
        if isinstance(item, dict)
    ]
    if len(actual_ids) != len(raw_conditions) or actual_ids != expected_ids:
        raise CandidateError(
            f"semantic audit IDs/order mismatch: expected {expected_ids}, got {actual_ids}"
        )
    normalized_conditions: list[dict[str, Any]] = []
    allowed_statuses = {"clear", "leaked", "coherence_problem"}
    allowed_recommendations = {"keep", "expand_deletion", "replace_condition"}
    for raw in raw_conditions:
        condition_id = str(raw["condition_id"])
        status = str(raw.get("status") or "").strip()
        recommendation = str(raw.get("recommendation") or "").strip()
        explanation = str(raw.get("explanation") or "").strip()
        quotes = raw.get("evidence_quotes")
        if status not in allowed_statuses:
            raise CandidateError(f"{condition_id}: invalid audit status")
        if recommendation not in allowed_recommendations:
            raise CandidateError(f"{condition_id}: invalid audit recommendation")
        if not explanation:
            raise CandidateError(f"{condition_id}: empty audit explanation")
        if not isinstance(quotes, list) or any(
            not isinstance(value, str) or not value for value in quotes
        ):
            raise CandidateError(f"{condition_id}: invalid audit evidence_quotes")
        quotes = list(dict.fromkeys(quotes))
        for quote in quotes:
            if quote not in document_after:
                raise CandidateError(
                    f"{condition_id}: audit quote is not verbatim in after-document"
                )
        if status == "leaked" and not quotes:
            raise CandidateError(
                f"{condition_id}: leaked audit result lacks evidence quote"
            )
        if status == "clear" and recommendation != "keep":
            raise CandidateError(
                f"{condition_id}: clear condition must recommend keep"
            )
        normalized_conditions.append(
            {
                "condition_id": condition_id,
                "status": status,
                "evidence_quotes": quotes,
                "explanation": explanation,
                "recommendation": recommendation,
            }
        )
    raw_issues = data.get("issues")
    if not isinstance(raw_issues, list) or any(
        not isinstance(value, str) or not value.strip() for value in raw_issues
    ):
        raise CandidateError("semantic audit issues must be a string list")
    issues = [str(value).strip() for value in raw_issues]
    document_coherent = data.get("document_coherent")
    sample_valid = data.get("sample_valid")
    if not isinstance(document_coherent, bool) or not isinstance(sample_valid, bool):
        raise CandidateError("semantic audit booleans are missing")
    computed_valid = (
        document_coherent
        and not issues
        and all(item["status"] == "clear" for item in normalized_conditions)
    )
    if sample_valid != computed_valid:
        raise CandidateError(
            "semantic audit sample_valid contradicts condition/issues results"
        )
    return {
        "sample_valid": sample_valid,
        "document_coherent": document_coherent,
        "conditions": normalized_conditions,
        "issues": issues,
    }


def audit_errors(audit: dict[str, Any]) -> list[str]:
    errors = list(audit['issues'])
    if not audit['document_coherent']:
        errors.append('The document after deletion has structural or readability problems.')
    for condition in audit['conditions']:
        if condition['status'] == 'clear':
            continue
        quote_text = ' | '.join((repr(value) for value in condition['evidence_quotes']))
        errors.append(f'{condition['condition_id']} status={condition['status']}; {condition['explanation']}; recommendation={condition['recommendation']}; remaining quotes={quote_text}')
    return errors or ['semantic audit rejected the condition proposal']


def expand_audited_leaks(
    *,
    review_conditions: Sequence[dict[str, Any]],
    selected_ids: Sequence[str],
    annotation_notes: str,
    audit: dict[str, Any],
    document_before: str,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[str],
    str,
] | None:
    """Deterministically add every exact leak quote when expansion is safe.

    The adversarial audit already supplies verbatim after-document evidence.
    Requiring another model to copy those strings back into deletion targets is
    both slower and less reliable.  Replacement/coherence decisions still go
    back to the condition annotator; only explicit ``expand_deletion`` results
    are handled here.
    """
    if not audit["document_coherent"]:
        return None
    by_id = {
        condition["condition_id"]: dict(condition)
        for condition in review_conditions
    }
    changed = False
    for result in audit["conditions"]:
        if result["status"] == "clear":
            continue
        if (
            result["status"] != "leaked"
            or result["recommendation"] != "expand_deletion"
        ):
            return None
        condition = by_id[result["condition_id"]]
        deletion_texts = list(condition["deletion_texts"])
        for quote in result["evidence_quotes"]:
            # Quotes came from the after-document and therefore cannot overlap
            # an already deleted span. They must still be unique in the before
            # document to retain deterministic coordinates.
            exact_occurrence(document_before, quote)
            if quote not in deletion_texts:
                deletion_texts.append(quote)
                changed = True
        condition["deletion_texts"] = deletion_texts
        by_id[result["condition_id"]] = condition
    if not changed:
        return None
    condition_payload = {
        "considered_conditions": [
            {
                key: value
                for key, value in by_id[condition["condition_id"]].items()
                if key != "selected_for_deletion"
            }
            for condition in review_conditions
        ],
        "selected_condition_ids": list(selected_ids),
        "annotation_notes": annotation_notes,
    }
    return normalize_condition_output(condition_payload, document_before)


def generate_audited_conditions(*, row: dict[str, Any], run_dir: Path, args: argparse.Namespace, provider: str, model: str, base_url: str, api_key: str, initial_condition_payload: dict[str, Any] | None=None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], str, dict[str, Any]]:
    document_before = str(row['document'])
    original_messages = condition_prompt(row)
    condition_messages = original_messages
    last_audit: dict[str, Any] | None = None
    for round_index in range(1, args.retries + 1):
        if round_index == 1 and initial_condition_payload is not None:
            try:
                condition_result = normalize_condition_output(initial_condition_payload, document_before)
            except Exception as exc:
                write_json(run_dir / 'condition_seed_validation.json', {'valid': False, 'errors': [str(exc)]})
                raise
            write_json(run_dir / 'condition_seed_validation.json', {'valid': True, 'errors': []})
        else:
            condition_result = call_candidate_stage(stage=f'conditions_round_{round_index:02d}', base_messages=condition_messages, normalize=lambda value: normalize_condition_output(value, document_before), run_dir=run_dir, args=args, provider=provider, model=model, base_url=base_url, api_key=api_key)
        review_conditions, deleted_parts, selected_ids, annotation_notes = condition_result
        document_after = delete_exact_spans(document_before, deleted_parts)
        audit_messages = condition_audit_prompt(deleted_parts, document_after)
        last_audit = call_candidate_stage(stage=f'semantic_audit_round_{round_index:02d}', base_messages=audit_messages, normalize=lambda value: normalize_condition_audit(value, deleted_parts, document_after), run_dir=run_dir, args=args, provider=provider, model=model, base_url=base_url, api_key=api_key)
        write_json(run_dir / 'semantic_audit.latest.json', last_audit)
        if last_audit['sample_valid']:
            return (review_conditions, deleted_parts, selected_ids, annotation_notes, last_audit)
        for expansion_index in range(1, args.retries + 1):
            expanded = expand_audited_leaks(review_conditions=review_conditions, selected_ids=selected_ids, annotation_notes=annotation_notes, audit=last_audit, document_before=document_before)
            if expanded is None:
                break
            review_conditions, deleted_parts, selected_ids, annotation_notes = expanded
            document_after = delete_exact_spans(document_before, deleted_parts)
            expansion_audit_messages = condition_audit_prompt(deleted_parts, document_after)
            last_audit = call_candidate_stage(stage=f'semantic_audit_round_{round_index:02d}_expansion_{expansion_index:02d}', base_messages=expansion_audit_messages, normalize=lambda value: normalize_condition_audit(value, deleted_parts, document_after), run_dir=run_dir, args=args, provider=provider, model=model, base_url=base_url, api_key=api_key)
            write_json(run_dir / 'semantic_audit.latest.json', last_audit)
            if last_audit['sample_valid']:
                return (review_conditions, deleted_parts, selected_ids, annotation_notes, last_audit)
        condition_payload = canonical_condition_payload(review_conditions, selected_ids, annotation_notes)
        condition_messages = repair_prompt(original_messages, json.dumps(condition_payload, ensure_ascii=False), audit_errors(last_audit), 'Post-deletion semantic review')
    raise CandidateError(f'condition proposal failed semantic leak/coherence audit after {args.retries} rounds: {'; '.join(audit_errors(last_audit or {'issues': [], 'document_coherent': False, 'conditions': []}))}')


def compact_deleted_condition(part: dict[str, Any]) -> dict[str, Any]:
    """Return only the condition fields an independent mapping pass may see."""

    return {
        "condition_id": part["condition_id"],
        "type": part["type"],
        "source_text": part["source_text"],
        "normalized_condition": part["normalized_condition"],
        "deleted_spans": [span["exact_text"] for span in part["deleted_spans"]],
        "why_important": part["why_important"],
    }


def mapping_output_schema(condition_id: str = "kc_001") -> dict[str, Any]:
    """Describe one complete, pre-normalization mapping returned by an AI."""

    return {
        "code_mappings": [
            {
                "condition_id": condition_id,
                "mapping_status": "direct",
                "coverage": "full",
                "mapping_explanation": "how all evidence jointly maps the requirement",
                "no_direct_mapping_reason": None,
                "evidence": [
                    {
                        "evidence_type": "implementation",
                        "relation": "implements",
                        "strength": "direct",
                        "locations": [
                            {
                                "file_path": "src/package/module.py",
                                "symbol": {
                                    "kind": "function",
                                    "qualified_name": "function_name",
                                },
                                "line_ranges": [{"start": 10, "end": 24}],
                                "flow_step": None,
                            }
                        ],
                        "explanation": "precise correspondence",
                        "confidence": "high",
                    }
                ],
                "related_tests": {
                    "status": "none_found",
                    "evidence_indexes": [],
                    "search_notes": "which test locations were inspected",
                },
            }
        ]
    }


def mapping_rules() -> str:
    """Shared semantic contract for both final mapping routes."""
    return f"""\n1. code_mappings must correspond one-to-one to the input conditions in their original order.\n2. Set mapping_status=direct only with a production implementation anchor, including evidence_type=implementation and strength=direct. Add data_flow evidence for cross-call relationships without replacing implementation anchors. Each location must identify the function/method/class that actually contains its lines.\n3. If evidence exists only in tests, configuration, fixtures, documentation, or cross-file data flow, or no reliable anchor exists, use mapping_status=no_direct_mapping and provide no_direct_mapping_reason:\n   {{"code": <one of {sorted(NO_DIRECT_REASON_CODES)}>, "detail": "specific reason"}}.\n   Do not force a match to avoid a null result.\n4. Record distinct files, functions, classes, and flow steps separately. Do not concatenate paths or symbols. Number locations within a flow using flow_step=1,2,...\n5. Cite only supplied repository-relative paths and 1-based inclusive line ranges. Do not cite task-document line numbers or invent unseen files.\n6. Assign implementation/test/configuration/data_flow/fixture/build_metadata/documentation according to the actual evidence. Locate tests at the function or class level when possible.\n7. related_tests.status is found or none_found. When found, evidence_indexes are 1-based and refer only to test-related test/fixture/documentation(doctest)/build_metadata/configuration evidence. Describe the search scope when none is found.\n8. coverage is full/partial/none. For a partly supported compound condition, use partial and explain missing clauses in mapping_explanation.\n9. Allowed relation values: {sorted(RELATIONS)}; strength is direct/indirect; confidence is high/medium/low.\n10. symbol.kind is function/method/class/module/config_key/data/none. Do not invent function names for module or configuration evidence; qualified_name may be null.\n11. Check normalized_condition and every deleted_span. Unsupported or contradictory extra semantics require partial coverage, an explicit explanation, and contradicts evidence where appropriate.\n12. A test is direct only if its assertions, mocks/spies, or fixtures distinguish the precise condition. Record success-only, compatible-implementation, partial-clause, doctest, and build-test evidence as indirect when it cannot distinguish that condition.\n13. Use related_tests=none_found only after examining all repository test, fixture, doctest, and build-test material. Retain indirect tests. Full coverage requires an implementation location for every clause.\n14. For top-level-only or non-propagating parameters, full coverage requires both the allowed scope's receipt and forwarding/consumption and the excluded scope's omission/default behavior. An omitted recursive argument alone is partial. Cover each named variant, such as List, Tuple, and Dict, separately.\n""".strip()


def condition_shard_prompt(row: dict[str, Any], deleted_part: dict[str, Any], *, shard_id: str, shard_count: int, repo_context: str) -> list[dict[str, str]]:
    """Build one route-A scan prompt for one condition and one complete shard."""
    condition = compact_deleted_condition(deleted_part)
    condition_id = str(condition['condition_id'])
    schema = holistic_shard_output_schema([condition_id], shard_id)
    system = "You independently scan one exhaustive shard of a pinned repository for exactly one deleted requirement. You are not given another condition, the complete task document, or either route's final mapping. Return JSON only."
    user = f'\nCondition-by-condition evidence discovery for condition {condition_id}, shard\n{shard_id} of {shard_count}). Examine only the supplied condition. Discover evidence within this shard; do not infer repository-wide absence from one shard. Aggregate after all shards are scanned.\n\nRules:\n1. condition_findings must include exactly the applicable input conditions, with no additions.\n2. Record all relevant implementation, test, configuration, fixture, documentation, and data-flow evidence with relevant=true; otherwise use relevant=false and evidence=[].\n3. Provide a nonempty summary of what was found or why nothing was found.\n4. Record every distinct file, function, class, and flow step.\n5. Cite only supplied paths and 1-based inclusive line coordinates.\n6. Allowed evidence_type values: {sorted(EVIDENCE_TYPES)}; allowed relations:\n   {sorted(RELATIONS)}; allowed strength values: {sorted(STRENGTH_VALUES)};\n   allowed confidence values: {sorted(CONFIDENCE_VALUES)}.\n7. symbol.kind is function/method/class/module/config_key/data/none.\n8. Record relevant partial-clause, success-only, doctest, or build-test evidence as indirect even when it cannot establish the complete condition.\n9. Search separately for every clause of a compound condition; retain any portion found in this shard.\n10. Record production statements implementing the behavior as implementation/direct. Add cross-function data_flow relationships without dropping the implementation anchor. Each location must name its actual containing symbol.\n11. For top-level-only/non-propagating boundaries, seek both allowed-scope forwarding/consumption and recursive omission. Examine every named variant and record either side found in the current shard.\n\nOutput structure:\n{json.dumps(schema, ensure_ascii=False, indent=2)}\n\nPinned repository: {row.get('github_url')}@{row.get('parent_commit')}\nOnly deleted condition:\n{json.dumps(condition, ensure_ascii=False, indent=2)}\n\nCurrent repository shard:\n<repository_shard>\n{repo_context}\n</repository_shard>\n'.strip()
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]


def condition_aggregation_prompt(row: dict[str, Any], deleted_part: dict[str, Any], shard_results: Sequence[dict[str, Any]]) -> list[dict[str, str]]:
    """Build route A's final one-condition, all-repository aggregation prompt."""
    condition = compact_deleted_condition(deleted_part)
    condition_id = str(condition['condition_id'])
    system = "You independently aggregate every exhaustive repository-shard finding for exactly one deleted requirement. You have not seen another condition, the complete task document, or the holistic route's output. Return JSON only."
    user = f'\nFinal condition-by-condition aggregation. Process only condition\n{condition_id}. Combine all repository shards, locate files, functions/classes, lines, and relevant tests, and explain their correspondence.\n\nRules:\n{mapping_rules()}\n15. Consider every shard_result before choosing direct or no_direct_mapping. Absence in one shard is not absence from the repository.\n16. Final evidence may only merge, deduplicate, or narrow coordinates already discovered in shard_results. Do not add undiscovered files, symbols, or lines.\n17. Return only this condition.\n\nOutput structure:\n{json.dumps(mapping_output_schema(condition_id), ensure_ascii=False, indent=2)}\n\nPinned repository: {row.get('github_url')}@{row.get('parent_commit')}\nOnly deleted condition:\n{json.dumps(condition, ensure_ascii=False, indent=2)}\n\nAll shard results for this condition, without truncation:\n<repository_shard_results>\n{json.dumps(list(shard_results), ensure_ascii=False, indent=2)}\n</repository_shard_results>\n'.strip()
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]


def holistic_shard_output_schema(
    condition_ids: Sequence[str],
    shard_id: str,
) -> dict[str, Any]:
    return {
        "shard_id": shard_id,
        "condition_findings": [
            {
                "condition_id": condition_id,
                "relevant": True,
                "summary": "what this repository shard does or does not establish",
                "evidence": [
                    {
                        "evidence_type": "implementation",
                        "relation": "implements",
                        "strength": "direct",
                        "locations": [
                            {
                                "file_path": "src/package/module.py",
                                "symbol": {
                                    "kind": "function",
                                    "qualified_name": "function_name",
                                },
                                "line_ranges": [{"start": 10, "end": 24}],
                                "flow_step": None,
                            }
                        ],
                        "explanation": "precise correspondence within this shard",
                        "confidence": "high",
                    }
                ],
            }
            for condition_id in condition_ids
        ],
    }


def holistic_shard_prompt(row: dict[str, Any], document_before: str, deleted_parts: Sequence[dict[str, Any]], *, shard_id: str, shard_count: int, repo_context: str) -> list[dict[str, str]]:
    """Build one independent repository-wide alignment scan shard."""
    compact_conditions = [compact_deleted_condition(part) for part in deleted_parts]
    condition_ids = [str(part['condition_id']) for part in deleted_parts]
    schema = holistic_shard_output_schema(condition_ids, shard_id)
    system = 'You scan one exhaustive shard of a pinned repository for evidence relevant to every selected requirement. This is an independent holistic alignment route. You have not seen and must not assume any per-condition mapping output. Return JSON only.'
    user = f'\nWhole-repository scan, shard {shard_id} of {shard_count}). Examine every selected condition in input order. Discover evidence within this shard; do not infer repository-wide absence from one shard. Aggregate after all shards are scanned.\n\nRules:\n1. condition_findings must include exactly the applicable input conditions, with no additions.\n2. Record all relevant implementation, test, configuration, fixture, documentation, and data-flow evidence with relevant=true; otherwise use relevant=false and evidence=[].\n3. Provide a nonempty summary of what was found or why nothing was found.\n4. Record every distinct file, function, class, and flow step.\n5. Cite only supplied paths and 1-based inclusive line coordinates.\n6. Allowed evidence_type values: {sorted(EVIDENCE_TYPES)}; allowed relations:\n   {sorted(RELATIONS)}; allowed strength values: {sorted(STRENGTH_VALUES)};\n   allowed confidence values: {sorted(CONFIDENCE_VALUES)}.\n7. symbol.kind is function/method/class/module/config_key/data/none.\n8. Record relevant partial-clause, success-only, doctest, or build-test evidence as indirect even when it cannot establish the complete condition.\n9. Search separately for every clause of a compound condition; retain any portion found in this shard.\n10. Record production statements implementing the behavior as implementation/direct. Add cross-function data_flow relationships without dropping the implementation anchor. Each location must name its actual containing symbol.\n11. For top-level-only/non-propagating boundaries, seek both allowed-scope forwarding/consumption and recursive omission. Examine every named variant and record either side found in the current shard.\n\nOutput structure:\n{json.dumps(schema, ensure_ascii=False, indent=2)}\n\nPinned repository: {row.get('github_url')}@{row.get('parent_commit')}\nAll selected conditions:\n{json.dumps(compact_conditions, ensure_ascii=False, indent=2)}\n\nComplete task document, identical for every shard:\n<document_before>\n{document_before}\n</document_before>\n\nCurrent repository shard:\n<repository_shard>\n{repo_context}\n</repository_shard>\n'.strip()
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]


def holistic_aggregation_prompt(row: dict[str, Any], document_before: str, deleted_parts: Sequence[dict[str, Any]], shard_results: Sequence[dict[str, Any]]) -> list[dict[str, str]]:
    """Build route B's final document/repository-wide alignment prompt."""
    compact_conditions = [compact_deleted_condition(part) for part in deleted_parts]
    condition_ids = [str(part['condition_id']) for part in deleted_parts]
    schema = {'alignment_summary': 'non-empty summary of the complete document/repository alignment', 'code_mappings': [mapping_output_schema(condition_id)['code_mappings'][0] for condition_id in condition_ids]}
    system = 'You perform a complete document-to-repository alignment using every repository shard result. This route is independent: you have not seen and must not assume any condition-by-condition mapping output. Return JSON only.'
    user = f'\nAlign the complete task document with all pinned-repository shard results and return complete, inspectable mappings for every selected condition.\n\nRules:\n{mapping_rules()}\n15. Consider all shard_results before choosing direct or no_direct_mapping; one shard cannot establish repository-wide absence.\n16. Merge duplicate or overlapping evidence while retaining every distinct file, function, class, and flow step.\n17. Provide a nonempty alignment_summary of the complete document, selected conditions, and implementation/test/configuration alignment.\n\nOutput structure (code_mappings must cover every input condition):\n{json.dumps(schema, ensure_ascii=False, indent=2)}\n\nPinned repository: {row.get('github_url')}@{row.get('parent_commit')}\nAll selected conditions:\n{json.dumps(compact_conditions, ensure_ascii=False, indent=2)}\n\nComplete task document:\n<document_before>\n{document_before}\n</document_before>\n\nAll shard results, without truncation or condition-by-condition outputs:\n<repository_shard_results>\n{json.dumps(list(shard_results), ensure_ascii=False, indent=2)}\n</repository_shard_results>\n'.strip()
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]


def complete_repository_context(
    repository_shards: Sequence[dict[str, Any]],
) -> str:
    """Render every deterministic repository shard without truncation."""

    rendered: list[str] = []
    for shard_index, shard in enumerate(repository_shards, start=1):
        rendered.append(
            f'<repository_shard id="shard_{shard_index:03d}">\n'
            f'{str(shard["context"])}\n'
            "</repository_shard>"
        )
    return "\n\n".join(rendered)


def condition_quality_audit_prompt(row: dict[str, Any], deleted_part: dict[str, Any], current_mapping: dict[str, Any], *, repo_context: str) -> list[dict[str, str]]:
    """Recheck one route-A mapping against the complete indexed repository."""
    condition = compact_deleted_condition(deleted_part)
    condition_id = str(condition['condition_id'])
    system = 'You are the final evidence-completeness auditor inside an isolated single-condition mapping route. Recheck the draft against the complete indexed repository. You are not given another condition, the complete task document, or any holistic-route output. Return JSON only.'
    user = f"\nFinal evidence-completeness review for condition {condition_id} in the condition-by-condition route.\ncurrent_mapping is a draft from this route. It may omit files, functions, callers, indirect tests, fixtures, doctests, configuration, or data flow, and may incorrectly claim full/none_found. Re-read the complete repository index and return a corrected mapping rather than assuming the draft is correct.\n\nRules:\n{mapping_rules()}\n15. Decompose normalized_condition and all deleted_spans into clauses and locate each clause's implementation. Missing any clause precludes full coverage.\n16. Inspect definitions, callers/forwarding, result consumers, and all test/fixture/doctest/build-test material. Record tests of omitted optional fields, success paths, top-level hooks, and compatible behavior as indirect when they cannot distinguish the complete condition, and explain why.\n17. Top-level-only/non-propagating conditions require both argument receipt/forwarding into top-level construction and omission during recursive construction. The latter alone is partial. Cover every named variant.\n18. Claim that tests exercise/validate a precise branch only when their inputs trigger it and their assertions distinguish it. Mark adjacent behavior indirect and state the untested part.\n19. Full coverage requires named code terms in backticks to appear in relevant production excerpts that show recognition, dispatch, or processing. Repeating terms in explanations or adding import lines does not suffice; use partial when no implementation is found.\n20. Preserve valid existing evidence, add missing evidence, and remove or downgrade incorrect evidence.\n21. Return only this condition. Do not mention, infer, or claim to have seen other conditions or the holistic route.\n22. Use only paths and lines in the complete index below. For none_found, search_notes must accurately describe all inspected tests, not one shard as the entire repository.\n\nOutput structure:\n{json.dumps(mapping_output_schema(condition_id), ensure_ascii=False, indent=2)}\n\nPinned repository: {row.get('github_url')}@{row.get('parent_commit')}\nOnly deleted condition:\n{json.dumps(condition, ensure_ascii=False, indent=2)}\n\nCurrent draft from this route:\n<current_mapping>\n{json.dumps(current_mapping, ensure_ascii=False, indent=2)}\n</current_mapping>\n\nComplete repository index, all shards:\n<complete_repository_index>\n{repo_context}\n</complete_repository_index>\n".strip()
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]


def holistic_quality_audit_prompt(row: dict[str, Any], document_before: str, deleted_parts: Sequence[dict[str, Any]], alignment_summary: str, current_mappings: Sequence[dict[str, Any]], *, repo_context: str) -> list[dict[str, str]]:
    """Recheck route B using the full document and complete repository index."""
    compact_conditions = [compact_deleted_condition(part) for part in deleted_parts]
    condition_ids = [str(part['condition_id']) for part in deleted_parts]
    schema = {'alignment_summary': 'non-empty corrected summary of the complete alignment', 'code_mappings': [mapping_output_schema(condition_id)['code_mappings'][0] for condition_id in condition_ids]}
    system = 'You are the final evidence-completeness auditor inside an independent complete-document/repository alignment route. Recheck its draft against the full document and complete indexed repository. You have not seen and must not assume any condition-by-condition output. Return JSON only.'
    user = f"\nFinal evidence-completeness review for the holistic-alignment route. current_alignment is only this route's draft and may omit files, functions, callers, indirect tests, fixtures, doctests, configuration, or data flow, or incorrectly claim full/none_found. Realign the complete document, every condition, and the complete repository index. Return a corrected alignment_summary and all mappings; do not assume the draft is correct.\n\nRules:\n{mapping_rules()}\n15. Decompose each normalized_condition and all deleted_spans into clauses and locate each clause's implementation. Missing a clause precludes full coverage.\n16. Inspect definitions, callers/forwarding, result consumers, and all test/fixture/doctest/build-test material. Preserve success-only and partial evidence as indirect and explain its limitations.\n17. Top-level-only/non-propagating conditions require both top-level receipt/forwarding and recursive omission. The latter alone is partial. Cover every named variant.\n18. Claim that a test exercises/validates a precise branch only when its input triggers it and assertions distinguish it. Mark adjacent behavior indirect and identify the uncovered part.\n19. Full coverage requires backticked code terms in relevant production excerpts showing recognition, dispatch, or processing. Explanations and imports alone do not suffice; otherwise use partial.\n20. Preserve valid existing evidence, add omissions, and remove or downgrade errors. Keep alignment_summary consistent with the corrected evidence.\n21. Do not cite, guess, or claim to have seen the condition-by-condition route.\n22. Use only paths and lines in the complete index below. For none_found, accurately describe the full test search scope.\n\nOutput structure:\n{json.dumps(schema, ensure_ascii=False, indent=2)}\n\nPinned repository: {row.get('github_url')}@{row.get('parent_commit')}\nAll selected conditions:\n{json.dumps(compact_conditions, ensure_ascii=False, indent=2)}\n\nComplete task document:\n<document_before>\n{document_before}\n</document_before>\n\nCurrent draft from this route:\n<current_alignment>\n{json.dumps({'alignment_summary': alignment_summary, 'code_mappings': list(current_mappings)}, ensure_ascii=False, indent=2)}\n</current_alignment>\n\nComplete repository index, all shards:\n<complete_repository_index>\n{repo_context}\n</complete_repository_index>\n".strip()
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]


def safe_relative_repo_file(repo_root: Path, raw_path: Any) -> tuple[str, Path]:
    value = str(raw_path or "").strip().replace("\\", "/")
    pure = PurePosixPath(value)
    if (
        not value
        or pure.is_absolute()
        or ".." in pure.parts
        or value.startswith(".git/")
    ):
        raise CandidateError(f"unsafe evidence file_path: {value!r}")
    normalized = pure.as_posix()
    path = repo_root.joinpath(*pure.parts)
    try:
        path.resolve().relative_to(repo_root.resolve())
    except ValueError as exc:
        raise CandidateError(f"evidence path escapes repo: {value!r}") from exc
    if not path.is_file():
        raise CandidateError(f"evidence path does not exist: {value!r}")
    return normalized, path


def read_text_lines(path: Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8").splitlines(keepends=True)
    except UnicodeDecodeError as exc:
        raise CandidateError(f"evidence file is not UTF-8: {path}") from exc


def normalize_symbol(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise CandidateError("location.symbol must be an object")
    kind = str(raw.get("kind") or "").strip()
    allowed = {
        "function",
        "method",
        "class",
        "module",
        "config_key",
        "data",
        "none",
    }
    if kind not in allowed:
        raise CandidateError(f"invalid symbol kind: {kind!r}")
    qualified_name = raw.get("qualified_name")
    if qualified_name is not None:
        qualified_name = str(qualified_name).strip() or None
    if kind in {"function", "method", "class"} and not qualified_name:
        raise CandidateError(f"{kind} symbol must have qualified_name")
    return {"kind": kind, "qualified_name": qualified_name}


def normalize_location(
    raw: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    file_path, absolute_path = safe_relative_repo_file(repo_root, raw.get("file_path"))
    lines = read_text_lines(absolute_path)
    raw_ranges = raw.get("line_ranges")
    if not isinstance(raw_ranges, list) or not raw_ranges:
        raise CandidateError(f"{file_path}: line_ranges must be non-empty")
    ranges: list[dict[str, int]] = []
    excerpt_parts: list[str] = []
    for raw_range in raw_ranges:
        if not isinstance(raw_range, dict):
            raise CandidateError(f"{file_path}: line range is not an object")
        try:
            start = int(raw_range.get("start"))
            end = int(raw_range.get("end"))
        except (TypeError, ValueError) as exc:
            raise CandidateError(f"{file_path}: non-integer line range") from exc
        if start < 1 or end < start or end > len(lines):
            raise CandidateError(
                f"{file_path}: invalid line range {start}-{end}; file has {len(lines)} lines"
            )
        ranges.append({"start": start, "end": end})
        excerpt_parts.append("".join(lines[start - 1 : end]))
    excerpt = "\n".join(excerpt_parts)
    flow_step = raw.get("flow_step")
    if flow_step is not None:
        try:
            flow_step = int(flow_step)
        except (TypeError, ValueError) as exc:
            raise CandidateError(f"{file_path}: flow_step must be an integer") from exc
        if flow_step < 1:
            raise CandidateError(f"{file_path}: flow_step must be positive")
    symbol = normalize_symbol(raw.get("symbol"))
    if (
        absolute_path.suffix.lower() in {".py", ".pyi"}
        and symbol["kind"] in {"function", "method", "class"}
        and symbol["qualified_name"] is not None
    ):
        try:
            symbols = _python_symbol_ranges("".join(lines))
        except (SyntaxError, TypeError, ValueError) as exc:
            raise CandidateError(
                f"{file_path}: cannot parse Python file to verify symbol"
            ) from exc
        matches = [
            (start, end)
            for actual_kind, actual_name, start, end in symbols
            if actual_kind == symbol["kind"]
            and (
                symbol["qualified_name"] == actual_name
                or symbol["qualified_name"].endswith(f".{actual_name}")
            )
        ]
        if not matches:
            raise CandidateError(
                f"{file_path}: declared {symbol['kind']} "
                f"{symbol['qualified_name']!r} does not exist"
            )
        if not any(
            all(
                symbol_start <= line_range["start"]
                and line_range["end"] <= symbol_end
                for line_range in ranges
            )
            for symbol_start, symbol_end in matches
        ):
            raise CandidateError(
                f"{file_path}: evidence line ranges are outside declared "
                f"symbol {symbol['qualified_name']!r}"
            )
    return {
        "file_path": file_path,
        "file_sha256": sha256_file(absolute_path),
        "symbol": symbol,
        "line_ranges": ranges,
        "excerpt": excerpt,
        "excerpt_sha256": sha256_text(excerpt),
        "flow_step": flow_step,
    }


def normalize_no_direct_reason(raw: Any) -> dict[str, str]:
    if not isinstance(raw, dict):
        raise CandidateError("no_direct_mapping_reason must be an object")
    code = str(raw.get("code") or "").strip()
    detail = str(raw.get("detail") or "").strip()
    if code not in NO_DIRECT_REASON_CODES or not detail:
        raise CandidateError("invalid or empty no_direct_mapping_reason")
    return {"code": code, "detail": detail}


def normalize_evidence_items(
    raw_evidence: Any,
    *,
    condition_id: str,
    repo_root: Path,
    evidence_prefix: str | None,
) -> list[dict[str, Any]]:
    """Validate evidence and optionally assign a final route-specific ID."""

    if not isinstance(raw_evidence, list):
        raise CandidateError(f"{condition_id}: evidence must be a list")
    evidence: list[dict[str, Any]] = []
    for evidence_index, raw_item in enumerate(raw_evidence, start=1):
        if not isinstance(raw_item, dict):
            raise CandidateError(f"{condition_id}: evidence is not an object")
        evidence_type = str(raw_item.get("evidence_type") or "").strip()
        relation = str(raw_item.get("relation") or "").strip()
        strength = str(raw_item.get("strength") or "").strip()
        item_explanation = str(raw_item.get("explanation") or "").strip()
        confidence = str(raw_item.get("confidence") or "").strip()
        if evidence_type not in EVIDENCE_TYPES:
            raise CandidateError(
                f"{condition_id}: invalid evidence_type {evidence_type!r}"
            )
        if relation not in RELATIONS:
            raise CandidateError(
                f"{condition_id}: invalid evidence relation {relation!r}"
            )
        if strength not in STRENGTH_VALUES:
            raise CandidateError(f"{condition_id}: invalid evidence strength")
        if confidence not in CONFIDENCE_VALUES:
            raise CandidateError(f"{condition_id}: invalid confidence")
        if not item_explanation:
            raise CandidateError(f"{condition_id}: empty evidence explanation")
        raw_locations = raw_item.get("locations")
        if not isinstance(raw_locations, list) or not raw_locations:
            raise CandidateError(f"{condition_id}: evidence locations are empty")
        locations = [
            normalize_location(location, repo_root)
            for location in raw_locations
            if isinstance(location, dict)
        ]
        if len(locations) != len(raw_locations):
            raise CandidateError(f"{condition_id}: invalid evidence location")
        flow_steps = [location["flow_step"] for location in locations]
        if evidence_type == "data_flow":
            if any(step is None for step in flow_steps):
                raise CandidateError(
                    f"{condition_id}: data_flow requires flow_step on every "
                    "location"
                )
            concrete_steps = [int(step) for step in flow_steps]
            if (
                len(concrete_steps) != len(set(concrete_steps))
                or sorted(concrete_steps)
                != list(range(1, len(concrete_steps) + 1))
            ):
                raise CandidateError(
                    f"{condition_id}: data_flow flow_step values must be "
                    "unique and contiguous from 1"
                )
        elif any(step is not None for step in flow_steps):
            raise CandidateError(
                f"{condition_id}: flow_step is only valid for data_flow "
                "evidence"
            )
        normalized_item: dict[str, Any] = {
            "evidence_type": evidence_type,
            "relation": relation,
            "strength": strength,
            "provenance": "original_repo",
            "locations": locations,
            "explanation": item_explanation,
            "confidence": confidence,
        }
        if evidence_prefix is not None:
            normalized_item = {
                "evidence_id": (
                    f"{condition_id}_{evidence_prefix}_ev{evidence_index:03d}"
                ),
                **normalized_item,
            }
        evidence.append(normalized_item)
    return evidence


def _ranges_are_covered(
    target_ranges: Sequence[dict[str, int]],
    allowed_ranges: Sequence[tuple[int, int]],
) -> bool:
    """Return whether every target line is covered by allowed line intervals."""

    merged: list[list[int]] = []
    for start, end in sorted(allowed_ranges):
        if not merged or start > merged[-1][1] + 1:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    for target in target_ranges:
        start = int(target["start"])
        end = int(target["end"])
        covered = any(
            allowed_start <= start and end <= allowed_end
            for allowed_start, allowed_end in merged
        )
        if not covered:
            return False
    return True


def enforce_evidence_within_shard(
    evidence: Sequence[dict[str, Any]],
    allowed_snippets: Sequence[Any],
    *,
    scope: str,
) -> None:
    """Reject evidence whose file/ranges were not visible in the current shard."""

    snippets_by_file: dict[str, list[Any]] = {}
    for snippet in allowed_snippets:
        snippets_by_file.setdefault(str(snippet.file_path), []).append(snippet)

    for evidence_index, item in enumerate(evidence, start=1):
        for location_index, location in enumerate(item["locations"], start=1):
            file_path = str(location["file_path"])
            evidence_type = str(item["evidence_type"])
            compatible_index_types = {
                "implementation": {"implementation"},
                # Executable doctests can be embedded in implementation files.
                "test": {"implementation", "test", "fixture"},
                "fixture": {"test", "fixture"},
                "configuration": {"configuration", "build_metadata"},
                "build_metadata": {"build_metadata"},
                "documentation": {"documentation"},
                "data_flow": {
                    "implementation",
                    "test",
                    "fixture",
                    "configuration",
                    "build_metadata",
                    "documentation",
                },
            }[evidence_type]
            file_snippets = [
                snippet
                for snippet in snippets_by_file.get(file_path, [])
                if str(snippet.evidence_type) in compatible_index_types
            ]
            if not file_snippets:
                raise CandidateError(
                    f"{scope}: evidence {evidence_index} location {location_index} "
                    f"references {file_path!r} outside the current shard or uses "
                    f"evidence_type={evidence_type!r} incompatible with the "
                    "file's repository-index category"
                )
            matching_hash_snippets = [
                snippet
                for snippet in file_snippets
                if str(snippet.file_sha256) == str(location["file_sha256"])
            ]
            if not matching_hash_snippets:
                raise CandidateError(
                    f"{scope}: evidence {evidence_index} location {location_index} "
                    f"file hash was not present in the current shard"
                )
            allowed_ranges = [
                (int(snippet.start_line), int(snippet.end_line))
                for snippet in matching_hash_snippets
            ]
            if not _ranges_are_covered(location["line_ranges"], allowed_ranges):
                raise CandidateError(
                    f"{scope}: evidence {evidence_index} location {location_index} "
                    f"line ranges are outside the current shard snippets"
                )


def enforce_aggregated_mappings_from_shards(
    mappings: Sequence[dict[str, Any]],
    shard_results: Sequence[dict[str, Any]],
    *,
    scope: str,
) -> None:
    """Require final mapping locations to come from that condition's shard evidence."""

    evidence_by_condition: dict[str, list[dict[str, Any]]] = {}
    for shard_result in shard_results:
        for finding in shard_result["condition_findings"]:
            condition_id = str(finding["condition_id"])
            evidence_by_condition.setdefault(condition_id, []).extend(
                finding["evidence"]
            )

    for mapping in mappings:
        condition_id = str(mapping["condition_id"])
        source_evidence = evidence_by_condition.get(condition_id, [])
        for evidence_index, item in enumerate(mapping["evidence"], start=1):
            same_type_sources = [
                source
                for source in source_evidence
                if source["evidence_type"] == item["evidence_type"]
            ]
            if not same_type_sources:
                raise CandidateError(
                    f"{scope}/{condition_id}: final evidence {evidence_index} "
                    "has no same-type source in any shard finding"
                )
            source_locations = [
                source_location
                for source in same_type_sources
                for source_location in source["locations"]
            ]
            for location_index, location in enumerate(item["locations"], start=1):
                matching_locations = [
                    source_location
                    for source_location in source_locations
                    if source_location["file_path"] == location["file_path"]
                    and source_location["file_sha256"] == location["file_sha256"]
                    and source_location["symbol"] == location["symbol"]
                ]
                allowed_ranges = [
                    (int(line_range["start"]), int(line_range["end"]))
                    for source_location in matching_locations
                    for line_range in source_location["line_ranges"]
                ]
                if not matching_locations or not _ranges_are_covered(
                    location["line_ranges"], allowed_ranges
                ):
                    raise CandidateError(
                        f"{scope}/{condition_id}: final evidence {evidence_index} "
                        f"location {location_index} was not discovered for this "
                        "condition by any repository shard"
                    )


def normalize_mapping_output(
    data: dict[str, Any],
    deleted_parts: Sequence[dict[str, Any]],
    repo_root: Path,
    *,
    evidence_prefix: str,
) -> list[dict[str, Any]]:
    if evidence_prefix not in {"cbc", "ha"}:
        raise CandidateError(f"invalid final evidence prefix: {evidence_prefix!r}")
    raw_mappings = data.get("code_mappings")
    if not isinstance(raw_mappings, list):
        raise CandidateError("code_mappings must be a list")
    expected_ids = [part["condition_id"] for part in deleted_parts]
    actual_ids = [
        str(value.get("condition_id") or "").strip()
        for value in raw_mappings
        if isinstance(value, dict)
    ]
    if len(actual_ids) != len(raw_mappings) or actual_ids != expected_ids:
        raise CandidateError(
            f"code_mappings IDs/order mismatch: expected {expected_ids}, got {actual_ids}"
        )

    normalized: list[dict[str, Any]] = []
    for raw_mapping in raw_mappings:
        condition_id = str(raw_mapping["condition_id"])
        status = str(raw_mapping.get("mapping_status") or "").strip()
        coverage = str(raw_mapping.get("coverage") or "").strip()
        explanation = str(raw_mapping.get("mapping_explanation") or "").strip()
        if status not in MAPPING_STATUSES:
            raise CandidateError(f"{condition_id}: invalid mapping_status")
        if coverage not in COVERAGE_VALUES:
            raise CandidateError(f"{condition_id}: invalid coverage")
        if not explanation:
            raise CandidateError(f"{condition_id}: empty mapping_explanation")

        evidence = normalize_evidence_items(
            raw_mapping.get("evidence"),
            condition_id=condition_id,
            repo_root=repo_root,
            evidence_prefix=evidence_prefix,
        )

        direct_impl = any(
            item["evidence_type"] == "implementation"
            and item["strength"] == "direct"
            and item["relation"] != "contradicts"
            for item in evidence
        )
        has_contradiction = any(
            item["relation"] == "contradicts" for item in evidence
        )
        raw_reason = raw_mapping.get("no_direct_mapping_reason")
        if status == "direct":
            if not direct_impl:
                raise CandidateError(
                    f"{condition_id}: direct mapping lacks direct implementation evidence"
                )
            if raw_reason not in (None, "", {}):
                raise CandidateError(
                    f"{condition_id}: direct mapping has no_direct_mapping_reason"
                )
            no_direct_reason = None
            if coverage == "none":
                raise CandidateError(f"{condition_id}: direct mapping has coverage=none")
            if has_contradiction and coverage == "full":
                raise CandidateError(
                    f"{condition_id}: contradicting evidence cannot have "
                    "coverage=full"
                )
        else:
            if direct_impl:
                raise CandidateError(
                    f"{condition_id}: no_direct_mapping contains direct implementation"
                )
            no_direct_reason = normalize_no_direct_reason(raw_reason)
            if not evidence and coverage != "none":
                raise CandidateError(
                    f"{condition_id}: empty evidence requires coverage=none"
                )

        raw_tests = raw_mapping.get("related_tests")
        if not isinstance(raw_tests, dict):
            raise CandidateError(f"{condition_id}: related_tests must be an object")
        test_status = str(raw_tests.get("status") or "").strip()
        search_notes = str(raw_tests.get("search_notes") or "").strip()
        if test_status not in {"found", "none_found"} or not search_notes:
            raise CandidateError(f"{condition_id}: invalid related_tests status/notes")
        raw_indexes = raw_tests.get("evidence_indexes", [])
        if not isinstance(raw_indexes, list):
            raise CandidateError(
                f"{condition_id}: related_tests.evidence_indexes must be a list"
            )
        test_ids: list[str] = []
        for raw_index in raw_indexes:
            try:
                index = int(raw_index)
            except (TypeError, ValueError) as exc:
                raise CandidateError(
                    f"{condition_id}: invalid related test evidence index"
                ) from exc
            if index < 1 or index > len(evidence):
                raise CandidateError(
                    f"{condition_id}: related test evidence index out of range"
                )
            item = evidence[index - 1]
            if item["evidence_type"] not in RELATED_TEST_EVIDENCE_TYPES:
                raise CandidateError(
                    f"{condition_id}: related test index does not reference "
                    "test/fixture/doctest/build-test evidence"
                )
            if item["evidence_id"] not in test_ids:
                test_ids.append(item["evidence_id"])
        if test_status == "found" and not test_ids:
            raise CandidateError(
                f"{condition_id}: tests marked found without test evidence"
            )
        if test_status == "none_found" and test_ids:
            raise CandidateError(
                f"{condition_id}: tests marked none_found with test evidence"
            )

        evidence_basis = list(
            dict.fromkeys(item["evidence_type"] for item in evidence)
        )
        normalized.append(
            {
                "condition_id": condition_id,
                "mapping_status": status,
                "coverage": coverage,
                "evidence_basis": evidence_basis,
                "mapping_explanation": explanation,
                "no_direct_mapping_reason": no_direct_reason,
                "evidence": evidence,
                "related_tests": {
                    "status": test_status,
                    "evidence_ids": test_ids,
                    "search_notes": search_notes,
                },
            }
        )
    return normalized


def normalize_holistic_shard_output(
    data: dict[str, Any],
    *,
    shard_id: str,
    deleted_parts: Sequence[dict[str, Any]],
    repo_root: Path,
    allowed_snippets: Sequence[Any],
) -> dict[str, Any]:
    if str(data.get("shard_id") or "").strip() != shard_id:
        raise CandidateError(f"{shard_id}: shard_id mismatch")
    raw_findings = data.get("condition_findings")
    if not isinstance(raw_findings, list):
        raise CandidateError(f"{shard_id}: condition_findings must be a list")
    expected_ids = [str(part["condition_id"]) for part in deleted_parts]
    actual_ids = [
        str(item.get("condition_id") or "").strip()
        for item in raw_findings
        if isinstance(item, dict)
    ]
    if len(actual_ids) != len(raw_findings) or actual_ids != expected_ids:
        raise CandidateError(
            f"{shard_id}: condition IDs/order mismatch: "
            f"expected {expected_ids}, got {actual_ids}"
        )

    findings: list[dict[str, Any]] = []
    for raw_finding in raw_findings:
        condition_id = str(raw_finding["condition_id"])
        relevant = raw_finding.get("relevant")
        summary = str(raw_finding.get("summary") or "").strip()
        if not isinstance(relevant, bool):
            raise CandidateError(
                f"{shard_id}/{condition_id}: relevant must be a boolean"
            )
        if not summary:
            raise CandidateError(
                f"{shard_id}/{condition_id}: summary must be non-empty"
            )
        evidence = normalize_evidence_items(
            raw_finding.get("evidence"),
            condition_id=condition_id,
            repo_root=repo_root,
            evidence_prefix=None,
        )
        enforce_evidence_within_shard(
            evidence,
            allowed_snippets,
            scope=f"{shard_id}/{condition_id}",
        )
        if relevant != bool(evidence):
            raise CandidateError(
                f"{shard_id}/{condition_id}: relevant must agree with evidence"
            )
        findings.append(
            {
                "condition_id": condition_id,
                "relevant": relevant,
                "summary": summary,
                "evidence": evidence,
            }
        )
    return {"shard_id": shard_id, "condition_findings": findings}


def normalize_holistic_aggregation_output(
    data: dict[str, Any],
    *,
    deleted_parts: Sequence[dict[str, Any]],
    repo_root: Path,
    shard_results: Sequence[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    alignment_summary = str(data.get("alignment_summary") or "").strip()
    if not alignment_summary:
        raise CandidateError("holistic alignment_summary must be non-empty")
    mappings = normalize_mapping_output(
        data,
        deleted_parts,
        repo_root,
        evidence_prefix="ha",
    )
    enforce_aggregated_mappings_from_shards(
        mappings,
        shard_results,
        scope="holistic_alignment",
    )
    return alignment_summary, mappings


def normalize_condition_aggregation_output(
    data: dict[str, Any],
    *,
    deleted_part: dict[str, Any],
    repo_root: Path,
    shard_results: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Normalize route A's final mapping and bind it to its shard findings."""

    mappings = normalize_mapping_output(
        data,
        [deleted_part],
        repo_root,
        evidence_prefix="cbc",
    )
    enforce_aggregated_mappings_from_shards(
        mappings,
        shard_results,
        scope="condition_by_condition",
    )
    return mappings


def enforce_mappings_within_repository(
    mappings: Sequence[dict[str, Any]],
    repository_snippets: Sequence[Any],
    *,
    scope: str,
) -> None:
    """Bind a final quality-audit result to the complete deterministic index."""

    for mapping in mappings:
        enforce_evidence_within_shard(
            mapping["evidence"],
            repository_snippets,
            scope=f"{scope}/{mapping['condition_id']}",
        )


def explicit_backticked_code_terms(
    deleted_part: dict[str, Any],
) -> set[str]:
    """Extract explicit code identifiers/literals that full evidence must cover."""

    texts = [
        str(deleted_part.get("source_text") or ""),
        str(deleted_part.get("normalized_condition") or ""),
        *[
            str(span.get("exact_text") or "")
            for span in deleted_part.get("deleted_spans", [])
            if isinstance(span, dict)
        ],
    ]
    terms: set[str] = set()
    for text in texts:
        for block in re.findall(r"`([^`\n]+)`", text):
            if re.fullmatch(
                r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+",
                block,
            ):
                terms.add(block)
            else:
                terms.update(
                    re.findall(r"[A-Za-z_][A-Za-z0-9_]*", block)
                )
    return terms


def enforce_full_mapping_covers_explicit_code_terms(
    mappings: Sequence[dict[str, Any]],
    deleted_parts: Sequence[dict[str, Any]],
    *,
    scope: str,
) -> None:
    """Reject a full direct mapping that omits named code-level subclauses."""

    parts_by_id = {
        str(part["condition_id"]): part for part in deleted_parts
    }
    missing_messages: list[str] = []
    for mapping in mappings:
        if (
            mapping["mapping_status"] != "direct"
            or mapping["coverage"] != "full"
        ):
            continue
        condition_id = str(mapping["condition_id"])
        terms = explicit_backticked_code_terms(parts_by_id[condition_id])
        implementation_corpus_parts: list[str] = []
        implementation_symbols: set[str] = set()
        for evidence in mapping["evidence"]:
            if (
                evidence["evidence_type"] != "implementation"
                or evidence["relation"] == "contradicts"
            ):
                continue
            for location in evidence["locations"]:
                qualified_name = str(
                    location["symbol"].get("qualified_name") or ""
                )
                if qualified_name:
                    implementation_symbols.add(qualified_name)
                implementation_corpus_parts.extend(
                    [
                        str(location["file_path"]),
                        qualified_name,
                        str(location["excerpt"]),
                    ]
                )
        corpus = "\n".join(implementation_corpus_parts)
        missing = sorted(term for term in terms if term not in corpus)
        if missing:
            missing_messages.append(
                f"{condition_id}: " + ", ".join(missing)
            )
            continue
        condition_text = "\n".join(
            [
                str(parts_by_id[condition_id].get("source_text") or ""),
                str(
                    parts_by_id[condition_id].get(
                        "normalized_condition"
                    )
                    or ""
                ),
                *[
                    str(span.get("exact_text") or "")
                    for span in parts_by_id[condition_id].get(
                        "deleted_spans", []
                    )
                    if isinstance(span, dict)
                ],
            ]
        ).casefold()
        is_scope_boundary = (
            "top-level" in condition_text
            and (
                "not propagated" in condition_text
                or "do not inherit" in condition_text
                or "only" in condition_text
            )
        )
        if is_scope_boundary:
            hook_terms = sorted(term for term in terms if term.startswith("__"))
            single_occurrence = [
                term for term in hook_terms if corpus.count(term) < 2
            ]
            if single_occurrence:
                raise CandidateError(
                    f"{scope}/{condition_id}: coverage=full for a top-level "
                    "scope boundary must show each hook at both its accepted "
                    "interface and top-level forwarding/consumption site; "
                    "insufficient evidence for: "
                    + ", ".join(single_occurrence)
                )
            if len(implementation_symbols) < 2:
                raise CandidateError(
                    f"{scope}/{condition_id}: coverage=full for a top-level "
                    "scope boundary must include both the outer-model and "
                    "recursive/nested implementation symbols"
                )
    if missing_messages:
        raise CandidateError(
            f"{scope}: coverage=full omits implementation evidence for "
            "explicit code terms from the deleted requirement(s): "
            + "; ".join(missing_messages)
            + ". Add production-code locations whose excerpts literally "
            "contain these terms and show their classifier/routing/handling "
            "logic (an import line or explanation text is insufficient), "
            "or downgrade coverage to partial and explain the gap."
        )


def enforce_explanation_line_references_are_evidenced(
    mappings: Sequence[dict[str, Any]],
    *,
    scope: str,
) -> None:
    """Require every explicit code-line claim to have a recorded location."""

    line_reference = re.compile(
        r"\blines?\s+(\d+)(?:\s*[-–—]\s*(\d+))?",
        flags=re.IGNORECASE,
    )
    for mapping in mappings:
        condition_id = str(mapping["condition_id"])
        covered_ranges = [
            (
                int(line_range["start"]),
                int(line_range["end"]),
            )
            for evidence in mapping["evidence"]
            for location in evidence["locations"]
            for line_range in location["line_ranges"]
        ]
        explanation_texts = [
            str(mapping["mapping_explanation"]),
            *[
                str(evidence["explanation"])
                for evidence in mapping["evidence"]
            ],
        ]
        for explanation in explanation_texts:
            for match in line_reference.finditer(explanation):
                start = int(match.group(1))
                end = int(match.group(2) or start)
                if end < start or not any(
                    covered_start <= start
                    and end <= covered_end
                    for covered_start, covered_end in covered_ranges
                ):
                    raise CandidateError(
                        f"{scope}/{condition_id}: explanation cites code "
                        f"line(s) {start}-{end} but no evidence location covers "
                        "that range. Add the cited production/test location or "
                        "remove the unsupported line claim."
                    )


def normalize_condition_quality_audit_output(
    data: dict[str, Any],
    *,
    deleted_part: dict[str, Any],
    repo_root: Path,
    repository_snippets: Sequence[Any],
) -> list[dict[str, Any]]:
    mappings = normalize_mapping_output(
        data,
        [deleted_part],
        repo_root,
        evidence_prefix="cbc",
    )
    enforce_full_mapping_covers_explicit_code_terms(
        mappings,
        [deleted_part],
        scope="condition_by_condition_quality_audit",
    )
    enforce_explanation_line_references_are_evidenced(
        mappings,
        scope="condition_by_condition_quality_audit",
    )
    enforce_mappings_within_repository(
        mappings,
        repository_snippets,
        scope="condition_by_condition_quality_audit",
    )
    return mappings


def normalize_holistic_quality_audit_output(
    data: dict[str, Any],
    *,
    deleted_parts: Sequence[dict[str, Any]],
    repo_root: Path,
    repository_snippets: Sequence[Any],
) -> tuple[str, list[dict[str, Any]]]:
    alignment_summary = str(data.get("alignment_summary") or "").strip()
    if not alignment_summary:
        raise CandidateError(
            "holistic quality-audit alignment_summary must be non-empty"
        )
    mappings = normalize_mapping_output(
        data,
        deleted_parts,
        repo_root,
        evidence_prefix="ha",
    )
    enforce_full_mapping_covers_explicit_code_terms(
        mappings,
        deleted_parts,
        scope="holistic_alignment_quality_audit",
    )
    enforce_explanation_line_references_are_evidenced(
        mappings,
        scope="holistic_alignment_quality_audit",
    )
    enforce_mappings_within_repository(
        mappings,
        repository_snippets,
        scope="holistic_alignment_quality_audit",
    )
    return alignment_summary, mappings


def render_repository_shards(
    snippets: Sequence[Any],
    *,
    max_chars: int,
) -> list[dict[str, Any]]:
    """Return complete formatted shards plus their exact source-snippet groups."""

    if max_chars < 4_000:
        raise CandidateError("--holistic-shard-chars must be at least 4000")
    if not snippets:
        return [
            {
                "context": (
                    "[repository index contains no eligible UTF-8 text snippets]"
                ),
                "snippets": (),
            }
        ]

    # Estimate packing cost from independently formatted blocks.  A reserve
    # absorbs the longer evidence ordinal labels used when a group is rendered.
    individual_lengths: list[int] = []
    for snippet in snippets:
        rendered = format_snippets_for_llm(
            [snippet],
            include_line_numbers=True,
            max_lines_per_snippet=100_000,
            max_chars=2_000_000_000,
        )
        if not rendered:
            raise CandidateError("repository formatter dropped an indexed snippet")
        if len(rendered) > max_chars:
            raise CandidateError(
                "one repository snippet exceeds --holistic-shard-chars "
                f"({len(rendered)} > {max_chars}); increase the explicit limit"
            )
        individual_lengths.append(len(rendered))

    groups: list[list[Any]] = []
    current: list[Any] = []
    estimated = 0
    soft_limit = max(1, max_chars - 2_048)
    for snippet, rendered_length in zip(snippets, individual_lengths):
        added = rendered_length + (2 if current else 0)
        if current and estimated + added > soft_limit:
            groups.append(current)
            current = []
            estimated = 0
            added = rendered_length
        current.append(snippet)
        estimated += added
    if current:
        groups.append(current)

    def render_group(group: Sequence[Any]) -> list[dict[str, Any]]:
        rendered = format_snippets_for_llm(
            group,
            include_line_numbers=True,
            max_lines_per_snippet=100_000,
            max_chars=2_000_000_000,
        )
        if len(rendered) <= max_chars:
            return [{"context": rendered, "snippets": tuple(group)}]
        if len(group) == 1:
            raise CandidateError(
                "one formatted repository snippet exceeds the holistic shard "
                f"budget ({len(rendered)} > {max_chars})"
            )
        midpoint = len(group) // 2
        return [
            *render_group(group[:midpoint]),
            *render_group(group[midpoint:]),
        ]

    shards: list[dict[str, Any]] = []
    for group in groups:
        shards.extend(render_group(group))
    if not shards or any(not shard["context"] for shard in shards):
        raise CandidateError("holistic repository sharding produced an empty shard")
    return shards


def call_candidate_stage(
    *,
    stage: str,
    base_messages: list[dict[str, str]],
    normalize: Any,
    run_dir: Path,
    args: argparse.Namespace,
    provider: str,
    model: str,
    base_url: str,
    api_key: str,
) -> Any:
    stage_dir = run_dir / stage
    if args.resume and stage_dir.is_dir():
        for validation_path in sorted(
            stage_dir.glob("attempt_*_validation.json")
        ):
            try:
                validation = json.loads(
                    validation_path.read_text(encoding="utf-8")
                )
                if validation.get("valid") is not True:
                    continue
                raw_path = validation_path.with_name(
                    validation_path.name.replace(
                        "_validation.json", "_raw.txt"
                    )
                )
                value = normalize(
                    parse_json_object(raw_path.read_text(encoding="utf-8"))
                )
            except (OSError, ValueError, TypeError, KeyError, CandidateError):
                continue
            print(f"[resume] reusing validated stage {stage_dir}", flush=True)
            return value
    # A work directory may be reused after an interrupted/rejected generation.
    # Remove only this exact stage so stale attempts from an older condition
    # set cannot be mistaken for the current audit trail.
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    messages = base_messages
    prior_output = ""
    errors: list[str] = []
    for attempt in range(1, args.retries + 1):
        write_json(
            stage_dir / f"attempt_{attempt:02d}_prompt.json",
            {"messages": messages},
        )
        try:
            prior_output = call_chat_completion(
                messages=messages,
                provider=provider,
                model=model,
                base_url=base_url,
                api_key=api_key,
                args=args,
                attempts=1,
                telemetry_path=(
                    stage_dir / f"attempt_{attempt:02d}_api.json"
                ),
            )
            (stage_dir / f"attempt_{attempt:02d}_raw.txt").write_text(
                prior_output, encoding="utf-8"
            )
            parsed = parse_json_object(prior_output)
            value = normalize(parsed)
            write_json(
                stage_dir / f"attempt_{attempt:02d}_validation.json",
                {"valid": True, "errors": []},
            )
            return value
        except BatchAbortError as exc:
            write_json(
                stage_dir / f"attempt_{attempt:02d}_validation.json",
                {"valid": False, "errors": [str(exc)]},
            )
            raise
        except Exception as exc:  # noqa: BLE001 - persist candidate diagnostics.
            errors = [str(exc)]
            write_json(
                stage_dir / f"attempt_{attempt:02d}_validation.json",
                {"valid": False, "errors": errors},
            )
            if attempt >= args.retries:
                break
            messages = repair_prompt(
                base_messages, prior_output or "{}", errors, stage
            )
            time.sleep(min(2**attempt, 10))
    raise CandidateError(
        f"{stage} failed after {args.retries} attempts: {'; '.join(errors)}"
    )


def generate_condition_by_condition_mappings(
    *,
    row: dict[str, Any],
    deleted_parts: Sequence[dict[str, Any]],
    repo_root: Path,
    repository_shards: Sequence[dict[str, Any]],
    repository_scan: dict[str, int],
    run_dir: Path,
    args: argparse.Namespace,
    provider: str,
    model: str,
    base_url: str,
    api_key: str,
) -> list[dict[str, Any]]:
    """Independently scan the complete repository for each single condition."""

    route_dir = run_dir / "condition_code_mapping" / "condition_by_condition"
    route_dir.mkdir(parents=True, exist_ok=True)
    repository_context = complete_repository_context(repository_shards)
    repository_snippets = tuple(
        snippet
        for repository_shard in repository_shards
        for snippet in repository_shard["snippets"]
    )
    mappings: list[dict[str, Any]] = []
    for deleted_part in deleted_parts:
        condition_id = str(deleted_part["condition_id"])
        condition_dir = route_dir / condition_id
        condition_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            condition_dir / "scan_metadata.json",
            {
                "condition_id": condition_id,
                "input_scope": "single_deleted_condition_complete_repository",
                **repository_scan,
                "shard_char_budget": args.holistic_shard_chars,
                "aggregate_char_limit": args.max_holistic_aggregate_chars,
                "complete_index_scanned": True,
            },
        )

        shard_results: list[dict[str, Any]] = []
        shards_dir = condition_dir / "shards"
        for shard_index, repository_shard in enumerate(
            repository_shards, start=1
        ):
            shard_id = f"shard_{shard_index:03d}"
            shard_dir = shards_dir / shard_id
            shard_dir.mkdir(parents=True, exist_ok=True)
            repo_context = str(repository_shard["context"])
            allowed_snippets = tuple(repository_shard["snippets"])
            (shard_dir / "repository_evidence.txt").write_text(
                repo_context, encoding="utf-8"
            )
            write_json(
                shard_dir / "scan_metadata.json",
                {
                    "condition_id": condition_id,
                    "shard_id": shard_id,
                    "shard_number": shard_index,
                    "shard_count": len(repository_shards),
                    "repository_context_chars": len(repo_context),
                    "indexed_snippets_in_shard": len(allowed_snippets),
                    "contains_other_conditions": False,
                    "contains_complete_document": False,
                },
            )
            messages = condition_shard_prompt(
                row,
                deleted_part,
                shard_id=shard_id,
                shard_count=len(repository_shards),
                repo_context=repo_context,
            )
            result = call_candidate_stage(
                stage="ai_scan",
                base_messages=messages,
                normalize=(
                    lambda value,
                    expected_shard_id=shard_id,
                    part=deleted_part,
                    allowed=allowed_snippets: normalize_holistic_shard_output(
                        value,
                        shard_id=expected_shard_id,
                        deleted_parts=[part],
                        repo_root=repo_root,
                        allowed_snippets=allowed,
                    )
                ),
                run_dir=shard_dir,
                args=args,
                provider=provider,
                model=model,
                base_url=base_url,
                api_key=api_key,
            )
            write_json(shard_dir / "normalized_evidence.json", result)
            shard_results.append(result)

        messages = condition_aggregation_prompt(
            row,
            deleted_part,
            shard_results,
        )
        aggregation_chars = sum(
            len(message.get("content", "")) for message in messages
        )
        write_json(
            condition_dir / "aggregation_input_metadata.json",
            {
                "condition_id": condition_id,
                "prompt_chars": aggregation_chars,
                "prompt_char_limit": args.max_holistic_aggregate_chars,
                "shard_results": len(shard_results),
                "contains_other_conditions": False,
                "contains_complete_document": False,
                "contains_holistic_alignment_output": False,
            },
        )
        if aggregation_chars > args.max_holistic_aggregate_chars:
            raise CandidateError(
                f"{condition_id}: complete per-condition aggregation prompt "
                "exceeds the explicit limit "
                f"({aggregation_chars} > {args.max_holistic_aggregate_chars}); "
                "nothing was truncated. Increase "
                "--max-holistic-aggregate-chars."
            )
        normalized = call_candidate_stage(
            stage="aggregation",
            base_messages=messages,
            normalize=lambda value, part=deleted_part: (
                normalize_condition_aggregation_output(
                    value,
                    deleted_part=part,
                    repo_root=repo_root,
                    shard_results=shard_results,
                )
            ),
            run_dir=condition_dir,
            args=args,
            provider=provider,
            model=model,
            base_url=base_url,
            api_key=api_key,
        )
        if len(normalized) != 1:
            raise CandidateError(
                f"{condition_id}: isolated mapping did not return exactly one item"
            )
        write_json(
            condition_dir / "pre_quality_audit_normalized_evidence.json",
            normalized[0],
        )

        quality_messages = condition_quality_audit_prompt(
            row,
            deleted_part,
            normalized[0],
            repo_context=repository_context,
        )
        quality_chars = sum(
            len(message.get("content", "")) for message in quality_messages
        )
        write_json(
            condition_dir / "quality_audit_input_metadata.json",
            {
                "condition_id": condition_id,
                "prompt_chars": quality_chars,
                "prompt_char_limit": args.max_holistic_aggregate_chars,
                "repository_shards": len(repository_shards),
                "repository_context_chars": len(repository_context),
                "contains_other_conditions": False,
                "contains_complete_document": False,
                "contains_holistic_alignment_output": False,
                "rechecks_current_route_draft": True,
            },
        )
        if quality_chars > args.max_holistic_aggregate_chars:
            raise CandidateError(
                f"{condition_id}: complete per-condition quality-audit prompt "
                "exceeds the explicit limit "
                f"({quality_chars} > {args.max_holistic_aggregate_chars}); "
                "nothing was truncated. Increase "
                "--max-holistic-aggregate-chars."
            )
        audited = call_candidate_stage(
            stage="quality_audit",
            base_messages=quality_messages,
            normalize=lambda value, part=deleted_part: (
                normalize_condition_quality_audit_output(
                    value,
                    deleted_part=part,
                    repo_root=repo_root,
                    repository_snippets=repository_snippets,
                )
            ),
            run_dir=condition_dir,
            args=args,
            provider=provider,
            model=model,
            base_url=base_url,
            api_key=api_key,
        )
        if len(audited) != 1:
            raise CandidateError(
                f"{condition_id}: quality audit did not return exactly one item"
            )
        write_json(condition_dir / "normalized_evidence.json", audited[0])
        mappings.append(audited[0])
    return mappings


def generate_holistic_alignment(
    *,
    row: dict[str, Any],
    document_before: str,
    deleted_parts: Sequence[dict[str, Any]],
    repo_root: Path,
    repository_shards: Sequence[dict[str, Any]],
    repository_scan: dict[str, int],
    run_dir: Path,
    args: argparse.Namespace,
    provider: str,
    model: str,
    base_url: str,
    api_key: str,
) -> tuple[str, list[dict[str, Any]], dict[str, int]]:
    """Run the independent full-repository scan and complete-doc aggregation."""

    route_dir = run_dir / "condition_code_mapping" / "holistic_alignment"
    route_dir.mkdir(parents=True, exist_ok=True)
    repository_context = complete_repository_context(repository_shards)
    repository_snippets = tuple(
        snippet
        for repository_shard in repository_shards
        for snippet in repository_shard["snippets"]
    )
    write_json(
        route_dir / "scan_metadata.json",
        {
            **repository_scan,
            "shard_char_budget": args.holistic_shard_chars,
            "aggregate_char_limit": args.max_holistic_aggregate_chars,
            "complete_index_scanned": True,
        },
    )

    shard_results: list[dict[str, Any]] = []
    shards_dir = route_dir / "shards"
    for shard_index, repository_shard in enumerate(
        repository_shards, start=1
    ):
        shard_id = f"shard_{shard_index:03d}"
        shard_dir = shards_dir / shard_id
        shard_dir.mkdir(parents=True, exist_ok=True)
        repo_context = str(repository_shard["context"])
        allowed_snippets = tuple(repository_shard["snippets"])
        (shard_dir / "repository_evidence.txt").write_text(
            repo_context, encoding="utf-8"
        )
        write_json(
            shard_dir / "scan_metadata.json",
            {
                "shard_id": shard_id,
                "shard_number": shard_index,
                "shard_count": len(repository_shards),
                "repository_context_chars": len(repo_context),
                "indexed_snippets_in_shard": len(allowed_snippets),
                "contains_complete_document": True,
                "document_before_chars": len(document_before),
                "selected_condition_ids": [
                    part["condition_id"] for part in deleted_parts
                ],
            },
        )
        messages = holistic_shard_prompt(
            row,
            document_before,
            deleted_parts,
            shard_id=shard_id,
            shard_count=len(repository_shards),
            repo_context=repo_context,
        )
        result = call_candidate_stage(
            stage="ai_scan",
            base_messages=messages,
            normalize=lambda value, expected_shard_id=shard_id, allowed=allowed_snippets: (
                normalize_holistic_shard_output(
                    value,
                    shard_id=expected_shard_id,
                    deleted_parts=deleted_parts,
                    repo_root=repo_root,
                    allowed_snippets=allowed,
                )
            ),
            run_dir=shard_dir,
            args=args,
            provider=provider,
            model=model,
            base_url=base_url,
            api_key=api_key,
        )
        write_json(shard_dir / "normalized_evidence.json", result)
        shard_results.append(result)

    aggregation_messages = holistic_aggregation_prompt(
        row,
        document_before,
        deleted_parts,
        shard_results,
    )
    aggregation_chars = sum(
        len(message.get("content", "")) for message in aggregation_messages
    )
    write_json(
        route_dir / "aggregation_input_metadata.json",
        {
            "prompt_chars": aggregation_chars,
            "prompt_char_limit": args.max_holistic_aggregate_chars,
            "document_before_chars": len(document_before),
            "shard_results": len(shard_results),
            "contains_condition_by_condition_output": False,
        },
    )
    if aggregation_chars > args.max_holistic_aggregate_chars:
        raise CandidateError(
            "complete holistic aggregation prompt exceeds the explicit limit "
            f"({aggregation_chars} > {args.max_holistic_aggregate_chars}); "
            "nothing was truncated. Increase --max-holistic-aggregate-chars "
            "or reduce per-shard findings."
        )
    alignment_summary, mappings = call_candidate_stage(
        stage="aggregation",
        base_messages=aggregation_messages,
        normalize=lambda value: normalize_holistic_aggregation_output(
            value,
            deleted_parts=deleted_parts,
            repo_root=repo_root,
            shard_results=shard_results,
        ),
        run_dir=route_dir,
        args=args,
        provider=provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
    )
    write_json(
        route_dir / "pre_quality_audit_normalized_evidence.json",
        {
            "alignment_summary": alignment_summary,
            "code_mappings": mappings,
        },
    )

    quality_messages = holistic_quality_audit_prompt(
        row,
        document_before,
        deleted_parts,
        alignment_summary,
        mappings,
        repo_context=repository_context,
    )
    quality_chars = sum(
        len(message.get("content", "")) for message in quality_messages
    )
    write_json(
        route_dir / "quality_audit_input_metadata.json",
        {
            "prompt_chars": quality_chars,
            "prompt_char_limit": args.max_holistic_aggregate_chars,
            "document_before_chars": len(document_before),
            "repository_shards": len(repository_shards),
            "repository_context_chars": len(repository_context),
            "contains_condition_by_condition_output": False,
            "rechecks_current_route_draft": True,
        },
    )
    if quality_chars > args.max_holistic_aggregate_chars:
        raise CandidateError(
            "complete holistic quality-audit prompt exceeds the explicit limit "
            f"({quality_chars} > {args.max_holistic_aggregate_chars}); "
            "nothing was truncated. Increase "
            "--max-holistic-aggregate-chars."
        )
    alignment_summary, mappings = call_candidate_stage(
        stage="quality_audit",
        base_messages=quality_messages,
        normalize=lambda value: normalize_holistic_quality_audit_output(
            value,
            deleted_parts=deleted_parts,
            repo_root=repo_root,
            repository_snippets=repository_snippets,
        ),
        run_dir=route_dir,
        args=args,
        provider=provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
    )
    write_json(
        route_dir / "normalized_evidence.json",
        {
            "alignment_summary": alignment_summary,
            "code_mappings": mappings,
        },
    )
    return alignment_summary, mappings, repository_scan


def build_code_mapping_bundle(
    *,
    deleted_parts: Sequence[dict[str, Any]],
    condition_by_condition: Sequence[dict[str, Any]],
    holistic_alignment: Sequence[dict[str, Any]],
    alignment_summary: str,
    repository_scan: dict[str, int],
) -> dict[str, Any]:
    expected_ids = [str(part["condition_id"]) for part in deleted_parts]
    cbc_ids = [str(mapping["condition_id"]) for mapping in condition_by_condition]
    holistic_ids = [str(mapping["condition_id"]) for mapping in holistic_alignment]
    if cbc_ids != expected_ids or holistic_ids != expected_ids:
        raise CandidateError(
            "dual mapping routes do not match selected condition IDs/order: "
            f"expected={expected_ids}, condition_by_condition={cbc_ids}, "
            f"holistic_alignment={holistic_ids}"
        )
    return {
        "methods": {
            "condition_by_condition": {
                "mode": "one_condition_full_repository_sharded_alignment",
                "input_scope": "single_deleted_condition_complete_repository",
                "repository_scan": {
                    "indexed_files": int(repository_scan["indexed_files"]),
                    "indexed_snippets": int(repository_scan["indexed_snippets"]),
                    "shards": int(repository_scan["shards"]),
                },
            },
            "holistic_alignment": {
                "mode": "complete_document_repo_wide_alignment",
                "input_scope": (
                    "complete_document_all_selected_conditions_and_repository"
                ),
                "alignment_summary": alignment_summary,
                "repository_scan": {
                    "indexed_files": int(repository_scan["indexed_files"]),
                    "indexed_snippets": int(repository_scan["indexed_snippets"]),
                    "shards": int(repository_scan["shards"]),
                },
            },
        },
        "comparison_items": [
            {
                "condition_id": condition_id,
                "condition_by_condition": condition_by_condition[index],
                "holistic_alignment": holistic_alignment[index],
                "manual_comparison": {"agreement": None, "notes": ""},
            }
            for index, condition_id in enumerate(expected_ids)
        ],
    }


def generate_dual_mappings(
    *,
    row: dict[str, Any],
    document_before: str,
    deleted_parts: Sequence[dict[str, Any]],
    repo_root: Path,
    snippets: Sequence[Any],
    run_dir: Path,
    args: argparse.Namespace,
    provider: str,
    model: str,
    base_url: str,
    api_key: str,
) -> dict[str, Any]:
    """Generate two independent mappings and package them for manual comparison."""

    mapping_run_dir = run_dir / "condition_code_mapping"
    if mapping_run_dir.exists():
        shutil.rmtree(mapping_run_dir)
    repository_shards = render_repository_shards(
        snippets,
        max_chars=args.holistic_shard_chars,
    )
    repository_scan = {
        "indexed_files": len({snippet.file_path for snippet in snippets}),
        "indexed_snippets": len(snippets),
        "shards": len(repository_shards),
    }
    write_json(
        mapping_run_dir / "repository_index.json",
        {
            **repository_scan,
            "shard_index": [
                {
                    "shard_id": f"shard_{index:03d}",
                    "context_chars": len(str(shard["context"])),
                    "snippet_count": len(shard["snippets"]),
                    "snippets": [
                        {
                            "file_path": snippet.file_path,
                            "file_sha256": snippet.file_sha256,
                            "start_line": snippet.start_line,
                            "end_line": snippet.end_line,
                        }
                        for snippet in shard["snippets"]
                    ],
                }
                for index, shard in enumerate(repository_shards, start=1)
            ],
        },
    )

    condition_by_condition = generate_condition_by_condition_mappings(
        row=row,
        deleted_parts=deleted_parts,
        repo_root=repo_root,
        repository_shards=repository_shards,
        repository_scan=repository_scan,
        run_dir=run_dir,
        args=args,
        provider=provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
    )
    # Deliberately pass only source inputs and the deterministic repository
    # index to route B.  Route A's outputs never enter this call.
    alignment_summary, holistic_alignment, repository_scan = (
        generate_holistic_alignment(
            row=row,
            document_before=document_before,
            deleted_parts=deleted_parts,
            repo_root=repo_root,
            repository_shards=repository_shards,
            repository_scan=repository_scan,
            run_dir=run_dir,
            args=args,
            provider=provider,
            model=model,
            base_url=base_url,
            api_key=api_key,
        )
    )
    return build_code_mapping_bundle(
        deleted_parts=deleted_parts,
        condition_by_condition=condition_by_condition,
        holistic_alignment=holistic_alignment,
        alignment_summary=alignment_summary,
        repository_scan=repository_scan,
    )


def projection_payload(
    sample_id: str, field: str, value: Any
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "sample_id": sample_id,
        field: value,
    }


def write_sample_artifacts(
    *,
    row: dict[str, Any],
    sample_id: str,
    document_before: str,
    document_after: str,
    review_conditions: list[dict[str, Any]],
    deleted_parts: list[dict[str, Any]],
    selected_ids: list[str],
    code_mappings: dict[str, Any],
    stage_dir: Path,
) -> None:
    before_path = stage_dir / "1_document_before.md"
    deleted_path = stage_dir / "2_deleted_parts.json"
    after_path = stage_dir / "3_document_after.md"
    mapping_path = stage_dir / "4_code_mapping.json"
    selection_path = stage_dir / "condition_selection.json"
    test_patch_path = stage_dir / "evaluation_tests.patch"

    before_path.write_text(document_before, encoding="utf-8")
    after_path.write_text(document_after, encoding="utf-8")
    write_json(
        deleted_path,
        projection_payload(sample_id, "deleted_parts", deleted_parts),
    )
    write_json(
        mapping_path,
        projection_payload(sample_id, "code_mappings", code_mappings),
    )
    write_json(
        selection_path,
        {
            "schema_version": SCHEMA_VERSION,
            "sample_id": sample_id,
            "considered_conditions": review_conditions,
            "selected_condition_ids": selected_ids,
        },
    )

    test_patch = row.get("test_patch")
    if isinstance(test_patch, str) and test_patch:
        test_patch_path.write_text(test_patch, encoding="utf-8")


def ensure_artifact_consistency(
    stage_dir: Path,
    deleted_parts: Sequence[dict[str, Any]],
    code_mappings: dict[str, Any],
) -> None:
    deleted_projection = json.loads(
        (stage_dir / "2_deleted_parts.json").read_text(encoding="utf-8")
    )
    mapping_projection = json.loads(
        (stage_dir / "4_code_mapping.json").read_text(encoding="utf-8")
    )
    if deleted_projection.get("deleted_parts") != list(deleted_parts):
        raise CandidateError("deleted parts artifact differs from generated payload")
    if mapping_projection.get("code_mappings") != code_mappings:
        raise CandidateError("code mapping artifact differs from generated payload")
    before = (stage_dir / "1_document_before.md").read_text(encoding="utf-8")
    after = (stage_dir / "3_document_after.md").read_text(encoding="utf-8")
    rebuilt = delete_exact_spans(before, list(deleted_parts))
    if rebuilt != after:
        raise CandidateError("after document is not the exact deletion derivation")


def remove_snapshot_marker(repo_dir: Path, marker_path: str) -> None:
    """Remove the downloader sidecar before publishing a self-contained sample."""

    expected = repo_dir.parent / f".{repo_dir.name}.snapshot.json"
    actual = Path(marker_path)
    if actual.resolve(strict=False) != expected.resolve(strict=False):
        raise CandidateError(
            f"unexpected repository snapshot marker path: {actual}"
        )
    if actual.is_symlink() or not actual.is_file():
        raise CandidateError(
            f"repository snapshot marker is not a regular file: {actual}"
        )
    actual.unlink()


def atomic_publish(stage_dir: Path, destination: Path, overwrite: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        stage_dir.replace(destination)
        return
    if not overwrite:
        raise CandidateError(f"output already exists: {destination}")
    if destination.is_symlink() or not destination.is_dir():
        raise CandidateError(
            f"refusing to overwrite a non-directory output: {destination}"
        )

    backup_parent = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.overwrite-backup-",
            dir=destination.parent,
        )
    )
    backup = backup_parent / destination.name
    destination.replace(backup)
    try:
        stage_dir.replace(destination)
    except BaseException:
        if destination.exists():
            failed_new = backup_parent / f"{destination.name}.failed-new"
            destination.replace(failed_new)
        backup.replace(destination)
        shutil.rmtree(backup_parent, ignore_errors=True)
        raise
    shutil.rmtree(backup_parent)


def paths_overlap(first: Path, second: Path) -> bool:
    first_resolved = first.resolve()
    second_resolved = second.resolve()
    return (
        first_resolved == second_resolved
        or first_resolved in second_resolved.parents
        or second_resolved in first_resolved.parents
    )


def is_complete_sample(path: Path) -> bool:
    required = (
        "1_document_before.md",
        "2_deleted_parts.json",
        "3_document_after.md",
        "4_code_mapping.json",
        "5_original_repo",
    )
    return path.is_dir() and all((path / value).exists() for value in required)


def generate_one(
    row: dict[str, Any],
    *,
    output_dir: Path,
    work_dir: Path,
    args: argparse.Namespace,
    provider: str,
    model: str,
    base_url: str,
    api_key: str,
) -> Path:
    validate_source_row(row)
    sample_id = safe_sample_id(row["instance_id"])
    destination = output_dir / sample_id
    if args.resume and is_complete_sample(destination):
        print(f"[skip] {sample_id}: complete output exists", flush=True)
        return destination
    if destination.exists() and not args.overwrite_output:
        raise CandidateError(
            f"{sample_id}: output exists; use --resume or --overwrite-output"
        )

    sample_run_dir = work_dir / sample_id
    sample_run_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        sample_run_dir / "source_metadata.json",
        {
            key: row.get(key)
            for key in (
                "instance_id",
                "github_url",
                "parent_commit",
                "repo",
                "user",
                "pypi_name",
                "difficulty",
                "license_spdx_id",
                "image_url",
                "workdir",
                "test_files",
                "passed_ptp",
            )
        },
    )

    stage_parent = Path(
        tempfile.mkdtemp(prefix=f".{sample_id}.", dir=str(work_dir))
    )
    stage_dir = stage_parent / sample_id
    stage_dir.mkdir()
    try:
        document_before = str(row["document"])
        initial_condition_payload: dict[str, Any] | None = None
        if args.condition_seed_dir:
            seed_path = Path(args.condition_seed_dir) / f"{sample_id}.json"
            if not seed_path.is_file():
                raise CandidateError(
                    f"condition seed is missing for {sample_id}: {seed_path}"
                )
            seed_value = json.loads(seed_path.read_text(encoding="utf-8"))
            if not isinstance(seed_value, dict):
                raise CandidateError(f"condition seed is not an object: {seed_path}")
            initial_condition_payload = seed_value
            write_json(
                sample_run_dir / "condition_seed.json",
                initial_condition_payload,
            )
        base_condition_messages = condition_prompt(row)
        write_json(
            sample_run_dir / "condition_base_prompt.json",
            {"messages": base_condition_messages},
        )
        condition_result = generate_audited_conditions(
            row=row,
            run_dir=sample_run_dir,
            args=args,
            provider=provider,
            model=model,
            base_url=base_url,
            api_key=api_key,
            initial_condition_payload=initial_condition_payload,
        )
        (
            review_conditions,
            deleted_parts,
            selected_ids,
            annotation_notes,
            semantic_audit,
        ) = condition_result
        write_json(sample_run_dir / "semantic_audit.final.json", semantic_audit)
        document_after = delete_exact_spans(document_before, deleted_parts)

        repo_dir = stage_dir / "5_original_repo"
        print(f"[repo] {sample_id}: downloading pinned snapshot", flush=True)
        snapshot_info = download_github_snapshot(
            str(row["github_url"]),
            str(row["parent_commit"]),
            repo_dir,
            timeout=args.timeout,
        )
        if snapshot_info.marker_path is None:
            raise CandidateError("repository snapshot download did not return a marker")
        remove_snapshot_marker(repo_dir, snapshot_info.marker_path)
        repository_snippets = extract_repository_snippets(repo_dir)
        code_mappings = generate_dual_mappings(
            row=row,
            document_before=document_before,
            deleted_parts=deleted_parts,
            repo_root=repo_dir,
            snippets=repository_snippets,
            run_dir=sample_run_dir,
            args=args,
            provider=provider,
            model=model,
            base_url=base_url,
            api_key=api_key,
        )

        write_sample_artifacts(
            row=row,
            sample_id=sample_id,
            document_before=document_before,
            document_after=document_after,
            review_conditions=review_conditions,
            deleted_parts=deleted_parts,
            selected_ids=selected_ids,
            code_mappings=code_mappings,
            stage_dir=stage_dir,
        )
        ensure_artifact_consistency(
            stage_dir,
            deleted_parts,
            code_mappings,
        )
        strict_errors = validate_sample(stage_dir)
        if strict_errors:
            raise CandidateError(
                "strict sample validation failed:\n- "
                + "\n- ".join(strict_errors)
            )
        write_json(
            sample_run_dir / "run.json",
            {
                "status": "complete",
                "sample_id": sample_id,
                "output": str(destination),
                "generator": f"generate_specgap_v2.py/{GENERATOR_VERSION}",
                "model": model,
                "completed_at": utc_now(),
            },
        )
        atomic_publish(stage_dir, destination, args.overwrite_output)
        stage_parent.rmdir()
        print(
            f"[ok] {sample_id}: {len(deleted_parts)} conditions, "
            f"{sum(len(item['condition_by_condition']['evidence']) + len(item['holistic_alignment']['evidence']) for item in code_mappings['comparison_items'])} "
            "cross-checked evidence items",
            flush=True,
        )
        return destination
    except BatchAbortError:
        # Account/authentication failures affect the whole batch rather than
        # the quality of this candidate.  Leave the run unclassified so a
        # later --resume can retry it after the account issue is resolved.
        shutil.rmtree(stage_parent, ignore_errors=True)
        raise
    except Exception as exc:
        batch_abort_event = getattr(args, "batch_abort_event", None)
        if batch_abort_event is not None and batch_abort_event.is_set():
            shutil.rmtree(stage_parent, ignore_errors=True)
            raise BatchAbortError(
                "batch is stopping after an API account/authentication error"
            ) from exc
        write_json(
            sample_run_dir / "run.json",
            {
                "status": "failed",
                "sample_id": sample_id,
                "error": str(exc),
                "failed_at": utc_now(),
            },
        )
        if args.keep_failed_stage:
            failed_destination = sample_run_dir / "failed_stage"
            if failed_destination.exists():
                shutil.rmtree(failed_destination)
            if stage_dir.exists():
                stage_dir.replace(failed_destination)
        shutil.rmtree(stage_parent, ignore_errors=True)
        raise


def main() -> int:
    args = parse_args()
    args.batch_abort_event = threading.Event()
    if args.limit < 1:
        raise SystemExit("--limit must be positive")
    if args.target_successes is not None:
        if args.target_successes < 1:
            raise SystemExit("--target-successes must be positive")
        if args.target_successes > args.limit:
            raise SystemExit("--target-successes cannot exceed --limit")
    if args.retries < 1:
        raise SystemExit("--retries must be positive")
    if args.source_retries < 1:
        raise SystemExit("--source-retries must be positive")
    if args.source_timeout < 1:
        raise SystemExit("--source-timeout must be positive")
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    if args.skip_recorded_failures and not args.resume:
        raise SystemExit("--skip-recorded-failures requires --resume")
    if args.holistic_shard_chars < 4_000:
        raise SystemExit("--holistic-shard-chars must be at least 4000")
    if args.max_holistic_aggregate_chars < 10_000:
        raise SystemExit("--max-holistic-aggregate-chars must be at least 10000")
    project_root = Path(__file__).resolve().parents[1]
    load_env_file(project_root / ".env")
    load_env_file(Path.cwd() / ".env")
    provider = args.provider
    model = get_model(provider, args.model)
    base_url = get_base_url(provider, args.base_url)
    api_key = get_api_key(provider, args.api_key)
    output_dir = Path(args.output_dir)
    work_dir = Path(args.work_dir)
    if paths_overlap(output_dir, work_dir):
        raise SystemExit(
            "--output-dir and --work-dir must be disjoint directories; "
            "neither may contain the other"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    failures: list[tuple[str, str]] = []
    count = 0
    succeeded = 0
    processed_sample_ids: set[str] = set()
    results: list[tuple[str, str | None]] = []
    try:
        rows = load_or_create_source_cache(work_dir, args)
        target_successes = args.target_successes or len(rows)
        rows_to_process = rows

        if args.skip_recorded_failures:
            rows_to_process = []
            for row in rows:
                sample_id = safe_sample_id(row.get("instance_id"))
                run_path = work_dir / sample_id / "run.json"
                destination = output_dir / sample_id
                try:
                    run_record = (
                        json.loads(run_path.read_text(encoding="utf-8"))
                        if run_path.is_file()
                        else {}
                    )
                except (OSError, ValueError, TypeError):
                    run_record = {}
                status = run_record.get("status")
                if status == "complete" and is_complete_sample(destination):
                    count += 1
                    succeeded += 1
                    processed_sample_ids.add(sample_id)
                    results.append((sample_id, None))
                    print(
                        f"[resume] counted completed candidate {sample_id}",
                        flush=True,
                    )
                elif status == "failed":
                    count += 1
                    processed_sample_ids.add(sample_id)
                    error = str(
                        run_record.get("error")
                        or "recorded failure without an error message"
                    )
                    results.append((sample_id, error))
                    print(
                        f"[resume] skipped recorded failure {sample_id}",
                        flush=True,
                    )
                else:
                    rows_to_process.append(row)
            print(
                f"[resume] restored {succeeded} completions and "
                f"{len(results) - succeeded} failures; "
                f"{len(rows_to_process)} candidates remain eligible",
                flush=True,
            )

        def run_row(row: dict[str, Any]) -> tuple[str, str | None]:
            sample_id = safe_sample_id(row.get("instance_id"))
            try:
                if args.batch_abort_event.is_set():
                    raise BatchAbortError(
                        "batch is stopping after an API "
                        "account/authentication error"
                    )
                generate_one(
                    row,
                    output_dir=output_dir,
                    work_dir=work_dir,
                    args=args,
                    provider=provider,
                    model=model,
                    base_url=base_url,
                    api_key=api_key,
                )
                if args.sleep > 0:
                    time.sleep(args.sleep)
                return sample_id, None
            except BatchAbortError:
                raise
            except Exception as exc:  # noqa: BLE001 - report independent sample.
                return sample_id, str(exc)

        if args.workers == 1:
            for index, row in enumerate(rows_to_process, start=1):
                if succeeded >= target_successes:
                    break
                print(
                    f"[{index}/{args.limit}] {safe_sample_id(row.get('instance_id'))}",
                    flush=True,
                )
                result = run_row(row)
                results.append(result)
                count += 1
                processed_sample_ids.add(result[0])
                if result[1] is None:
                    succeeded += 1
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                row_iterator = iter(rows_to_process)
                future_to_id: dict[Any, str] = {}

                def fill_worker_slots() -> None:
                    remaining = target_successes - succeeded
                    desired_in_flight = min(args.workers, remaining)
                    while len(future_to_id) < desired_in_flight:
                        if args.batch_abort_event.is_set():
                            break
                        try:
                            row = next(row_iterator)
                        except StopIteration:
                            break
                        future = executor.submit(run_row, row)
                        future_to_id[future] = safe_sample_id(
                            row.get("instance_id")
                        )

                fill_worker_slots()
                while future_to_id:
                    completed_futures, _ = wait(
                        future_to_id,
                        return_when=FIRST_COMPLETED,
                    )
                    for future in completed_futures:
                        sample_id = future_to_id.pop(future)
                        count += 1
                        processed_sample_ids.add(sample_id)
                        print(
                            f"[progress] {count}/{args.limit}: {sample_id}",
                            flush=True,
                        )
                        try:
                            result = future.result()
                        except BatchAbortError:
                            for pending_future in future_to_id:
                                pending_future.cancel()
                            raise
                        except Exception as exc:  # defensive; run_row captures normally.
                            result = (sample_id, str(exc))
                        results.append(result)
                        if result[1] is None:
                            succeeded += 1
                    if succeeded >= target_successes:
                        break
                    fill_worker_slots()
        for sample_id, error in results:
            if error is not None:
                failures.append((sample_id, error))
                print(f"[error] {sample_id}: {error}", file=sys.stderr, flush=True)
    except BatchAbortError as exc:
        print(
            f"[fatal] API account/authentication error; batch stopped and "
            f"unfinished candidates remain retryable: {exc}",
            file=sys.stderr,
        )
        return 2
    except Exception as exc:  # source/network failure.
        print(f"[fatal] failed to read source rows: {exc}", file=sys.stderr)
        return 1

    try:
        manifest_records: dict[str, dict[str, Any]] = {}
        for row in rows:
            sample_id = safe_sample_id(row.get("instance_id"))
            if sample_id not in processed_sample_ids:
                continue
            sample_run_dir = work_dir / sample_id
            run_record: dict[str, Any] = {}
            run_path = sample_run_dir / "run.json"
            try:
                value = json.loads(run_path.read_text(encoding="utf-8"))
                if isinstance(value, dict):
                    run_record = value
            except (OSError, ValueError, TypeError):
                pass

            recorded_model = run_record.get("model")
            if not recorded_model and run_record.get("status") == "complete":
                for directory, child_dirs, child_files in os.walk(sample_run_dir):
                    child_dirs[:] = [
                        name
                        for name in child_dirs
                        if name != "5_original_repo" and not name.startswith(".")
                    ]
                    telemetry_names = sorted(
                        name
                        for name in child_files
                        if Path(name).match("attempt_*_api.json")
                    )
                    for telemetry_name in telemetry_names:
                        try:
                            telemetry = json.loads(
                                (Path(directory) / telemetry_name).read_text(
                                    encoding="utf-8"
                                )
                            )
                        except (OSError, ValueError, TypeError):
                            continue
                        if isinstance(telemetry, dict) and telemetry.get("model"):
                            recorded_model = str(telemetry["model"])
                            break
                    if recorded_model:
                        break

            manifest_records[sample_id] = {
                "github_url": str(row["github_url"]),
                "parent_commit": str(row["parent_commit"]).lower(),
                "generator": str(
                    run_record.get("generator")
                    or "generate_specgap_v2.py/2.7.0"
                ),
                "model": str(recorded_model or model),
                "generated_at": str(
                    run_record.get("completed_at") or utc_now()
                ),
            }
        manifest_path = rebuild_collection_manifest(
            output_dir,
            manifest_records,
        )
    except Exception as exc:
        print(
            f"[fatal] failed to rebuild qualified manifest: {exc}",
            file=sys.stderr,
        )
        return 1

    summary = {
        "schema_version": SCHEMA_VERSION,
        "requested": args.limit,
        "target_successes": args.target_successes,
        "seen": count,
        "succeeded": succeeded,
        "failed": [
            {"sample_id": sample_id, "error": error}
            for sample_id, error in failures
        ],
        "generated_at": utc_now(),
        "model": model,
        "manifest": str(manifest_path),
    }
    write_json(work_dir / "summary.json", summary)
    target_reached = (
        args.target_successes is not None
        and succeeded >= args.target_successes
    )
    if failures and not target_reached:
        print(
            f"[done] {succeeded}/{count} complete; "
            f"{len(failures)} failed (see {work_dir / 'summary.json'})",
            file=sys.stderr,
        )
        return 1
    if target_reached:
        print(
            f"[done] generated target {succeeded} samples from {count} candidates "
            f"under {output_dir}; {len(failures)} candidates failed"
        )
    else:
        print(f"[done] generated {count} samples under {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
