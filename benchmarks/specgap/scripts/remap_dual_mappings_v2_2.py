#!/usr/bin/env python3
"""Regenerate only the strict dual mappings of existing SpecGAP samples.

All AI calls finish before the qualified collection is changed.  The script
then builds and validates a complete staged collection and swaps it into place.
It never supplies an old mapping or the other route's output to either route.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

if __package__:
    from .generate_specgap_v2 import (
        GENERATOR_VERSION,
        SCHEMA_VERSION,
        enforce_explanation_line_references_are_evidenced,
        enforce_full_mapping_covers_explicit_code_terms,
        generate_dual_mappings,
        projection_payload,
        rebuild_collection_manifest,
        sha256_file,
        utc_now,
        write_json,
    )
    from .llm_source import (
        get_api_key,
        get_base_url,
        get_model,
        load_env_file,
    )
    from .repo_evidence import extract_repository_snippets, tree_sha256
    from .validate_specgap_v2 import (
        discover_samples,
        validate_collection_manifest,
        validate_sample,
    )
else:
    from generate_specgap_v2 import (  # type: ignore
        GENERATOR_VERSION,
        SCHEMA_VERSION,
        enforce_explanation_line_references_are_evidenced,
        enforce_full_mapping_covers_explicit_code_terms,
        generate_dual_mappings,
        projection_payload,
        rebuild_collection_manifest,
        sha256_file,
        utc_now,
        write_json,
    )
    from llm_source import (  # type: ignore
        get_api_key,
        get_base_url,
        get_model,
        load_env_file,
    )
    from repo_evidence import extract_repository_snippets, tree_sha256  # type: ignore
    from validate_specgap_v2 import (  # type: ignore
        discover_samples,
        validate_collection_manifest,
        validate_sample,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate two independent code-mapping routes for existing "
            "SpecGAP samples, then atomically replace the validated collection."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("SpecGAP"),
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("runs/qualified_builds/dual_mapping_v2_2"),
    )
    parser.add_argument(
        "--provider",
        choices=["deepseek", "glm", "openai_compatible"],
        default="deepseek",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--holistic-shard-chars", type=int, default=60_000)
    parser.add_argument(
        "--max-holistic-aggregate-chars",
        type=int,
        default=480_000,
    )
    parser.add_argument(
        "--reset-manual-comparison",
        action="store_true",
        help=(
            "Allow replacement of non-empty manual_comparison values. "
            "Without this flag the command stops before any AI call."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Reuse a completed, revalidated per-sample bundle already present "
            "in --work-dir and regenerate only incomplete samples."
        ),
    )
    return parser.parse_args(argv)


def read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def paths_overlap(first: Path, second: Path) -> bool:
    first_resolved = first.resolve()
    second_resolved = second.resolve()
    return (
        first_resolved == second_resolved
        or first_resolved in second_resolved.parents
        or second_resolved in first_resolved.parents
    )


def manual_comparison_is_filled(sample: dict[str, Any]) -> bool:
    bundle = sample.get("code_mappings")
    if not isinstance(bundle, dict):
        return False
    comparisons = bundle.get("comparison_items")
    if not isinstance(comparisons, list):
        return False
    for comparison in comparisons:
        if not isinstance(comparison, dict):
            continue
        manual = comparison.get("manual_comparison")
        if not isinstance(manual, dict):
            continue
        if manual.get("agreement") is not None:
            return True
        if str(manual.get("notes") or "").strip():
            return True
    return False


def validate_inputs_before_ai(
    sample_dirs: Sequence[Path],
    *,
    reset_manual_comparison: bool,
) -> None:
    """Reject damaged immutable artifacts before sending anything to an AI."""

    errors: list[str] = []
    if sample_dirs:
        errors.extend(
            f"collection: {error}"
            for error in validate_collection_manifest(
                sample_dirs[0].parent,
                sample_dirs,
            )
        )
    migration_error_prefixes = (
        "schema_version:",
        "2_deleted_parts.json.schema_version:",
        "4_code_mapping.json.schema_version:",
        "condition_selection.json.schema_version:",
        "code_mappings",
    )
    for sample_dir in sample_dirs:
        sample_errors = validate_sample(sample_dir)
        immutable_errors = [
            error
            for error in sample_errors
            if not error.startswith(migration_error_prefixes)
        ]
        errors.extend(
            f"{sample_dir.name}: {error}" for error in immutable_errors
        )
        mapping_artifact = read_json_object(
            sample_dir / "4_code_mapping.json"
        )
        if (
            manual_comparison_is_filled(
                {"code_mappings": mapping_artifact.get("code_mappings")}
            )
            and not reset_manual_comparison
        ):
            errors.append(
                f"{sample_dir.name}: manual_comparison contains user data; "
                "rerun with --reset-manual-comparison only if it may be cleared"
            )
    if errors:
        raise RuntimeError(
            "refusing to call the AI because remap inputs are unsafe:\n- "
            + "\n- ".join(errors)
        )


def archive_existing_work_dir(work_dir: Path) -> None:
    """Preserve an earlier run before writing a new audit trail."""

    if not work_dir.exists():
        work_dir.mkdir(parents=True, exist_ok=True)
        return
    existing = [path for path in work_dir.iterdir() if path.name != "history"]
    if not existing:
        return
    timestamp = utc_now().replace(":", "").replace("+", "_")
    archive = work_dir / "history" / timestamp
    suffix = 1
    while archive.exists():
        archive = work_dir / "history" / f"{timestamp}_{suffix:02d}"
        suffix += 1
    archive.mkdir(parents=True)
    for path in existing:
        shutil.move(str(path), archive / path.name)


def collect_sample_inputs(
    sample_dir: Path,
) -> tuple[dict[str, Any], str, list[dict[str, Any]], Path, dict[str, str]]:
    document_before_path = sample_dir / "1_document_before.md"
    document_before = document_before_path.read_text(encoding="utf-8")
    deleted_artifact = read_json_object(sample_dir / "2_deleted_parts.json")
    deleted_parts = deleted_artifact.get("deleted_parts")
    if not isinstance(deleted_parts, list) or not deleted_parts:
        raise ValueError(f"{sample_dir}: deleted_parts must be a non-empty list")
    repo_root = sample_dir / "5_original_repo"
    if not repo_root.is_dir() or repo_root.is_symlink():
        raise ValueError(f"{sample_dir}: original repository is unavailable")
    manifest = read_json_object(sample_dir.parent / "manifest.json")
    entries = manifest.get("samples")
    if not isinstance(entries, list):
        raise ValueError("collection manifest samples must be a list")
    entry = next(
        (
            item
            for item in entries
            if isinstance(item, dict)
            and item.get("sample_id") == sample_dir.name
        ),
        None,
    )
    if entry is None:
        raise ValueError(f"{sample_dir}: collection provenance is missing")
    row = {
        "github_url": str(entry["github_url"]),
        "parent_commit": str(entry["parent_commit"]),
    }
    input_snapshot = {
        "sample_id": sample_dir.name,
        "document_before_sha256": sha256_file(document_before_path),
        "deleted_parts": deleted_parts,
        "condition_selection_sha256": sha256_file(
            sample_dir / "condition_selection.json"
        ),
        "repository_tree_sha256": tree_sha256(repo_root),
        "github_url": row["github_url"],
        "parent_commit": row["parent_commit"],
    }
    return input_snapshot, document_before, deleted_parts, repo_root, row


def immutable_sample_view(sample: dict[str, Any]) -> dict[str, Any]:
    """Return the sample fields that a mapping-only resume may not change."""

    mutable = {"schema_version", "code_mappings", "annotation"}
    return {
        key: value for key, value in sample.items() if key not in mutable
    }


def stage_has_valid_attempt(stage_dir: Path) -> bool:
    for validation_path in sorted(stage_dir.glob("attempt_*_validation.json")):
        try:
            validation = read_json_object(validation_path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if validation.get("valid") is True:
            return True
    return False


def has_complete_quality_audit(
    sample_run_dir: Path,
    condition_ids: Sequence[str],
) -> bool:
    route_a = (
        sample_run_dir
        / "condition_code_mapping"
        / "condition_by_condition"
    )
    for condition_id in condition_ids:
        condition_dir = route_a / condition_id
        if not (condition_dir / "normalized_evidence.json").is_file():
            return False
        if not stage_has_valid_attempt(condition_dir / "quality_audit"):
            return False
    route_b = (
        sample_run_dir
        / "condition_code_mapping"
        / "holistic_alignment"
    )
    return (
        (route_b / "normalized_evidence.json").is_file()
        and stage_has_valid_attempt(route_b / "quality_audit")
    )


def load_reusable_bundle(
    sample_dir: Path,
    sample_run_dir: Path,
    current_sample: dict[str, Any],
    deleted_parts: Sequence[dict[str, Any]],
    *,
    model: str,
    work_dir: Path,
) -> dict[str, Any] | None:
    """Load a completed bundle only after current guards and validator pass."""

    previous_path = sample_run_dir / "previous_inputs.json"
    bundle_path = sample_run_dir / "dual_mapping_bundle.json"
    if not previous_path.is_file() or not bundle_path.is_file():
        return None
    try:
        previous_sample = read_json_object(previous_path)
        bundle = read_json_object(bundle_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if immutable_sample_view(previous_sample) != immutable_sample_view(
        current_sample
    ):
        return None
    condition_ids = [
        str(part["condition_id"]) for part in deleted_parts
    ]
    if not has_complete_quality_audit(sample_run_dir, condition_ids):
        return None
    try:
        comparisons = bundle["comparison_items"]
        if not isinstance(comparisons, list):
            return None
        for route in ("condition_by_condition", "holistic_alignment"):
            mappings = [item[route] for item in comparisons]
            enforce_full_mapping_covers_explicit_code_terms(
                mappings,
                deleted_parts,
                scope=f"resume/{route}",
            )
            enforce_explanation_line_references_are_evidenced(
                mappings,
                scope=f"resume/{route}",
            )
        with tempfile.TemporaryDirectory(
            prefix=".specgap-resume-check-",
            dir=work_dir,
        ) as temporary:
            staged_sample = Path(temporary) / sample_dir.name
            shutil.copytree(sample_dir, staged_sample)
            update_staged_sample(staged_sample, bundle, model=model)
            if validate_sample(staged_sample):
                return None
    except (KeyError, TypeError, ValueError, RuntimeError):
        return None
    return bundle


def generate_all_bundles(
    sample_dirs: Sequence[Path],
    *,
    work_dir: Path,
    args: argparse.Namespace,
    provider: str,
    model: str,
    base_url: str,
    api_key: str,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Finish every expensive call without changing the data collection."""

    bundles: dict[str, dict[str, Any]] = {}
    reused_sample_ids: list[str] = []
    for index, sample_dir in enumerate(sample_dirs, start=1):
        sample_id = sample_dir.name
        print(
            f"[mapping {index}/{len(sample_dirs)}] {sample_id}",
            flush=True,
        )
        sample, document_before, deleted_parts, repo_root, row = (
            collect_sample_inputs(sample_dir)
        )
        sample_run_dir = work_dir / sample_id
        sample_run_dir.mkdir(parents=True, exist_ok=True)
        if args.resume:
            reusable = load_reusable_bundle(
                sample_dir,
                sample_run_dir,
                sample,
                deleted_parts,
                model=model,
                work_dir=work_dir,
            )
            if reusable is not None:
                print(
                    f"[resume] reusing validated bundle for {sample_id}",
                    flush=True,
                )
                bundles[sample_id] = reusable
                reused_sample_ids.append(sample_id)
                continue
            print(
                f"[resume] no reusable bundle for {sample_id}; regenerating",
                flush=True,
            )
        write_json(sample_run_dir / "previous_inputs.json", sample)
        write_json(
            sample_run_dir / "previous_4_code_mapping.json",
            read_json_object(sample_dir / "4_code_mapping.json"),
        )
        snippets = extract_repository_snippets(repo_root)
        if not snippets:
            raise RuntimeError(
                f"{sample_id}: repository has no indexable text evidence"
            )
        bundles[sample_id] = generate_dual_mappings(
            row=row,
            document_before=document_before,
            deleted_parts=deleted_parts,
            repo_root=repo_root,
            snippets=snippets,
            run_dir=sample_run_dir,
            args=args,
            provider=provider,
            model=model,
            base_url=base_url,
            api_key=api_key,
        )
        write_json(
            sample_run_dir / "dual_mapping_bundle.json",
            bundles[sample_id],
        )
        write_json(
            sample_run_dir / "bundle_generation.json",
            {
                "schema_version": SCHEMA_VERSION,
                "generator_version": GENERATOR_VERSION,
                "provider": provider,
                "model": model,
                "completed_at": utc_now(),
            },
        )
    return bundles, reused_sample_ids


def merge_annotation_note(prior_notes: str, note: str) -> str:
    """Append a generated note once, even across repeated remap runs."""

    cleaned = prior_notes.strip()
    while note in cleaned:
        cleaned = cleaned.replace(note, " ")
    cleaned = " ".join(cleaned.split())
    return f"{cleaned} {note}".strip()


def update_staged_sample(
    sample_dir: Path,
    bundle: dict[str, Any],
    *,
    model: str,
) -> None:
    sample_id = sample_dir.name

    selection_path = sample_dir / "condition_selection.json"
    selection = read_json_object(selection_path)
    selection["schema_version"] = SCHEMA_VERSION
    write_json(selection_path, selection)

    write_json(
        sample_dir / "4_code_mapping.json",
        projection_payload(sample_id, "code_mappings", bundle),
    )


def stage_and_replace_collection(
    data_dir: Path,
    bundles: dict[str, dict[str, Any]],
    *,
    model: str,
) -> None:
    parent = data_dir.parent
    stage_parent = Path(
        tempfile.mkdtemp(prefix=f".{data_dir.name}.dual-stage-", dir=parent)
    )
    staged_collection = stage_parent / data_dir.name
    backup = parent / f".{data_dir.name}.pre-dual-backup"
    if backup.exists():
        shutil.rmtree(stage_parent)
        raise RuntimeError(f"refusing to overwrite existing backup: {backup}")

    try:
        shutil.copytree(data_dir, staged_collection)
        for sample_id, bundle in bundles.items():
            target = staged_collection / sample_id
            if not target.is_dir():
                raise RuntimeError(f"staged sample is missing: {sample_id}")
            update_staged_sample(target, bundle, model=model)
            errors = validate_sample(target)
            if errors:
                raise RuntimeError(
                    f"{sample_id}: staged validation failed:\n- "
                    + "\n- ".join(errors)
                )

        staged_manifest = read_json_object(
            staged_collection / "manifest.json"
        )
        records = {
            str(entry["sample_id"]): {
                **entry,
                "generator": (
                    f"remap_dual_mappings_v2_2.py/{GENERATOR_VERSION}"
                ),
                "model": model,
                "generated_at": utc_now(),
            }
            for entry in staged_manifest.get("samples", [])
            if isinstance(entry, dict) and entry.get("sample_id")
        }
        rebuild_collection_manifest(staged_collection, records)
        staged_samples = discover_samples(staged_collection)
        manifest_errors = validate_collection_manifest(
            staged_collection, staged_samples
        )
        if manifest_errors:
            raise RuntimeError(
                "staged manifest validation failed:\n- "
                + "\n- ".join(manifest_errors)
            )

        data_dir.replace(backup)
        try:
            staged_collection.replace(data_dir)
            final_errors: list[str] = []
            for sample_dir in discover_samples(data_dir):
                final_errors.extend(
                    f"{sample_dir.name}: {error}"
                    for error in validate_sample(sample_dir)
                )
            final_errors.extend(
                validate_collection_manifest(data_dir, discover_samples(data_dir))
            )
            if final_errors:
                raise RuntimeError(
                    "post-swap validation failed:\n- "
                    + "\n- ".join(final_errors)
                )
        except Exception:
            failed_new = stage_parent / f"{data_dir.name}.failed-new"
            if data_dir.exists():
                data_dir.replace(failed_new)
            backup.replace(data_dir)
            raise
        shutil.rmtree(backup)
    finally:
        if stage_parent.exists():
            shutil.rmtree(stage_parent)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.retries < 1:
        raise SystemExit("--retries must be positive")
    if args.holistic_shard_chars < 4_000:
        raise SystemExit("--holistic-shard-chars must be at least 4000")
    if args.max_holistic_aggregate_chars < 10_000:
        raise SystemExit(
            "--max-holistic-aggregate-chars must be at least 10000"
        )

    project_root = Path(__file__).resolve().parents[1]
    load_env_file(project_root / ".env")
    load_env_file(Path.cwd() / ".env")
    provider = args.provider
    model = get_model(provider, args.model)
    base_url = get_base_url(provider, args.base_url)
    api_key = get_api_key(provider, args.api_key)

    data_dir = args.data_dir
    if data_dir.is_symlink():
        raise SystemExit("--data-dir must not be a symbolic link")
    if paths_overlap(data_dir, args.work_dir):
        raise SystemExit(
            "--work-dir and --data-dir must be disjoint directories; "
            "neither may contain the other"
        )
    sample_dirs = discover_samples(data_dir)
    if not sample_dirs:
        print(f"[error] no samples found under {data_dir}", file=sys.stderr)
        return 2
    try:
        validate_inputs_before_ai(
            sample_dirs,
            reset_manual_comparison=args.reset_manual_comparison,
        )
        if args.resume:
            args.work_dir.mkdir(parents=True, exist_ok=True)
        else:
            archive_existing_work_dir(args.work_dir)
        bundles, reused_sample_ids = generate_all_bundles(
            sample_dirs,
            work_dir=args.work_dir,
            args=args,
            provider=provider,
            model=model,
            base_url=base_url,
            api_key=api_key,
        )
        stage_and_replace_collection(data_dir, bundles, model=model)
        write_json(
            args.work_dir / "summary.json",
            {
                "schema_version": SCHEMA_VERSION,
                "status": "complete",
                "sample_ids": [path.name for path in sample_dirs],
                "reused_sample_ids": reused_sample_ids,
                "model": model,
                "completed_at": utc_now(),
            },
        )
    except Exception as exc:
        write_json(
            args.work_dir / "summary.json",
            {
                "schema_version": SCHEMA_VERSION,
                "status": "failed",
                "sample_ids": [path.name for path in sample_dirs],
                "model": model,
                "error": str(exc),
                "failed_at": utc_now(),
            },
        )
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    print(
        f"[done] regenerated dual mappings for {len(sample_dirs)} sample(s)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
