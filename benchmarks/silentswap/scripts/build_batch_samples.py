#!/usr/bin/env python3
"""Build 100 formal SilentSwap samples from distinct DeNovoSWE instances."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import threading
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

import build_formal_samples as core


ROOT = Path(__file__).resolve().parent.parent
TARGET_COUNT = 100
DEFAULT_WORKERS = 2
DEFAULT_CANDIDATE_LIMIT = 1000
DEFAULT_DOCKER = "docker"
DEFAULT_DOCKER_PLATFORM = "linux/amd64"
MAX_INSTANCES_PER_REPOSITORY = 2

_write_lock = threading.Lock()


def append_progress(path: Path, event: dict[str, Any]) -> None:
    event = {"time": core.utc_now(), **event}
    with _write_lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        print(
            f"[{event['stage']}] {event.get('case_id', '')} {event.get('detail', '')}".rstrip(),
            flush=True,
        )


def record_projection(record: dict[str, Any]) -> dict[str, Any]:
    keys = {
        "instance_id", "github_url", "parent_commit", "document",
        "image_url", "workdir", "license_spdx_id",
    }
    return {key: record.get(key) for key in keys}


def eligible_record(record: dict[str, Any]) -> bool:
    return bool(
        record.get("instance_id")
        and record.get("github_url")
        and record.get("parent_commit")
        and record.get("document")
        and record.get("image_url")
        and record.get("workdir")
        and record.get("package_setup_files")
        and not record.get("submodule_uninitialized")
        and not record.get("error")
    )


def load_candidate_records(
    cache_path: Path,
    candidate_limit: int,
    target_count: int,
    resume: bool,
) -> list[dict[str, Any]]:
    if resume and cache_path.exists():
        cached = [json.loads(line) for line in cache_path.read_text(encoding="utf-8").splitlines()]
        if len(cached) >= candidate_limit:
            return cached[:candidate_limit]

    unique_repositories: list[dict[str, Any]] = []
    repeated_repositories: list[dict[str, Any]] = []
    seen_instances: set[str] = set()
    seen_repositories: set[str] = set()
    fs = core.HfFileSystem()
    with fs.open(core.DATASET_PATH, "rb") as handle:
        for line in handle:
            record = json.loads(line)
            if not eligible_record(record) or record["instance_id"] in seen_instances:
                continue
            seen_instances.add(record["instance_id"])
            projected = record_projection(record)
            if record["github_url"] in seen_repositories:
                repeated_repositories.append(projected)
            else:
                seen_repositories.add(record["github_url"])
                unique_repositories.append(projected)
            if len(unique_repositories) >= candidate_limit:
                break

    selected = select_diverse_records(
        unique_repositories, repeated_repositories, candidate_limit, target_count,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as handle:
        for record in selected:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return selected


def select_diverse_records(
    unique_repositories: list[dict[str, Any]],
    repeated_repositories: list[dict[str, Any]],
    candidate_limit: int,
    target_count: int,
) -> list[dict[str, Any]]:
    selected = unique_repositories[:candidate_limit]
    if len(unique_repositories) >= target_count:
        return selected
    repository_counts = Counter(record["github_url"] for record in selected)
    for record in repeated_repositories:
        if len(selected) >= candidate_limit:
            break
        repository = record["github_url"]
        if repository_counts[repository] >= MAX_INSTANCES_PER_REPOSITORY:
            continue
        selected.append(record)
        repository_counts[repository] += 1
    return selected


def validate_source_diversity(records: list[dict[str, Any]], target_count: int) -> None:
    instance_ids = [record["instance_id"] for record in records]
    if len(instance_ids) != len(set(instance_ids)):
        raise core.PipelineError("candidate records contain duplicate instance_id values")
    if len(instance_ids) < target_count:
        raise core.PipelineError("not enough distinct DeNovoSWE instances")
    repository_counts = Counter(record["github_url"] for record in records)
    if len(repository_counts) >= target_count and any(count > 1 for count in repository_counts.values()):
        raise core.PipelineError("repository repetition used despite sufficient unique repositories")


def repository_index(repo: Path) -> str:
    result = core.run(["git", "ls-files"], repo)
    if result["exit_code"] != 0:
        raise core.PipelineError(result["stderr"])
    entries = []
    for relative in result["stdout"].splitlines():
        file_path = repo / relative
        if file_path.is_file():
            entries.append(f"{relative}\t{file_path.stat().st_size} bytes")
    return "\n".join(entries)


def selector_prompt(record: dict[str, Any], index: str) -> str:
    contract = {
        "selection": {
            "original_document_text": "one exact unique substring copied from the complete document",
            "original_semantics": "concise externally observable documented behavior",
            "target_files": ["exact production-code paths from the repository index"],
            "related_test_files": ["exact existing test paths from the repository index"],
            "swap_direction": "one meaningful runnable semantic substitution",
            "selection_reason": "why the behavior matters and may evade existing tests",
        }
    }
    return (
        "Select exactly one candidate constraint for a formal SilentSwap sample. Read the complete document and "
        "complete repository file index, then independently choose an externally observable requirement and its "
        "production and test files. Prefer a focused boundary, default, error, validation, matching, ordering, state, "
        "or transformation behavior. Do not select style, logging-only, packaging, dependency, performance-only, "
        "documentation-only, or test changes. Return exactly the OUTPUT object and no Markdown.\n\n"
        f"OUTPUT:\n{json.dumps(contract, ensure_ascii=False, indent=2)}\n\n"
        f"INSTANCE_ID: {record['instance_id']}\n"
        f"COMPLETE DOCUMENT:\n{record['document']}\n\n"
        f"COMPLETE REPOSITORY FILE INDEX:\n{index}"
    )


def validate_selection(selection: dict[str, Any], record: dict[str, Any], repo: Path) -> None:
    required = {
        "original_document_text", "original_semantics", "target_files",
        "related_test_files", "swap_direction", "selection_reason",
    }
    if set(selection) != required:
        raise core.PipelineError("unexpected DeepSeek Selector fields")
    if record["document"].count(selection["original_document_text"]) != 1:
        raise core.PipelineError("selector document text must occur exactly once")
    if not selection["target_files"] or not selection["related_test_files"]:
        raise core.PipelineError("selector must identify production and test files")
    for relative in selection["target_files"] + selection["related_test_files"]:
        if not (repo / relative).is_file():
            raise core.PipelineError(f"selector file does not exist: {relative}")
    for relative in selection["target_files"]:
        parts = Path(relative).parts
        if "tests" in parts or Path(relative).name.startswith("test"):
            raise core.PipelineError("selector target must be production code")
    for relative in selection["related_test_files"]:
        parts = Path(relative).parts
        if not ({"test", "tests"} & set(parts)) and not Path(relative).name.startswith("test"):
            raise core.PipelineError("selector test file is not under a test path")


def builder_prompt(
    record: dict[str, Any],
    selection: dict[str, Any],
    context: str,
) -> str:
    contract = {
        "title": "short case title",
        "original_document_text": selection["original_document_text"],
        "replacement_document_text": "minimal text honestly describing the changed behavior",
        "target_file": "one selected production-code path",
        "original_code_text": "one exact unique production-code substring",
        "replacement_code_text": "minimal replacement implementing one semantic change",
        "semantic_test": "standalone Python: pass on original, fail after replacement, print observations",
        "gold": {
            "swap_type": f"one of: {', '.join(core.SWAP_TYPES)}",
            "original_semantics": "documented behavior",
            "swapped_semantics": "replacement behavior",
            "why_different": "meaningful consequence",
        },
    }
    return (
        "Construct one formal SilentSwap from the independent Selector decision. Use exactly the selected document "
        "substring and change exactly one selected production file. The smallest runnable substitution must affect a "
        "realistic class of inputs while the supplied repository tests remain unchanged and pass. The semantic test "
        "must observe the documented behavior, pass on the parent commit, fail after replacement, and print the "
        "difference. Do not modify tests, signatures, packaging, or dependencies. Return exactly the OUTPUT object.\n\n"
        f"OUTPUT:\n{json.dumps(contract, ensure_ascii=False, indent=2)}\n\n"
        f"SELECTOR DECISION:\n{json.dumps(selection, ensure_ascii=False, indent=2)}\n\n"
        f"COMPLETE DOCUMENT:\n{record['document']}\n\n"
        f"SELECTED COMPLETE FILES:\n{context}"
    )


def quality_prompt(
    selection: dict[str, Any],
    builder: dict[str, Any],
    patch: str,
    evidence: dict[str, Any],
) -> str:
    contract = {
        "verdict": "accept or reject",
        "quality": {
            "document_alignment": "assessment",
            "single_semantic_change": "assessment",
            "test_validity": "assessment",
            "generalization": "realistic affected input class",
        },
        "gold": {
            "swap_type": f"one of: {', '.join(core.SWAP_TYPES)}",
            "original_semantics": "original documented behavior",
            "swapped_semantics": "behavior after replacement",
            "why_different": "meaningful consequence",
            "evidence": ["specific document, patch, and execution references"],
        },
    }
    return (
        "Independently judge this formal-data candidate. Reject it if the patch is not aligned with the selected "
        "document constraint, changes multiple semantics, is trivial or test-tailored, affects only one hard-coded "
        "literal, or if the semantic test observes an incidental detail. Repository tests passing is insufficient. "
        "Return exactly the OUTPUT object and no Markdown.\n\n"
        f"OUTPUT:\n{json.dumps(contract, ensure_ascii=False, indent=2)}\n\n"
        f"SELECTOR:\n{json.dumps(selection, ensure_ascii=False, indent=2)}\n\n"
        f"BUILDER:\n{json.dumps(builder, ensure_ascii=False, indent=2)}\n\n"
        f"PATCH:\n{patch}\n\n"
        f"EVIDENCE:\n{json.dumps(evidence, ensure_ascii=False, indent=2)}"
    )


def call_model(
    api_key: str,
    prompt: str,
    timeout: int,
    progress: Path,
    case_id: str,
    role: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    append_progress(progress, {
        "stage": "api:start", "case_id": case_id, "role": role,
        "model": core.MODEL, "thinking": "enabled",
    })
    result, usage = core.call_deepseek(
        api_key, prompt, timeout, max_tokens=24000, reasoning_effort="high",
    )
    append_progress(progress, {
        "stage": "api:done", "case_id": case_id, "role": role,
        "model": core.MODEL, "thinking": "enabled", "usage": usage,
        "detail": f"tokens={usage.get('total_tokens', 'unknown')}",
    })
    return result, usage


def prepare_repository(record: dict[str, Any], repo: Path) -> Path:
    repo.parent.mkdir(parents=True, exist_ok=True)
    if not (repo / ".git").is_dir():
        cloned = core.run(["git", "clone", "--quiet", record["github_url"], str(repo)], repo.parent)
        if cloned["exit_code"] != 0:
            raise core.PipelineError(cloned["stderr"])
        checkout = core.run(["git", "checkout", "--quiet", record["parent_commit"]], repo)
        if checkout["exit_code"] != 0:
            raise core.PipelineError(checkout["stderr"])
    head = core.run(["git", "rev-parse", "HEAD"], repo)["stdout"].strip()
    if head != record["parent_commit"]:
        raise core.PipelineError("repository is not at parent_commit")
    return repo


def docker_command(
    runtime: dict[str, Any],
    repo: Path,
    command: list[str],
    mounts: list[tuple[Path, str, str]] | None = None,
) -> list[str]:
    result = [
        runtime["docker"], "run", "--rm",
        "--platform", runtime["platform"],
        "--volume", f"{repo.resolve()}:{runtime['workdir']}:rw",
        "--workdir", runtime["workdir"],
        "--env", f"PYTHONPATH={runtime['workdir']}",
    ]
    for source, target, mode in mounts or []:
        result.extend(["--volume", f"{source.resolve()}:{target}:{mode}"])
    result.extend(["--entrypoint", command[0], runtime["image"], *command[1:]])
    return result


def run_in_docker(
    runtime: dict[str, Any],
    repo: Path,
    command: list[str],
    *,
    mounts: list[tuple[Path, str, str]] | None = None,
    timeout: int = 1200,
) -> dict[str, Any]:
    return core.run(
        docker_command(runtime, repo, command, mounts),
        repo.parent,
        timeout=timeout,
    )


def prepare_runtime(
    record: dict[str, Any],
    repo: Path,
    docker: str,
    platform: str,
) -> dict[str, Any]:
    runtime = {
        "docker": docker,
        "platform": platform,
        "image": record["image_url"],
        "workdir": record["workdir"],
        "test_command": ["python", "-m", "pytest", "-q"],
        "image_removal_attempted": False,
    }
    pulled = core.run(
        [docker, "pull", "--platform", platform, record["image_url"]],
        repo.parent,
        timeout=3600,
    )
    if pulled["exit_code"] != 0:
        raise core.PipelineError(pulled["stderr"])
    runtime["image_pull"] = pulled
    baseline = run_in_docker(runtime, repo, runtime["test_command"])
    python_version = run_in_docker(runtime, repo, ["python", "--version"])
    runtime.update({
        "original": repo,
        "baseline": baseline,
        "python_version": (python_version["stdout"] + python_version["stderr"]).strip(),
    })
    return runtime


def remove_docker_image(runtime: dict[str, Any], repo: Path) -> dict[str, Any]:
    runtime["image_removal_attempted"] = True
    removal = core.run(
        [runtime["docker"], "image", "rm", runtime["image"]],
        repo.parent,
        timeout=300,
    )
    runtime["image_removal"] = removal
    return removal


def duplicate_patch(output_root: Path, patch: str) -> str | None:
    for existing in output_root.glob("*/swap.patch"):
        if existing.read_text(encoding="utf-8") == patch:
            return existing.parent.name
    return None


def is_docker_validated(case: dict[str, Any]) -> bool:
    environment = case.get("test_environment", {})
    return bool(
        case.get("status") == "validated"
        and environment.get("docker_platform")
        and environment.get("container_workdir")
    )


def build_sample(
    record: dict[str, Any],
    runtime: dict[str, Any],
    api_key: str,
    output_root: Path,
    work_root: Path,
    progress: Path,
    timeout: int,
    resume: bool,
) -> str:
    case_id = record["instance_id"]
    existing = core.find_sample_directory(output_root, case_id)
    if resume and existing is not None:
        case = json.loads((existing / "case.json").read_text(encoding="utf-8"))
        if is_docker_validated(case):
            return case_id

    state = work_root / "cases" / case_id
    state.mkdir(parents=True, exist_ok=True)
    original_repo = runtime["original"]
    candidate_repo = state / "candidate_repo"
    if not (candidate_repo / ".git").is_dir():
        cloned = core.run(
            ["git", "clone", "--quiet", str(original_repo), str(candidate_repo)], state,
        )
        if cloned["exit_code"] != 0:
            raise core.PipelineError(cloned["stderr"])

    patch_path = state / "swap.patch"
    if patch_path.exists():
        restored = core.run(["git", "apply", "--reverse", str(patch_path)], candidate_repo)
        if restored["exit_code"] != 0:
            raise core.PipelineError("cannot restore interrupted candidate patch")

    selector_path = state / "selector.json"
    selector_usage_path = state / "selector_usage.json"
    if resume and selector_path.exists():
        selector_response = json.loads(selector_path.read_text(encoding="utf-8"))
        selector_usage = json.loads(selector_usage_path.read_text(encoding="utf-8"))
    else:
        selector_response, selector_usage = call_model(
            api_key,
            selector_prompt(record, repository_index(original_repo)),
            timeout,
            progress,
            case_id,
            "selector",
        )
        core.write_json(selector_path, selector_response)
        core.write_json(selector_usage_path, selector_usage)
    if set(selector_response) != {"selection"}:
        raise core.PipelineError("unexpected Selector response")
    selection = selector_response["selection"]
    validate_selection(selection, record, original_repo)

    selected_files = list(dict.fromkeys(
        selection["target_files"] + selection["related_test_files"],
    ))
    context = core.repository_context(original_repo, selected_files)
    builder_path = state / "builder.json"
    builder_usage_path = state / "builder_usage.json"
    if resume and builder_path.exists():
        builder = json.loads(builder_path.read_text(encoding="utf-8"))
        builder_usage = json.loads(builder_usage_path.read_text(encoding="utf-8"))
    else:
        builder, builder_usage = call_model(
            api_key,
            builder_prompt(record, selection, context),
            timeout,
            progress,
            case_id,
            "builder",
        )
        core.validate_builder_output(builder)
        core.write_json(builder_path, builder)
        core.write_json(builder_usage_path, builder_usage)
    core.validate_builder_output(builder)
    if builder["original_document_text"] != selection["original_document_text"]:
        raise core.PipelineError("Builder changed the selected document constraint")
    if builder["target_file"] not in selection["target_files"]:
        raise core.PipelineError("Builder changed a file outside the Selector decision")

    core.replace_once(
        record["document"],
        builder["original_document_text"],
        builder["replacement_document_text"],
        "original_document_text",
    )
    semantic_test_path = state / "semantic_swap_verification_test.py"
    semantic_test_path.write_text(builder["semantic_test"], encoding="utf-8")
    container_semantic_test = "/tmp/silentswap_semantic_test.py"
    semantic_mount = [(semantic_test_path, container_semantic_test, "ro")]
    semantic_original = run_in_docker(
        runtime,
        original_repo,
        ["python", container_semantic_test],
        mounts=semantic_mount,
    )
    if semantic_original["exit_code"] != 0:
        raise core.PipelineError("semantic test failed on original code")

    target_path = candidate_repo / builder["target_file"]
    source = target_path.read_text(encoding="utf-8")
    target_path.write_text(
        core.replace_once(
            source,
            builder["original_code_text"],
            builder["replacement_code_text"],
            "original_code_text",
        ),
        encoding="utf-8",
    )
    patch = core.run(["git", "diff", "--", builder["target_file"]], candidate_repo)["stdout"]
    if not patch:
        raise core.PipelineError("code replacement produced no diff")
    patch_path.write_text(patch, encoding="utf-8")
    replay = core.run(["git", "apply", "--check", str(patch_path)], original_repo)
    if replay["exit_code"] != 0:
        raise core.PipelineError("generated patch is not replayable")
    changed_files = core.run(["git", "diff", "--name-only"], candidate_repo)["stdout"].splitlines()
    if changed_files != [builder["target_file"]]:
        raise core.PipelineError("candidate changed an unexpected file")

    after_swap = run_in_docker(
        runtime, candidate_repo, runtime["test_command"],
    )
    if after_swap["exit_code"] != 0:
        raise core.PipelineError("repository tests failed after swap")
    semantic_after = run_in_docker(
        runtime,
        candidate_repo,
        ["python", container_semantic_test],
        mounts=semantic_mount,
    )
    if semantic_after["exit_code"] == 0:
        raise core.PipelineError("semantic test did not detect the swap")

    evidence = {
        "original_suite_on_original": runtime["baseline"],
        "semantic_test_source": builder["semantic_test"],
        "semantic_test_on_original": semantic_original,
        "original_suite_after_swap": after_swap,
        "semantic_test_after_swap": semantic_after,
        "changed_files": changed_files,
        "behavior_difference": {
            "original": semantic_original["stdout"] + semantic_original["stderr"],
            "swapped": semantic_after["stdout"] + semantic_after["stderr"],
        },
    }
    quality, checker_usage = call_model(
        api_key,
        quality_prompt(selection, builder, patch, evidence),
        timeout,
        progress,
        case_id,
        "quality_checker",
    )
    if set(quality) != {"verdict", "quality", "gold"} or quality["verdict"] != "accept":
        raise core.PipelineError("quality checker rejected the sample")
    gold = core.prepare_generated_gold(quality["gold"])

    image_removal = remove_docker_image(runtime, original_repo)
    append_progress(progress, {
        "stage": "docker:image_removed",
        "case_id": case_id,
        "image": runtime["image"],
        "detail": f"exit_code={image_removal['exit_code']}",
    })
    if image_removal["exit_code"] != 0:
        raise core.PipelineError("Docker image removal failed")
    evidence["docker_image_lifecycle"] = {
        "image": runtime["image"],
        "platform": runtime["platform"],
        "pull": runtime["image_pull"],
        "remove_after_validation": image_removal,
    }
    core.sanitize_verification_commands(evidence)

    trajectory = [
        {"time": core.utc_now(), "actor": "pipeline", "action": "pull_docker_image", "result": runtime["image_pull"]},
        {"time": core.utc_now(), "actor": "deepseek_selector", "action": "select_constraint", "output": selection},
        {"time": core.utc_now(), "actor": "deepseek_builder", "action": "construct_swap", "output": builder},
        {"time": core.utc_now(), "actor": "pipeline", "action": "verify_swap", "result": evidence},
        {"time": core.utc_now(), "actor": "deepseek_quality_checker", "action": "judge_sample", "output": quality},
        {"time": core.utc_now(), "actor": "pipeline", "action": "remove_docker_image", "result": image_removal},
    ]
    with _write_lock:
        duplicate = duplicate_patch(output_root, patch)
        if duplicate:
            raise core.PipelineError(f"duplicate swap patch already used by {duplicate}")
        sample_number = core.next_sample_number(output_root)
        output_dir = output_root / str(sample_number)
        package_dir = state / "package"
        if package_dir.exists():
            shutil.rmtree(package_dir)
        package_dir.mkdir()
        case = {
            "sample_number": sample_number,
            "case_id": case_id,
            "status": "validated",
            "source": {
                "dataset": "AweAI-Team/DeNovoSWE",
                "instance_id": record["instance_id"],
                "repository_url": record["github_url"],
                "repository_path": f"data/{sample_number}/repository",
                "parent_commit": record["parent_commit"],
                "image_url": record["image_url"],
                "license_spdx_id": record["license_spdx_id"],
            },
            "test_environment": {
                "source_image": record["image_url"],
                "docker_platform": runtime["platform"],
                "container_workdir": runtime["workdir"],
                "container_python": runtime["python_version"],
                "test_command": shlex.join(runtime["test_command"]),
                "image_removed_after_validation": True,
            },
            "mapping": {
                "document_text": builder["original_document_text"],
                "target_file": builder["target_file"],
                "semantic_test": "verification_evidence.json#semantic_test_source",
            },
            "provenance": {
                "construction": "controlled_injection",
                "model": core.MODEL,
                "thinking_enabled": True,
                "selector_mode": "complete_document_and_repository_index",
                "selector_usage": selector_usage,
                "builder_usage": builder_usage,
                "quality_checker_usage": checker_usage,
            },
        }
        shutil.move(str(original_repo), str(package_dir / "repository"))
        (package_dir / "original_document.md").write_text(record["document"], encoding="utf-8")
        (package_dir / "swap.patch").write_text(patch, encoding="utf-8")
        with (package_dir / "construction_trajectory.jsonl").open("w", encoding="utf-8") as handle:
            for event in trajectory:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        core.write_json(package_dir / "verification_evidence.json", evidence)
        core.write_json(package_dir / "gold.json", gold)
        core.write_json(package_dir / "case.json", case)
        package_dir.replace(output_dir)
    append_progress(progress, {
        "stage": "case:validated", "case_id": case_id, "sample_number": sample_number,
    })
    return case_id


def validated_cases(output_root: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for sample_dir in core.sample_directories(output_root):
        case_path = sample_dir / "case.json"
        if not case_path.exists():
            continue
        case = json.loads(case_path.read_text(encoding="utf-8"))
        if is_docker_validated(case):
            result[case["case_id"]] = case
    return result


def run_record(
    record: dict[str, Any],
    args: argparse.Namespace,
    api_key: str,
    progress: Path,
) -> tuple[bool, str]:
    case_id = record["instance_id"]
    state = args.work / "cases" / case_id
    state.mkdir(parents=True, exist_ok=True)
    runtime: dict[str, Any] | None = None
    try:
        repo = prepare_repository(record, state / "repository")
        runtime = {
            "docker": args.docker,
            "image": record["image_url"],
            "image_removal_attempted": False,
        }
        runtime = prepare_runtime(record, repo, args.docker, args.docker_platform)
        if runtime["baseline"]["exit_code"] != 0:
            raise core.PipelineError("baseline repository tests failed")
        build_sample(
            record, runtime, api_key, args.output, args.work,
            progress, args.timeout, args.resume,
        )
        shutil.rmtree(state)
        return True, case_id
    except Exception as error:
        cleanup = None
        if runtime and not runtime["image_removal_attempted"]:
            cleanup = remove_docker_image(runtime, repo)
            append_progress(progress, {
                "stage": "docker:image_removed_after_failure",
                "case_id": case_id,
                "image": runtime["image"],
                "detail": f"exit_code={cleanup['exit_code']}",
            })
        failure = {
            "time": core.utc_now(),
            "case_id": case_id,
            "error": str(error),
        }
        if cleanup:
            failure["docker_image_cleanup"] = cleanup
        core.write_json(state / "failure.json", failure)
        append_progress(progress, {
            "stage": "case:failed",
            "case_id": case_id,
            "detail": f"{type(error).__name__}: {error}",
        })
        return False, case_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=TARGET_COUNT)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--candidate-limit", type=int, default=DEFAULT_CANDIDATE_LIMIT)
    parser.add_argument("--docker", default=DEFAULT_DOCKER)
    parser.add_argument("--docker-platform", default=DEFAULT_DOCKER_PLATFORM)
    parser.add_argument("--output", type=Path, default=ROOT / "data")
    parser.add_argument("--work", type=Path, default=ROOT / ".work" / "formal_batch")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise core.PipelineError("DEEPSEEK_API_KEY is required")
    if shutil.which(args.docker) is None:
        raise core.PipelineError("Docker CLI is unavailable")
    docker_version = core.run(
        [args.docker, "version", "--format", "{{.Server.Version}}"],
        ROOT,
        timeout=60,
    )
    if docker_version["exit_code"] != 0:
        raise core.PipelineError("Docker daemon is unavailable")
    args.output.mkdir(parents=True, exist_ok=True)
    args.work.mkdir(parents=True, exist_ok=True)
    progress = args.work / "progress.jsonl"
    records = load_candidate_records(
        args.work / "source_records.jsonl",
        args.candidate_limit,
        args.count,
        args.resume,
    )
    validate_source_diversity(records, args.count)

    completed = validated_cases(args.output)
    if len(completed) >= args.count:
        print(f"[batch:done] validated={len(completed)}", flush=True)
        return 0
    completed_instances = set(completed)
    completed_repositories = {
        case["source"]["repository_url"] for case in completed.values()
    }
    candidates = iter(
        record for record in records
        if record["instance_id"] not in completed_instances
        and record["github_url"] not in completed_repositories
        and not (
            args.resume
            and (args.work / "cases" / record["instance_id"] / "failure.json").exists()
        )
    )

    running: dict[Future[tuple[bool, str]], dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        while len(completed) < args.count:
            needed = args.count - len(completed)
            while len(running) < min(args.workers, needed):
                try:
                    record = next(candidates)
                except StopIteration as error:
                    raise core.PipelineError(
                        f"distinct-repository candidates exhausted at {len(completed)}/{args.count}",
                    ) from error
                future = pool.submit(run_record, record, args, api_key, progress)
                running[future] = record
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in done:
                success, case_id = future.result()
                record = running.pop(future)
                if success:
                    sample_dir = core.find_sample_directory(args.output, case_id)
                    completed[case_id] = json.loads(
                        (sample_dir / "case.json").read_text(encoding="utf-8"),
                    )
                    completed_repositories.add(record["github_url"])
                print(
                    f"[batch:progress] validated={len(completed)}/{args.count} "
                    f"running={len(running)}",
                    flush=True,
                )
    print(f"[batch:done] validated={len(completed)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
