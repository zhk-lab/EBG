#!/usr/bin/env python3
"""Expand the 100 formal samples to five independently validated SilentSwaps each."""

from __future__ import annotations

import argparse
import ast
import difflib
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import threading
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import build_batch_samples as batch
import build_formal_samples as core
import docker_trim
import code_locations as evaluation


ROOT = Path(
    os.environ.get(
        "SILENTSWAP_PROJECT_ROOT", str(Path(__file__).resolve().parent.parent)
    )
).resolve()
DATA_ROOT = ROOT / "data"
WORK_ROOT = Path(
    os.environ.get(
        "SILENTSWAP_WORK_ROOT", str(ROOT / ".work" / "multi_swap")
    )
).resolve()
MODEL = core.MODEL
TARGET_SWAP_COUNT = 5
NEW_SWAP_COUNT = TARGET_SWAP_COUNT - 1
DEFAULT_WORKERS = 2
DEFAULT_MAX_ATTEMPTS = 8
PORTABLE_ROOT = "${SILENTSWAP_ROOT}"
RUNTIME_ROOT = Path.home() / ".cache" / "silentswap_multi_swap"
PORTABLE_RUNTIME_ROOT = f"{PORTABLE_ROOT}/.work/multi_swap/runtime"
_print_lock = threading.Lock()


class ExpansionError(RuntimeError):
    pass


class CandidateRejected(ExpansionError):
    pass


class CombinationRejected(ExpansionError):
    def __init__(self, message: str, regenerate_from_slot: int) -> None:
        super().__init__(message)
        self.regenerate_from_slot = regenerate_from_slot


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    with _print_lock:
        print(message, flush=True)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def hydrate_file(path: Path) -> None:
    with path.open("rb") as handle:
        while handle.read(1024 * 1024):
            pass


def hydrate_sample(sample: Path) -> None:
    for path in sample.iterdir():
        if path.is_file():
            hydrate_file(path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def run(command: list[str], cwd: Path, timeout: int = 1200) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return {
        "command": shlex.join(command),
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "counts": core.parse_test_counts(completed.stdout + completed.stderr),
    }


def case_record(sample: Path) -> dict[str, Any]:
    case = read_json(sample / "case.json")
    source = case["source"]
    environment = case["test_environment"]
    return {
        "instance_id": case["case_id"],
        "github_url": source["repository_url"],
        "parent_commit": source["parent_commit"],
        "document": (sample / "original_document.md").read_text(encoding="utf-8"),
        "image_url": environment["source_image"],
        "workdir": environment["container_workdir"],
        "license_spdx_id": source.get("license_spdx_id"),
    }


def load_original_builder(sample: Path) -> dict[str, Any]:
    for line in (sample / "construction_trajectory.jsonl").read_text(
        encoding="utf-8"
    ).splitlines():
        event = json.loads(line)
        if event.get("actor") == "deepseek_builder" and event.get("action") == "construct_swap":
            return dict(event["output"])
    raise ExpansionError(f"sample {sample.name}: original Builder output is missing")


def load_original_quality(sample: Path) -> dict[str, Any]:
    for line in (sample / "construction_trajectory.jsonl").read_text(
        encoding="utf-8"
    ).splitlines():
        event = json.loads(line)
        if event.get("actor") == "deepseek_quality_checker" and event.get("action") == "judge_sample":
            return event["output"]
    raise ExpansionError(f"sample {sample.name}: original Quality Checker output is missing")


def runtime_from_case(sample: Path, docker: str) -> dict[str, Any]:
    case = read_json(sample / "case.json")
    environment = case["test_environment"]
    image = environment["source_image"]
    mirror = os.environ.get("SILENTSWAP_DOCKER_MIRROR", "").strip().rstrip("/")
    return {
        "docker": docker,
        "crane": os.environ.get("SILENTSWAP_CRANE", "crane"),
        "platform": environment["docker_platform"],
        "image": image,
        "pull_image": f"{mirror}/{image}" if mirror else image,
        "preloaded_image": os.environ.get("SILENTSWAP_DOCKER_PRELOADED") == "1",
        "crane_transport": os.environ.get("SILENTSWAP_DOCKER_CRANE") == "1",
        "workdir": environment["container_workdir"],
        "test_command": shlex.split(environment["test_command"]),
        "image_removal_attempted": False,
    }


def docker_command(
    runtime: dict[str, Any],
    repository: Path,
    command: list[str],
    mounts: list[tuple[Path, str, str]] | None = None,
) -> list[str]:
    return batch.docker_command(runtime, repository, command, mounts)


def run_in_docker(
    runtime: dict[str, Any],
    repository: Path,
    command: list[str],
    *,
    mounts: list[tuple[Path, str, str]] | None = None,
    timeout: int = 1200,
) -> dict[str, Any]:
    return run(docker_command(runtime, repository, command, mounts), repository.parent, timeout)


def pull_and_baseline(runtime: dict[str, Any], repository: Path) -> dict[str, Any]:
    if runtime["preloaded_image"]:
        pull = run(
            [
                runtime["docker"],
                "image",
                "inspect",
                "--format",
                "{{.Os}}/{{.Architecture}}",
                runtime["image"],
            ],
            repository.parent,
            timeout=300,
        )
        if pull["exit_code"] == 0 and pull["stdout"].strip() != runtime["platform"]:
            raise ExpansionError(
                f"Preloaded Docker image platform mismatch: {pull['stdout'].strip()}"
            )
    elif runtime["crane_transport"]:
        archive = repository.parent / "source-image.tar"
        try:
            pull = run(
                [
                    runtime["crane"],
                    "pull",
                    "--platform",
                    runtime["platform"],
                    runtime["image"],
                    str(archive),
                ],
                repository.parent,
                timeout=3600,
            )
            runtime["image_pull"] = pull
            if pull["exit_code"] != 0:
                raise ExpansionError(
                    f"Docker image pull failed: {pull['stderr'].strip()}"
                )
            load = run(
                [runtime["docker"], "load", "--input", str(archive)],
                repository.parent,
                timeout=1800,
            )
            runtime["image_load"] = load
            if load["exit_code"] != 0:
                raise ExpansionError(
                    f"Docker image load failed: {load['stderr'].strip()}"
                )
        finally:
            archive.unlink(missing_ok=True)
    else:
        pull = run(
            [
                runtime["docker"],
                "pull",
                "--platform",
                runtime["platform"],
                runtime["pull_image"],
            ],
            repository.parent,
            timeout=3600,
        )
    if pull["exit_code"] != 0:
        raise ExpansionError(f"Docker image pull failed: {pull['stderr'].strip()}")
    runtime["image_pull"] = pull
    if runtime["pull_image"] != runtime["image"]:
        tag = run(
            [
                runtime["docker"],
                "image",
                "tag",
                runtime["pull_image"],
                runtime["image"],
            ],
            repository.parent,
            timeout=300,
        )
        runtime["image_tag"] = tag
        if tag["exit_code"] != 0:
            raise ExpansionError(f"Docker mirror image tag failed: {tag['stderr'].strip()}")
    baseline = run_in_docker(runtime, repository, runtime["test_command"])
    runtime["baseline"] = baseline
    if baseline["exit_code"] != 0:
        raise ExpansionError("original repository test suite failed")
    python_version = run_in_docker(runtime, repository, ["python", "--version"])
    runtime["python_version"] = (python_version["stdout"] + python_version["stderr"]).strip()
    return baseline


def remove_image(runtime: dict[str, Any], repository: Path) -> dict[str, Any]:
    runtime["image_removal_attempted"] = True
    targets = list(dict.fromkeys([runtime["image"], runtime["pull_image"]]))
    removal = run(
        [runtime["docker"], "image", "rm", *targets],
        repository.parent,
        timeout=300,
    )
    runtime["image_removal"] = removal
    return removal


def trim_docker_storage(runtime: dict[str, Any]) -> dict[str, Any]:
    result = docker_trim.trim()
    runtime["disk_trim"] = result
    state = "skipped" if result["skipped"] else "done"
    if result["exit_code"] == 0:
        log(f"[docker:trim:{state}] image={runtime['image']}")
    else:
        log(
            f"[docker:trim:failed] image={runtime['image']} "
            f"error={result['stderr'].strip()}"
        )
    return result


def call_model(
    api_key: str,
    prompt: str,
    timeout: int,
    state: Path,
    role: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    log(f"[api:start] sample={state.parts[-3]} slot={state.parts[-2]} role={role}")
    response, usage = core.call_deepseek(
        api_key,
        prompt,
        timeout,
        max_tokens=32_768,
        reasoning_effort="high",
    )
    write_json(state / f"{role}.json", response)
    write_json(state / f"{role}_usage.json", usage)
    log(f"[api:done] sample={state.parts[-3]} slot={state.parts[-2]} role={role}")
    return response, usage


def call_or_load_model(
    api_key: str,
    prompt: str,
    timeout: int,
    state: Path,
    role: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    response_path = state / f"{role}.json"
    usage_path = state / f"{role}_usage.json"
    if response_path.exists() and usage_path.exists():
        return read_json(response_path), read_json(usage_path)
    return call_model(api_key, prompt, timeout, state, role)


def exclusion_text(builders: list[dict[str, Any]], rejected: list[dict[str, Any]]) -> str:
    exclusions = [
        {
            "original_document_text": builder["original_document_text"],
            "target_file": builder["target_file"],
            "original_code_text": builder["original_code_text"],
        }
        for builder in builders
    ]
    exclusions.extend(
        {
            "original_document_text": item.get("original_document_text"),
            "target_files": item.get("target_files"),
            "reason": item.get("reason"),
        }
        for item in rejected
    )
    return json.dumps(exclusions, ensure_ascii=False, indent=2)


def additional_selector_prompt(
    record: dict[str, Any],
    repository: Path,
    builders: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
) -> str:
    prompt = batch.selector_prompt(record, batch.repository_index(repository))
    return (
        prompt
        + "\n\nEXCLUDED PRIOR OR REJECTED LOCATIONS:\n"
        + exclusion_text(builders, rejected)
        + "\n\nSelect one new constraint that is semantically independent of every exclusion. "
        "Do not reuse an excluded document substring or code decision point. Prefer a production file not yet used."
    )


def validate_additional_selection(
    response: dict[str, Any],
    record: dict[str, Any],
    repository: Path,
    builders: list[dict[str, Any]],
) -> dict[str, Any]:
    if set(response) != {"selection"}:
        raise CandidateRejected("Selector response must contain only selection")
    selection = response["selection"]
    try:
        batch.validate_selection(selection, record, repository)
    except core.PipelineError as error:
        raise CandidateRejected(str(error)) from error
    used_document = {builder["original_document_text"] for builder in builders}
    if selection["original_document_text"] in used_document:
        raise CandidateRejected("Selector reused a document constraint")
    return selection


def validate_additional_builder(
    builder: dict[str, Any],
    selection: dict[str, Any],
    builders: list[dict[str, Any]],
) -> None:
    try:
        core.validate_builder_output(builder)
    except core.PipelineError as error:
        raise CandidateRejected(str(error)) from error
    if builder["original_document_text"] != selection["original_document_text"]:
        raise CandidateRejected("Builder changed the selected document constraint")
    if builder["target_file"] not in selection["target_files"]:
        raise CandidateRejected("Builder changed a file outside the Selector decision")
    locations = {
        (existing["target_file"], existing["original_code_text"])
        for existing in builders
    }
    if (builder["target_file"], builder["original_code_text"]) in locations:
        raise CandidateRejected("Builder reused an existing code location")


def clone_repository(source: Path, destination: Path) -> Path:
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cloned = run(
        ["git", "clone", "--quiet", "--no-hardlinks", str(source), str(destination)],
        destination.parent,
    )
    if cloned["exit_code"] != 0:
        raise ExpansionError(cloned["stderr"].strip())
    return destination


def clone_source_repository(case: dict[str, Any], destination: Path) -> Path:
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    repository_url = case["source"]["repository_url"]
    parsed = urlparse(repository_url)
    if parsed.hostname != "github.com":
        raise ExpansionError("source archive requires a github.com repository URL")
    repository_path = parsed.path.strip("/")
    if repository_path.endswith(".git"):
        repository_path = repository_path[:-4]
    if repository_path.count("/") != 1:
        raise ExpansionError("invalid GitHub repository URL")
    parent_commit = case["source"]["parent_commit"]
    archive = destination.parent / f"{destination.name}.tar.gz"
    downloaded = run(
        [
            "curl",
            "--http1.1",
            "--silent",
            "--show-error",
            "--location",
            "--max-time",
            "300",
            "--output",
            str(archive),
            f"https://codeload.github.com/{repository_path}/tar.gz/{parent_commit}",
        ],
        destination.parent,
        timeout=310,
    )
    if downloaded["exit_code"] != 0:
        raise ExpansionError(downloaded["stderr"].strip())
    destination.mkdir()
    extracted = run(
        [
            "tar",
            "-xzf",
            str(archive),
            "--strip-components=1",
            "-C",
            str(destination),
        ],
        destination.parent,
    )
    archive.unlink()
    if extracted["exit_code"] != 0:
        raise ExpansionError(extracted["stderr"].strip())
    initialized = run(["git", "init", "--quiet"], destination)
    if initialized["exit_code"] != 0:
        raise ExpansionError(initialized["stderr"].strip())
    added = run(["git", "add", "--all"], destination)
    if added["exit_code"] != 0:
        raise ExpansionError(added["stderr"].strip())
    committed = run(
        [
            "git",
            "-c",
            "user.name=SilentSwap Runtime",
            "-c",
            "user.email=runtime@silentswap.invalid",
            "commit",
            "--quiet",
            "-m",
            f"Runtime mirror of {parent_commit}",
        ],
        destination,
    )
    if committed["exit_code"] != 0:
        raise ExpansionError(committed["stderr"].strip())
    remote = run(["git", "remote", "add", "origin", repository_url], destination)
    if remote["exit_code"] != 0:
        raise ExpansionError(remote["stderr"].strip())
    return destination


def clean_repository_artifacts(repository: Path) -> None:
    cleaned = run(["git", "clean", "-fdx"], repository)
    if cleaned["exit_code"] != 0:
        raise ExpansionError(f"failed to remove test artifacts: {cleaned['stderr'].strip()}")
    status = run(
        ["git", "status", "--short", "--untracked-files=all", "--ignored"],
        repository,
    )
    if status["exit_code"] != 0 or status["stdout"].strip():
        raise ExpansionError("original repository is not clean after test-artifact cleanup")


def apply_builder(repository: Path, builder: dict[str, Any]) -> None:
    target = repository / builder["target_file"]
    source = target.read_text(encoding="utf-8")
    original = builder["original_code_text"]
    if source.count(original) != 1:
        raise CandidateRejected(
            f"original_code_text occurs {source.count(original)} times in {builder['target_file']}"
        )
    target.write_text(source.replace(original, builder["replacement_code_text"], 1), encoding="utf-8")


def repository_patch(repository: Path) -> tuple[str, list[str]]:
    checked = run(["git", "diff", "--check"], repository)
    if checked["exit_code"] != 0:
        raise CandidateRejected("git diff --check failed")
    changed = run(["git", "diff", "--name-only"], repository)
    changed_files = changed["stdout"].splitlines()
    patch = run(["git", "diff", "--binary", "--", *changed_files], repository)
    if patch["exit_code"] != 0 or not patch["stdout"]:
        raise CandidateRejected("code replacement produced no patch")
    return patch["stdout"], changed_files


def assert_single_production_file(builder: dict[str, Any], changed_files: list[str]) -> None:
    if changed_files != [builder["target_file"]]:
        raise CandidateRejected("Builder changed more than its selected production file")
    parts = Path(builder["target_file"]).parts
    if "tests" in parts or Path(builder["target_file"]).name.startswith("test"):
        raise CandidateRejected("Builder patch modifies tests")


def semantic_test(
    runtime: dict[str, Any],
    repository: Path,
    source: str,
    path: Path,
) -> dict[str, Any]:
    path.write_text(source, encoding="utf-8")
    container_path = "/tmp/silentswap_semantic_test.py"
    return run_in_docker(
        runtime,
        repository,
        ["python", container_path],
        mounts=[(path, container_path, "ro")],
    )


def portable_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    copied = json.loads(json.dumps(evidence))
    for section in core.VERIFICATION_COMMAND_SECTIONS:
        copied[section]["command"] = (
            copied[section]["command"]
            .replace(str(RUNTIME_ROOT), PORTABLE_RUNTIME_ROOT)
            .replace(str(ROOT), PORTABLE_ROOT)
        )
    return copied


def warning_only_stderr(stderr: str) -> bool:
    lines = [line for line in stderr.splitlines() if line.strip()]
    warning_seen = False
    for line in lines:
        if re.match(r"^.+:\d+(?::\d+)?: [A-Za-z_][A-Za-z0-9_.]*Warning:", line):
            warning_seen = True
        elif warning_seen and line[:1].isspace():
            continue
        else:
            return False
    return warning_seen


def failure_fingerprint(result: dict[str, Any]) -> str:
    stderr = result["stderr"].strip()
    stdout = result["stdout"].strip()
    output = stdout if stdout and warning_only_stderr(stderr) else stderr or stdout
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        raise CandidateRejected("semantic test failed without diagnostic output")
    fingerprint = lines[-1]
    infrastructure_errors = (
        "ModuleNotFoundError",
        "ImportError",
        "SyntaxError",
        "IndentationError",
        "exec format error",
        "can't open file",
        "No such file or directory",
    )
    if any(marker in result["stderr"] for marker in infrastructure_errors):
        raise CandidateRejected("semantic test failed because of an environment or code-loading error")
    fingerprint = re.sub(
        r"/tmp/tmp[A-Za-z0-9._-]+(?=/|\s|$)", "<temporary-directory>", fingerprint
    )
    fingerprint = re.sub(r"'[^'\n]*'", "'<value>'", fingerprint)
    fingerprint = re.sub(r'"[^"\n]*"', '"<value>"', fingerprint)
    fingerprint = re.sub(r"(?<![A-Za-z_])-?\d+(?:\.\d+)?", "<number>", fingerprint)
    return fingerprint


def verify_individual_swap(
    sample: Path,
    runtime: dict[str, Any],
    builder: dict[str, Any],
    attempt: Path,
) -> tuple[str, dict[str, Any]]:
    original = runtime["original"]
    runtime_attempt = runtime["state"] / attempt.parent.name / attempt.name
    runtime_attempt.mkdir(parents=True, exist_ok=True)
    semantic_path = runtime_attempt / "semantic_swap_verification_test.py"
    semantic_original = semantic_test(
        runtime, original, builder["semantic_test"], semantic_path
    )
    if semantic_original["exit_code"] != 0:
        raise CandidateRejected("semantic test failed on original code")

    candidate = clone_repository(original, runtime_attempt / "candidate")
    apply_builder(candidate, builder)
    patch, changed_files = repository_patch(candidate)
    assert_single_production_file(builder, changed_files)
    after_swap = run_in_docker(runtime, candidate, runtime["test_command"])
    if after_swap["exit_code"] != 0:
        raise CandidateRejected("complete repository test suite failed after isolated swap")
    semantic_after = semantic_test(
        runtime, candidate, builder["semantic_test"], semantic_path
    )
    if semantic_after["exit_code"] == 0:
        raise CandidateRejected("semantic test did not detect the isolated swap")
    fingerprint = failure_fingerprint(semantic_after)
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
        "expected_failure_fingerprint": fingerprint,
    }
    return patch, portable_evidence(evidence)


def build_one_swap(
    sample: Path,
    record: dict[str, Any],
    runtime: dict[str, Any],
    api_key: str,
    slot: int,
    builders: list[dict[str, Any]],
    timeout: int,
    max_attempts: int,
) -> dict[str, Any]:
    slot_dir = WORK_ROOT / sample.name / f"slot_{slot}"
    accepted_path = slot_dir / "accepted.json"
    if accepted_path.exists():
        accepted = read_json(accepted_path)
        if "sample_number" not in accepted:
            accepted["sample_number"] = int(sample.name)
            write_json(accepted_path, accepted)
        return accepted
    rejected: list[dict[str, Any]] = []
    for path in sorted(slot_dir.glob("attempt_*/rejected.json")):
        rejected.append(read_json(path))

    for attempt_number in range(1, max_attempts + 1):
        attempt = slot_dir / f"attempt_{attempt_number}"
        if (attempt / "rejected.json").exists():
            continue
        attempt.mkdir(parents=True, exist_ok=True)
        try:
            selector_response, selector_usage = call_or_load_model(
                api_key,
                additional_selector_prompt(record, runtime["original"], builders, rejected),
                timeout,
                attempt,
                "selector",
            )
            selection = validate_additional_selection(
                selector_response, record, runtime["original"], builders
            )
            context_files = list(
                dict.fromkeys(selection["target_files"] + selection["related_test_files"])
            )
            context = core.repository_context(runtime["original"], context_files)
            builder, builder_usage = call_or_load_model(
                api_key,
                batch.builder_prompt(record, selection, context),
                timeout,
                attempt,
                "builder",
            )
            validate_additional_builder(builder, selection, builders)
            patch, evidence = verify_individual_swap(sample, runtime, builder, attempt)
            quality, checker_usage = call_or_load_model(
                api_key,
                batch.quality_prompt(selection, builder, patch, evidence),
                timeout,
                attempt,
                "quality_checker",
            )
            if set(quality) != {"verdict", "quality", "gold"} or quality["verdict"] != "accept":
                raise CandidateRejected("independent Quality Checker rejected the candidate")
            gold = core.prepare_generated_gold(quality["gold"])
            accepted = {
                "sample_number": int(sample.name),
                "slot": slot,
                "attempt": attempt_number,
                "selection": selection,
                "builder": builder,
                "individual_patch": patch,
                "individual_evidence": evidence,
                "quality_checker": quality,
                "gold": gold,
                "usage": {
                    "selector": selector_usage,
                    "builder": builder_usage,
                    "quality_checker": checker_usage,
                },
            }
            write_json(accepted_path, accepted)
            log(f"[accepted] sample={sample.name} slot={slot} attempt={attempt_number}")
            return accepted
        except CandidateRejected as error:
            rejected_item = {
                "attempt": attempt_number,
                "reason": str(error),
            }
            selector_file = attempt / "selector.json"
            if selector_file.exists():
                response = read_json(selector_file)
                if isinstance(response, dict) and isinstance(response.get("selection"), dict):
                    rejected_item.update(response["selection"])
            write_json(attempt / "rejected.json", rejected_item)
            rejected.append(rejected_item)
            log(f"[rejected] sample={sample.name} slot={slot} attempt={attempt_number} reason={error}")
    raise ExpansionError(f"sample {sample.name} slot {slot}: exhausted {max_attempts} candidates")


def reject_accepted(item: dict[str, Any], reason: str) -> None:
    sample_number = item["sample_number"]
    slot = item["slot"]
    attempt = item["attempt"]
    slot_dir = WORK_ROOT / str(sample_number) / f"slot_{slot}"
    rejected = {
        "attempt": attempt,
        "reason": reason,
        **item["selection"],
    }
    write_json(slot_dir / f"attempt_{attempt}" / "rejected.json", rejected)
    (slot_dir / "accepted.json").unlink(missing_ok=True)


def apply_patch(repository: Path, patch: str, state: Path) -> None:
    state.write_text(patch, encoding="utf-8")
    applied = run(["git", "apply", str(state)], repository)
    if applied["exit_code"] != 0:
        raise ExpansionError(f"patch application failed: {applied['stderr'].strip()}")


def build_final_candidate(
    sample: Path,
    runtime: dict[str, Any],
    original_builder: dict[str, Any],
    accepted: list[dict[str, Any]],
) -> tuple[Path, list[dict[str, Any]], str, list[str]]:
    state = WORK_ROOT / sample.name / "final"
    candidate = clone_repository(
        runtime["original"], runtime["state"] / "final" / "candidate"
    )
    patch_records = [
        {
            "slot": 1,
            "builder": original_builder,
            "individual_patch": (sample / "swap.patch").read_text(encoding="utf-8"),
        },
        *accepted,
    ]
    for record in patch_records:
        try:
            apply_builder(candidate, record["builder"])
        except CandidateRejected as error:
            raise CombinationRejected(
                f"slot {record['slot']} cannot coexist with prior swaps: {error}",
                max(2, record["slot"]),
            ) from error
    combined_patch, changed_files = repository_patch(candidate)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".patch") as handle:
        handle.write(combined_patch)
        handle.flush()
        check = run(["git", "apply", "--check", handle.name], runtime["original"])
    if check["exit_code"] != 0:
        raise ExpansionError("combined patch is not replayable on the original repository")
    for relative in changed_files:
        parts = Path(relative).parts
        if "tests" in parts or Path(relative).name.startswith("test"):
            raise ExpansionError(f"combined patch changes a test file: {relative}")
    return candidate, patch_records, combined_patch, changed_files


def counterfactual_verification(
    sample: Path,
    runtime: dict[str, Any],
    candidate: Path,
    patch_records: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    final_suite = run_in_docker(runtime, candidate, runtime["test_command"])
    if final_suite["exit_code"] != 0:
        raise CombinationRejected(
            "complete repository test suite failed with all five swaps",
            TARGET_SWAP_COUNT,
        )
    results = []
    final_state = runtime["state"] / "final" / "counterfactual"
    for record in patch_records:
        slot = record["slot"]
        builder = record["builder"]
        test_path = runtime["state"] / "final" / f"semantic_test_{slot}.py"
        on_original = semantic_test(
            runtime,
            runtime["original"],
            builder["semantic_test"],
            test_path,
        )
        if on_original["exit_code"] != 0:
            raise ExpansionError(f"slot {slot} semantic test failed on original code")

        only_this = clone_repository(
            runtime["original"], final_state / f"only_slot_{slot}"
        )
        apply_builder(only_this, builder)
        on_only_this = semantic_test(
            runtime, only_this, builder["semantic_test"], test_path
        )
        if on_only_this["exit_code"] == 0:
            raise CombinationRejected(
                f"slot {slot} semantic test did not fail when only its own swap was applied",
                max(2, slot),
            )
        try:
            isolated_fingerprint = failure_fingerprint(on_only_this)
        except CandidateRejected as error:
            raise CombinationRejected(
                f"slot {slot} isolated semantic test has an environment or "
                f"code-loading failure",
                max(2, slot),
            ) from error

        without_this = clone_repository(
            runtime["original"], final_state / f"without_slot_{slot}"
        )
        other_slots = [item["slot"] for item in patch_records if item["slot"] != slot]
        for other in patch_records:
            if other["slot"] == slot:
                continue
            apply_builder(without_this, other["builder"])
        on_without_this = semantic_test(
            runtime, without_this, builder["semantic_test"], test_path
        )
        if on_without_this["exit_code"] != 0:
            regenerate = diagnose_interfering_slot(
                sample,
                runtime,
                builder,
                slot,
                [item for item in patch_records if item["slot"] != slot],
                test_path,
                final_state,
            )
            raise CombinationRejected(
                f"slot {slot} semantic test fails when its own swap is absent; "
                f"interference first appears at slot {regenerate}",
                regenerate,
            )

        on_final = semantic_test(runtime, candidate, builder["semantic_test"], test_path)
        if on_final["exit_code"] == 0:
            regenerate = diagnose_suppressing_slot(
                sample,
                runtime,
                record,
                [item for item in patch_records if item["slot"] != slot],
                test_path,
                final_state,
                isolated_fingerprint,
            )
            raise CombinationRejected(
                f"slot {slot} semantic test did not detect the final combined swaps",
                regenerate,
            )
        try:
            final_fingerprint = failure_fingerprint(on_final)
        except CandidateRejected as error:
            regenerate = diagnose_suppressing_slot(
                sample,
                runtime,
                record,
                [item for item in patch_records if item["slot"] != slot],
                test_path,
                final_state,
                isolated_fingerprint,
            )
            raise CombinationRejected(
                f"slot {slot} semantic test has an environment or code-loading "
                f"failure only in the combined repository",
                regenerate,
            ) from error
        if final_fingerprint != isolated_fingerprint:
            regenerate = diagnose_suppressing_slot(
                sample,
                runtime,
                record,
                [item for item in patch_records if item["slot"] != slot],
                test_path,
                final_state,
                isolated_fingerprint,
            )
            raise CombinationRejected(
                f"slot {slot} fails differently in the combined repository: "
                f"{isolated_fingerprint!r} != {final_fingerprint!r}",
                regenerate,
            )
        results.append(
            {
                "slot": slot,
                "semantic_test_on_original": on_original,
                "semantic_test_with_only_this_swap": on_only_this,
                "semantic_test_with_other_four_swaps": on_without_this,
                "semantic_test_after_all_swaps": on_final,
                "failure_fingerprint": isolated_fingerprint,
                "behavior_difference": {
                    "original": on_original["stdout"] + on_original["stderr"],
                    "swapped": on_final["stdout"] + on_final["stderr"],
                },
            }
        )
    return final_suite, results


def diagnose_interfering_slot(
    sample: Path,
    runtime: dict[str, Any],
    tested_builder: dict[str, Any],
    tested_slot: int,
    other_records: list[dict[str, Any]],
    test_path: Path,
    state: Path,
) -> int:
    probe = clone_repository(
        runtime["original"], state / f"diagnose_slot_{tested_slot}"
    )
    for other in sorted(other_records, key=lambda item: item["slot"]):
        apply_builder(probe, other["builder"])
        result = semantic_test(
            runtime, probe, tested_builder["semantic_test"], test_path
        )
        if result["exit_code"] == 0:
            continue
        if other["slot"] == 1:
            if tested_slot == 1:
                candidates = [item["slot"] for item in other_records if item["slot"] > 1]
                return min(candidates)
            return tested_slot
        return other["slot"]
    candidates = [item["slot"] for item in other_records if item["slot"] > 1]
    return max(candidates) if candidates else max(2, tested_slot)


def diagnose_suppressing_slot(
    sample: Path,
    runtime: dict[str, Any],
    tested_record: dict[str, Any],
    other_records: list[dict[str, Any]],
    test_path: Path,
    state: Path,
    expected_fingerprint: str,
) -> int:
    included = [tested_record]
    for other in sorted(other_records, key=lambda item: item["slot"]):
        included.append(other)
        probe = clone_repository(
            runtime["original"],
            state / f"diagnose_suppression_{tested_record['slot']}_{other['slot']}",
        )
        for item in sorted(included, key=lambda value: value["slot"]):
            apply_builder(probe, item["builder"])
        result = semantic_test(
            runtime,
            probe,
            tested_record["builder"]["semantic_test"],
            test_path,
        )
        changed = result["exit_code"] == 0
        if not changed:
            try:
                changed = failure_fingerprint(result) != expected_fingerprint
            except CandidateRejected:
                changed = True
        if not changed:
            continue
        if other["slot"] == 1:
            return max(2, tested_record["slot"])
        return other["slot"]
    candidates = [item["slot"] for item in other_records if item["slot"] > 1]
    return max(candidates) if candidates else max(2, tested_record["slot"])


def individual_swapped_source(
    source_repository: Path, patch: str, target_file: str, state: Path
) -> str:
    worktree = state / f"isolated_{len(list(state.glob('isolated_*')))}"
    worktree.mkdir(parents=True)
    source = source_repository / target_file
    destination = worktree / target_file
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    patch_path = worktree / "swap.patch"
    patch_path.write_text(patch, encoding="utf-8")
    applied = run(["git", "apply", str(patch_path)], worktree)
    if applied["exit_code"] != 0:
        raise ExpansionError("individual patch cannot be replayed for localization")
    return destination.read_text(encoding="utf-8")


def map_changed_lines(isolated: str, final: str, lines: set[int]) -> set[int]:
    matcher = difflib.SequenceMatcher(a=isolated.splitlines(), b=final.splitlines())
    mapped: set[int] = set()
    for tag, first_start, first_end, second_start, _ in matcher.get_opcodes():
        if tag != "equal":
            continue
        for line in lines:
            zero_based = line - 1
            if first_start <= zero_based < first_end:
                mapped.add(second_start + zero_based - first_start + 1)
    if len(mapped) != len(lines):
        raise ExpansionError("responsible lines could not be mapped into the combined patch")
    return mapped


def symbol_for_responsible_lines(
    target_file: str, source: str, responsible: set[int]
) -> dict[str, Any]:
    if Path(target_file).suffix != ".py":
        return {"kind": "module", "qualified_name": []}
    tree = ast.parse(source, filename=target_file)
    parents = evaluation._ast_parents(tree)
    symbols = [
        evaluation.symbol_for_line(tree, parents, line)[0]
        for line in sorted(responsible)
    ]
    return (
        symbols[0]
        if all(item == symbols[0] for item in symbols[1:])
        else evaluation.common_symbol_for_lines(tree, parents, responsible)
    )


def localization_for_patch(
    source_repository: Path,
    candidate: Path,
    target_file: str,
    patch: str,
    state: Path,
) -> dict[str, Any]:
    patch_lines = evaluation.changed_lines_from_patch(patch)
    if set(patch_lines) != {target_file}:
        raise ExpansionError("individual patch paths differ from its target file")
    isolated = individual_swapped_source(source_repository, patch, target_file, state)
    final = (candidate / target_file).read_text(encoding="utf-8")
    responsible = map_changed_lines(isolated, final, patch_lines[target_file])
    return {
        "file": target_file,
        "symbol": symbol_for_responsible_lines(target_file, final, responsible),
        "line_ranges": evaluation.ranges_from_lines(responsible),
    }


def sanitize_result(result: dict[str, Any]) -> dict[str, Any]:
    copied = json.loads(json.dumps(result))

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "command" and isinstance(item, str):
                    value[key] = (
                        item.replace(str(RUNTIME_ROOT), PORTABLE_RUNTIME_ROOT)
                        .replace(str(ROOT), PORTABLE_ROOT)
                    )
                else:
                    walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(copied)
    return copied


def build_artifacts(
    sample: Path,
    runtime: dict[str, Any],
    original_builder: dict[str, Any],
    original_quality: dict[str, Any],
    accepted: list[dict[str, Any]],
    candidate: Path,
    patch_records: list[dict[str, Any]],
    combined_patch: str,
    changed_files: list[str],
    final_suite: dict[str, Any],
    final_results: list[dict[str, Any]],
    removal: dict[str, Any],
) -> dict[str, str]:
    builders = [original_builder, *(item["builder"] for item in accepted)]
    original_gold = read_json(sample / "gold.json")
    quality_gold = [original_gold, *(item["gold"] for item in accepted)]
    state = WORK_ROOT / sample.name / "final" / "localization"
    state.mkdir(parents=True, exist_ok=True)
    gold_items = []
    for record, gold in zip(patch_records, quality_gold, strict=True):
        item = dict(gold)
        item["localization"] = localization_for_patch(
            runtime["original"],
            candidate,
            record["builder"]["target_file"],
            record["individual_patch"],
            state,
        )
        gold_items.append(item)

    final_by_slot = {item["slot"]: item for item in final_results}
    evidence_items = []
    original_evidence = read_json(sample / "verification_evidence.json")
    individual_items = [original_evidence, *(item["individual_evidence"] for item in accepted)]
    qualities = [original_quality, *(item["quality_checker"] for item in accepted)]
    for record, individual, quality in zip(
        patch_records, individual_items, qualities, strict=True
    ):
        slot = record["slot"]
        final = final_by_slot[slot]
        evidence_items.append(
            {
                "swap_number": slot,
                "title": record["builder"]["title"],
                "original_suite_on_original": sanitize_result(runtime["baseline"]),
                "semantic_test_source": record["builder"]["semantic_test"],
                "semantic_test_on_original": sanitize_result(final["semantic_test_on_original"]),
                "original_suite_after_swap": sanitize_result(final_suite),
                "semantic_test_after_swap": sanitize_result(final["semantic_test_after_all_swaps"]),
                "changed_files": [record["builder"]["target_file"]],
                "behavior_difference": final["behavior_difference"],
                "counterfactual_attribution": {
                    "only_this_swap": sanitize_result(
                        final["semantic_test_with_only_this_swap"]
                    ),
                    "other_four_without_this_swap": sanitize_result(
                        final["semantic_test_with_other_four_swaps"]
                    ),
                    "all_five_swaps": sanitize_result(
                        final["semantic_test_after_all_swaps"]
                    ),
                    "expected_failure_fingerprint": final["failure_fingerprint"],
                },
                "individual_validation": individual,
                "quality_checker": quality,
            }
        )

    case = read_json(sample / "case.json")
    case["status"] = "validated"
    case["mapping"] = {
        "swap_count": TARGET_SWAP_COUNT,
        "swaps": [
            {
                "swap_number": index,
                "document_text": builder["original_document_text"],
                "target_file": builder["target_file"],
                "semantic_test": f"verification_evidence.json#swaps/{index - 1}/semantic_test_source",
            }
            for index, builder in enumerate(builders, 1)
        ],
    }
    case.setdefault("provenance", {})["multi_swap_expansion"] = {
        "construction": "four_independent_controlled_injections",
        "model": MODEL,
        "thinking_enabled": True,
        "selector_mode": "complete_document_and_repository_index_per_swap",
        "swap_count": TARGET_SWAP_COUNT,
        "usage": [item["usage"] for item in accepted],
    }
    case["test_environment"]["container_python"] = runtime["python_version"]
    case["test_environment"]["image_removed_after_validation"] = True

    evidence = {
        "swap_count": TARGET_SWAP_COUNT,
        "original_suite_on_original": sanitize_result(runtime["baseline"]),
        "original_suite_after_all_swaps": sanitize_result(final_suite),
        "changed_files": changed_files,
        "swaps": evidence_items,
        "docker_image_lifecycle": {
            "image": runtime["image"],
            "pull_image": runtime["pull_image"],
            "platform": runtime["platform"],
            "pull": sanitize_result(runtime["image_pull"]),
            "tag_as_source_image": sanitize_result(runtime.get("image_tag", {})),
            "remove_after_validation": sanitize_result(removal),
        },
    }
    gold = {"swap_count": TARGET_SWAP_COUNT, "swaps": gold_items}
    trajectory = (sample / "construction_trajectory.jsonl").read_text(encoding="utf-8").rstrip("\n")
    new_events = []
    for item in accepted:
        new_events.extend(
            [
                {
                    "time": utc_now(),
                    "actor": "deepseek_selector",
                    "action": "select_additional_constraint",
                    "swap_number": item["slot"],
                    "output": item["selection"],
                },
                {
                    "time": utc_now(),
                    "actor": "deepseek_builder",
                    "action": "construct_additional_swap",
                    "swap_number": item["slot"],
                    "output": item["builder"],
                },
                {
                    "time": utc_now(),
                    "actor": "pipeline",
                    "action": "verify_additional_swap",
                    "swap_number": item["slot"],
                    "result": item["individual_evidence"],
                },
                {
                    "time": utc_now(),
                    "actor": "deepseek_quality_checker",
                    "action": "judge_additional_swap",
                    "swap_number": item["slot"],
                    "output": item["quality_checker"],
                },
            ]
        )
    new_events.extend(
        [
            {
                "time": utc_now(),
                "actor": "pipeline",
                "action": "verify_all_five_swaps",
                "result": evidence,
            },
            {
                "time": utc_now(),
                "actor": "pipeline",
                "action": "remove_docker_image_after_multi_swap_validation",
                "result": sanitize_result(removal),
            },
        ]
    )
    trajectory += "\n" + "\n".join(json.dumps(event, ensure_ascii=False) for event in new_events) + "\n"
    return {
        "swap.patch": combined_patch,
        "gold.json": json.dumps(gold, ensure_ascii=False, indent=2) + "\n",
        "case.json": json.dumps(case, ensure_ascii=False, indent=2) + "\n",
        "verification_evidence.json": json.dumps(evidence, ensure_ascii=False, indent=2) + "\n",
        "construction_trajectory.jsonl": trajectory,
    }


def publish(sample: Path, outputs: dict[str, str]) -> None:
    state = WORK_ROOT / sample.name
    backup = state / "backup"
    staged = state / "package"
    backup.mkdir(parents=True, exist_ok=True)
    staged.mkdir(parents=True, exist_ok=True)
    for name, content in outputs.items():
        backup_path = backup / name
        if not backup_path.exists():
            shutil.copy2(sample / name, backup_path)
        (staged / name).write_text(content, encoding="utf-8")
    for name in outputs:
        (staged / name).replace(sample / name)


def expand_sample(
    number: int,
    api_key: str,
    docker: str,
    timeout: int,
    max_attempts: int,
) -> None:
    sample = DATA_ROOT / str(number)
    hydrate_sample(sample)
    case = read_json(sample / "case.json")
    if (
        case.get("status") == "validated"
        and case.get("mapping", {}).get("swap_count") == TARGET_SWAP_COUNT
    ):
        log(f"[skip] sample={number} already validated")
        return
    runtime = runtime_from_case(sample, docker)
    runtime["state"] = RUNTIME_ROOT / sample.name
    repository = runtime["state"] / "original"
    original_builder = load_original_builder(sample)
    original_quality = load_original_quality(sample)
    accepted: list[dict[str, Any]] = []
    try:
        runtime["original"] = clone_source_repository(
            case, repository
        )
        clean_repository_artifacts(runtime["original"])
        log(f"[docker:pull] sample={number} image={runtime['image']}")
        pull_and_baseline(runtime, runtime["original"])
        log(f"[baseline:pass] sample={number}")
        record = case_record(sample)
        builders = [original_builder]
        slot = 2
        while slot <= TARGET_SWAP_COUNT:
            item = build_one_swap(
                sample,
                record,
                runtime,
                api_key,
                slot,
                builders,
                timeout,
                max_attempts,
            )
            accepted.append(item)
            builders.append(item["builder"])
            slot += 1
            if slot <= TARGET_SWAP_COUNT:
                continue
            try:
                candidate, patch_records, combined_patch, changed_files = build_final_candidate(
                    sample, runtime, original_builder, accepted
                )
                final_suite, final_results = counterfactual_verification(
                    sample, runtime, candidate, patch_records
                )
            except CombinationRejected as error:
                regenerate = error.regenerate_from_slot
                discarded = [item for item in accepted if item["slot"] >= regenerate]
                if not discarded:
                    raise
                for discarded_item in discarded:
                    reject_accepted(discarded_item, f"combined counterfactual rejection: {error}")
                accepted = [item for item in accepted if item["slot"] < regenerate]
                builders = [original_builder, *(item["builder"] for item in accepted)]
                slot = regenerate
                log(
                    f"[combination:rejected] sample={number} regenerate_from={regenerate} "
                    f"reason={error}"
                )
                continue
        log(f"[final:pass] sample={number} swaps={TARGET_SWAP_COUNT}")
        clean_repository_artifacts(runtime["original"])
        removal = remove_image(runtime, runtime["original"])
        if removal["exit_code"] != 0:
            raise ExpansionError("Docker image removal failed")
        trim_docker_storage(runtime)
        outputs = build_artifacts(
            sample,
            runtime,
            original_builder,
            original_quality,
            accepted,
            candidate,
            patch_records,
            combined_patch,
            changed_files,
            final_suite,
            final_results,
            removal,
        )
        publish(sample, outputs)
        shutil.rmtree(runtime["state"])
        write_json(
            WORK_ROOT / sample.name / "complete.json",
            {"sample_number": number, "swap_count": TARGET_SWAP_COUNT},
        )
        log(f"[published] sample={number} swaps={TARGET_SWAP_COUNT}")
    except Exception as error:
        try:
            if repository.exists() and (repository / ".git").exists():
                clean_repository_artifacts(repository)
        except Exception as cleanup_error:
            write_json(
                WORK_ROOT / sample.name / "repository_cleanup_failure.json",
                {"error": str(cleanup_error)},
            )
        if runtime.get("image_pull") and not runtime["image_removal_attempted"]:
            removal = remove_image(runtime, runtime["state"])
            write_json(WORK_ROOT / sample.name / "failure_cleanup.json", removal)
            if removal["exit_code"] == 0:
                trim = trim_docker_storage(runtime)
                write_json(WORK_ROOT / sample.name / "failure_trim.json", trim)
        write_json(
            WORK_ROOT / sample.name / "runtime_failure.json",
            {
                key: sanitize_result(runtime[key])
                for key in (
                    "image_pull",
                    "image_tag",
                    "baseline",
                    "image_removal",
                    "disk_trim",
                )
                if key in runtime
            },
        )
        if runtime["state"].exists():
            shutil.rmtree(runtime["state"])
        raise error


def parse_sample_numbers(value: str) -> list[int]:
    numbers: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            numbers.update(range(int(start_text), int(end_text) + 1))
        elif part:
            numbers.add(int(part))
    if not numbers or min(numbers) < 1 or max(numbers) > 100:
        raise argparse.ArgumentTypeError("samples must be within 1-100")
    return sorted(numbers)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=parse_sample_numbers, default=list(range(1, 101)))
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    parser.add_argument("--docker", default="docker")
    parser.add_argument("--api-key-stdin", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = input().strip() if args.api_key_stdin else os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise ExpansionError("DEEPSEEK_API_KEY is required")
    docker_version = run([args.docker, "version", "--format", "{{.Server.Version}}"], ROOT, 60)
    if docker_version["exit_code"] != 0:
        raise ExpansionError("Docker daemon is unavailable")
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    failures: list[tuple[int, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                expand_sample,
                number,
                api_key,
                args.docker,
                args.timeout,
                args.max_attempts,
            ): number
            for number in args.samples
        }
        for future in as_completed(futures):
            number = futures[future]
            try:
                future.result()
            except Exception as error:
                failures.append((number, str(error)))
                write_json(
                    WORK_ROOT / str(number) / "failure.json",
                    {"sample_number": number, "error": str(error)},
                )
                log(f"[failed] sample={number} error={error}")
    if failures:
        joined = ", ".join(str(number) for number, _ in sorted(failures))
        raise ExpansionError(f"failed samples: {joined}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
