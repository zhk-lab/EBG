#!/usr/bin/env python3
"""Build three validated SilentSwap samples from DeNovoSWE records."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfFileSystem


ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = "datasets/AweAI-Team/DeNovoSWE/denovoswe_public.jsonl"
MODEL = "deepseek-v4-flash"
SWAP_TYPES = (
    "input_validation_boundary",
    "parsing_matching",
    "ordering_precedence",
    "default_null_fallback",
    "exception_error_handling",
    "state_identity_lifecycle",
    "representation_normalization",
    "dispatch_routing_aggregation",
)
GENERATED_GOLD_FIELDS = {
    "swap_type",
    "original_semantics",
    "swapped_semantics",
    "why_different",
    "evidence",
}
VERIFICATION_COMMAND_SECTIONS = (
    "original_suite_on_original",
    "semantic_test_on_original",
    "original_suite_after_swap",
    "semantic_test_after_swap",
)
PORTABLE_DATASET_ROOT = "${SILENTSWAP_ROOT}"

SAMPLES = [
    {
        "instance_id": "singer-io_singer-python_pr183",
        "category": "full_data_to_sample",
        "python": "/usr/bin/python3",
        "install": ["pip", "install", "-e", "."],
        "test": ["python", "-m", "unittest", "discover", "-s", "tests"],
        "context_files": ["README.md", "singer/messages.py", "tests/test_singer.py"],
    },
    {
        "instance_id": "adamchainz_django-cors-headers_pr451",
        "category": "authorization_scope",
        "python": "/usr/bin/python3",
        "install": [
            "pip", "install", "Django==3.2.25", "pytest==7.4.4",
            "pytest-cov==4.1.0", "pytest-django==4.11.1",
        ],
        "test": ["python", "runtests.py", "-q"],
        "document_sections": ["## Origin access control"],
        "context_files": ["corsheaders/middleware.py", "tests/test_middleware.py"],
    },
    {
        "instance_id": "mapbox_mercantile_pr130",
        "category": "numeric_precision",
        "python": "python3",
        "install": [
            "pip", "install", "pytest==8.4.2", "hypothesis", "pytest-cov",
            "responses", "click",
        ],
        "test": ["python", "-m", "pytest", "-q", "-p", "no:pytest_durations"],
        "document_sections": [
            "## Geographic and Mercator transforms",
            "## Tile bounds and indexing",
        ],
        "context_files": ["mercantile/__init__.py", "tests/test_cli.py", "tests/test_funcs.py"],
    },
]


class PipelineError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run(command: list[str], cwd: Path, env: dict[str, str] | None = None, timeout: int = 900) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
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
        "counts": parse_test_counts(completed.stdout + completed.stderr),
    }


def parse_test_counts(output: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name in ("passed", "failed", "skipped", "xfailed", "errors"):
        matches = re.findall(rf"(\d+) {name}", output)
        if matches:
            counts[name] = int(matches[-1])
    unittest_match = re.search(r"Ran (\d+) tests?", output)
    if unittest_match:
        counts["ran"] = int(unittest_match.group(1))
    return counts


def parse_json_object(text: str) -> dict[str, Any]:
    content = text.strip()
    if not content:
        raise PipelineError("DeepSeek returned an empty final answer")
    if content.startswith("```"):
        lines = content.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines.pop()
        content = "\n".join(lines)
    value = json.loads(content)
    if not isinstance(value, dict):
        raise PipelineError("DeepSeek output must be a JSON object")
    return value


def call_deepseek(
    api_key: str,
    prompt: str,
    timeout: int,
    *,
    max_tokens: int = 16000,
    reasoning_effort: str = "medium",
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You construct controlled SilentSwap benchmark cases. Work only from supplied evidence. "
                    "Return one JSON object, without Markdown. The code must remain runnable while changing "
                    "one documented semantic. Do not modify repository tests or add dependencies."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "thinking": {"type": "enabled"},
        "reasoning_effort": reasoning_effort,
        "response_format": {"type": "json_object"},
        "max_tokens": max_tokens,
    }
    marker = "__HTTP_STATUS__="
    with tempfile.NamedTemporaryFile("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
        handle.flush()
        completed = subprocess.run(
            [
                "curl", "--http1.1", "--silent", "--show-error", "--max-time", str(timeout),
                "--request", "POST", "--header", "Content-Type: application/json",
                "--header", "@-", "--data-binary", f"@{handle.name}",
                "--write-out", f"\n{marker}%{{http_code}}",
                "https://api.deepseek.com/chat/completions",
            ],
            input=f"Authorization: Bearer {api_key}\n",
            capture_output=True,
            text=True,
            timeout=timeout + 10,
            check=False,
        )
    body, separator, status_text = completed.stdout.rpartition(f"\n{marker}")
    if completed.returncode != 0 or not separator or status_text.strip() != "200":
        raise PipelineError(completed.stderr.strip() or body[:500] or "DeepSeek request failed")
    response = json.loads(body)
    message = response["choices"][0]["message"]
    if not message.get("reasoning_content"):
        raise PipelineError("DeepSeek thinking mode returned no reasoning content")
    return parse_json_object(message["content"]), response.get("usage", {})


def load_source_records(instance_ids: set[str]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    fs = HfFileSystem()
    with fs.open(DATASET_PATH, "rb") as handle:
        for line in handle:
            row = json.loads(line)
            instance_id = row["instance_id"]
            if instance_id in instance_ids:
                records[instance_id] = row
                if len(records) == len(instance_ids):
                    break
    missing = instance_ids - records.keys()
    if missing:
        raise PipelineError(f"missing DeNovoSWE records: {', '.join(sorted(missing))}")
    return records


def prepare_repository(record: dict[str, Any], state_dir: Path) -> tuple[Path, Path]:
    original = state_dir / "original_repo"
    swapped = state_dir / "swapped_repo"
    for path in (original, swapped):
        if not path.exists():
            result = run(["git", "clone", "--quiet", record["github_url"], str(path)], state_dir)
            if result["exit_code"] != 0:
                raise PipelineError(result["stderr"])
            checkout = run(["git", "checkout", "--quiet", record["parent_commit"]], path)
            if checkout["exit_code"] != 0:
                raise PipelineError(checkout["stderr"])
        head = run(["git", "rev-parse", "HEAD"], path)
        if head["stdout"].strip() != record["parent_commit"]:
            raise PipelineError(f"unexpected commit in {path}")
    return original, swapped


def prepare_venv(config: dict[str, Any], repo: Path, state_dir: Path) -> Path:
    venv = state_dir / "venv"
    if not (venv / "bin" / "python").exists():
        result = run([config["python"], "-m", "venv", str(venv)], state_dir)
        if result["exit_code"] != 0:
            raise PipelineError(result["stderr"])
        commands = [config["install"]]
        if config.get("install_project"):
            commands.append(config["install_project"])
        for command in commands:
            expanded = [str(venv / "bin" / item) if item == "pip" else item for item in command]
            result = run(expanded, repo, timeout=1200)
            if result["exit_code"] != 0:
                raise PipelineError(result["stderr"])
    return venv


def expand_command(command: list[str], venv: Path) -> list[str]:
    return [str(venv / "bin" / item) if item in {"python", "pip"} else item for item in command]


def repository_context(repo: Path, paths: list[str]) -> str:
    sections = []
    for relative in paths:
        path = repo / relative
        sections.append(f"===== {relative} =====\n{path.read_text(encoding='utf-8')}")
    return "\n\n".join(sections)


def document_context(document: str, headings: list[str] | None) -> str:
    if not headings:
        return document
    sections = []
    for heading in headings:
        start = document.index(heading)
        remaining = document[start + len(heading):]
        next_section = re.search(r"^## ", remaining, re.MULTILINE)
        end = start + len(heading) + next_section.start() if next_section else len(document)
        sections.append(document[start:end].rstrip())
    return "\n\n".join(sections)


def builder_prompt(record: dict[str, Any], config: dict[str, Any], context: str) -> str:
    contract = {
        "title": "short case title",
        "original_document_text": "one exact, unique substring copied from DOCUMENT",
        "replacement_document_text": "minimal replacement describing the changed behavior",
        "target_file": "one production-code path",
        "original_code_text": "one exact, unique production-code substring copied from REPOSITORY FILES",
        "replacement_code_text": "minimal replacement code implementing the semantic swap",
        "semantic_test": "standalone Python source: pass on original code, fail on swapped code, print observed values",
        "gold": {
            "swap_type": f"one of: {', '.join(SWAP_TYPES)}",
            "original_semantics": "documented behavior",
            "swapped_semantics": "implemented replacement",
            "why_different": "meaningful consequence",
        },
    }
    document = document_context(record["document"], config.get("document_sections"))
    return (
        f"Create one {config['category']} SilentSwap for this real repository.\n"
        "Rules:\n"
        "- Change one production file with the smallest practical patch.\n"
        "- Do not modify tests, documentation, packaging, or dependencies.\n"
        "- Existing repository tests should still pass.\n"
        "- semantic_test must pass on the original commit and fail after the patch.\n"
        "- original_document_text must occur exactly once in DOCUMENT.\n"
        "- original_code_text must occur exactly once in target_file.\n"
        "- The replacement document text must honestly describe the new behavior.\n"
        "- Use only APIs and facts visible below.\n\n"
        f"OUTPUT:\n{json.dumps(contract, ensure_ascii=False, indent=2)}\n\n"
        f"INSTANCE_ID: {record['instance_id']}\n"
        f"REPOSITORY: {record['github_url']}\n"
        f"PARENT_COMMIT: {record['parent_commit']}\n\n"
        f"DOCUMENT EXCERPTS:\n{document}\n\n"
        f"REPOSITORY FILES:\n{context}"
    )


def checker_prompt(record: dict[str, Any], builder: dict[str, Any], evidence: dict[str, Any]) -> str:
    contract = {
        "swap_type": f"one of: {', '.join(SWAP_TYPES)}",
        "original_semantics": "original documented behavior",
        "swapped_semantics": "behavior after code patch",
        "why_different": "semantic consequence",
        "evidence": ["specific document, patch, and test references"],
    }
    return (
        "Independently summarize the validated semantic substitution. Return exactly the OUTPUT keys. "
        "Do not add a monitor question or verification card.\n\n"
        f"OUTPUT:\n{json.dumps(contract, ensure_ascii=False, indent=2)}\n\n"
        f"ORIGINAL DOCUMENT:\n{record['document']}\n\n"
        f"BUILDER ARTIFACTS:\n{json.dumps(builder, ensure_ascii=False, indent=2)}\n\n"
        f"EXECUTION EVIDENCE:\n{json.dumps(evidence, ensure_ascii=False, indent=2)}"
    )


def replace_once(text: str, original: str, replacement: str, name: str) -> str:
    if text.count(original) != 1:
        raise PipelineError(f"{name} must occur exactly once")
    return text.replace(original, replacement, 1)


def validate_builder_output(builder: dict[str, Any]) -> None:
    required = {
        "title", "original_document_text", "replacement_document_text", "target_file",
        "original_code_text", "replacement_code_text", "semantic_test", "gold",
    }
    if set(builder) != required:
        raise PipelineError("unexpected DeepSeek Builder output fields")
    if "test_" in builder["target_file"] or "tests" in Path(builder["target_file"]).parts:
        raise PipelineError("Builder patch modifies tests")


def prepare_generated_gold(gold: dict[str, Any]) -> dict[str, Any]:
    if set(gold) != GENERATED_GOLD_FIELDS:
        raise PipelineError("unexpected generated gold fields")
    if gold["swap_type"] not in SWAP_TYPES:
        raise PipelineError(f"unsupported swap_type: {gold['swap_type']}")
    return {
        **gold,
        "gold_source": "model_generated",
        "adjudication_status": "pending",
    }


def sanitize_verification_commands(evidence: dict[str, Any]) -> None:
    for section in VERIFICATION_COMMAND_SECTIONS:
        evidence[section]["command"] = evidence[section]["command"].replace(
            str(ROOT), PORTABLE_DATASET_ROOT
        )


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sample_directories(data_root: Path) -> list[Path]:
    return sorted(
        (path for path in data_root.iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name),
    )


def find_sample_directory(data_root: Path, case_id: str) -> Path | None:
    for sample_dir in sample_directories(data_root):
        case_path = sample_dir / "case.json"
        if case_path.exists():
            case = json.loads(case_path.read_text(encoding="utf-8"))
            if case.get("case_id") == case_id:
                return sample_dir
    return None


def next_sample_number(data_root: Path) -> int:
    numbers = {int(path.name) for path in sample_directories(data_root)}
    number = 1
    while number in numbers:
        number += 1
    return number


def build_sample(record: dict[str, Any], config: dict[str, Any], api_key: str, output_root: Path, work_root: Path, resume: bool, timeout: int) -> None:
    instance_id = record["instance_id"]
    existing = find_sample_directory(output_root, instance_id)
    if resume and existing is not None:
        print(f"[skip] {instance_id}", flush=True)
        return

    state_dir = work_root / instance_id
    state_dir.mkdir(parents=True, exist_ok=True)
    trajectory: list[dict[str, Any]] = []
    trajectory.append({"time": utc_now(), "actor": "pipeline", "action": "load_source", "instance_id": instance_id})

    original_repo, swapped_repo = prepare_repository(record, state_dir)
    venv = prepare_venv(config, original_repo, state_dir)
    test_command = expand_command(config["test"], venv)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(original_repo)

    baseline = run(test_command, original_repo, env=env, timeout=1200)
    if baseline["exit_code"] != 0:
        raise PipelineError(f"baseline tests failed for {instance_id}")
    trajectory.append({"time": utc_now(), "actor": "pipeline", "action": "baseline_test", "result": baseline})

    builder_state = state_dir / "builder.json"
    if resume and builder_state.exists():
        builder = json.loads(builder_state.read_text(encoding="utf-8"))
        builder_usage = json.loads((state_dir / "builder_usage.json").read_text(encoding="utf-8"))
    else:
        context = repository_context(original_repo, config["context_files"])
        builder, builder_usage = call_deepseek(api_key, builder_prompt(record, config, context), timeout)
        validate_builder_output(builder)
        write_json(builder_state, builder)
        write_json(state_dir / "builder_usage.json", builder_usage)
    validate_builder_output(builder)
    trajectory.append({"time": utc_now(), "actor": "deepseek_builder", "action": "construct_swap", "output": builder})

    patch_path = state_dir / "swap.patch"
    if patch_path.exists():
        restored = run(["git", "apply", "--reverse", str(patch_path)], swapped_repo)
        if restored["exit_code"] != 0:
            raise PipelineError(f"cannot restore interrupted swap for {instance_id}")

    replace_once(
        record["document"], builder["original_document_text"], builder["replacement_document_text"],
        "original_document_text",
    )
    semantic_test_path = state_dir / "semantic_swap_verification_test.py"
    semantic_test_path.write_text(builder["semantic_test"], encoding="utf-8")

    semantic_original = run([str(venv / "bin" / "python"), str(semantic_test_path)], original_repo, env=env)
    if semantic_original["exit_code"] != 0:
        raise PipelineError(f"semantic test does not pass on original code for {instance_id}")

    target_path = swapped_repo / builder["target_file"]
    target_source = target_path.read_text(encoding="utf-8")
    target_path.write_text(
        replace_once(
            target_source, builder["original_code_text"], builder["replacement_code_text"],
            "original_code_text",
        ),
        encoding="utf-8",
    )
    patch_result = run(["git", "diff", "--", builder["target_file"]], swapped_repo)
    swap_patch = patch_result["stdout"]
    if not swap_patch:
        raise PipelineError("DeepSeek code replacement produced no diff")
    patch_path.write_text(swap_patch, encoding="utf-8")
    apply_check = run(["git", "apply", "--check", str(patch_path)], original_repo)
    if apply_check["exit_code"] != 0:
        raise PipelineError(f"generated patch is not replayable for {instance_id}")

    changed = run(["git", "diff", "--name-only"], swapped_repo)
    changed_files = [line for line in changed["stdout"].splitlines() if line]
    if any(Path(path).name.startswith("test") or "tests" in Path(path).parts for path in changed_files):
        raise PipelineError("swap patch changes repository tests")

    swapped_env = env.copy()
    swapped_env["PYTHONPATH"] = str(swapped_repo)
    after_swap = run(test_command, swapped_repo, env=swapped_env, timeout=1200)
    if after_swap["exit_code"] != 0:
        raise PipelineError(f"repository tests fail after swap for {instance_id}")

    semantic_after = run([str(venv / "bin" / "python"), str(semantic_test_path)], swapped_repo, env=swapped_env)
    if semantic_after["exit_code"] == 0:
        raise PipelineError(f"semantic test does not detect swap for {instance_id}")

    evidence = {
        "original_suite_on_original": baseline,
        "semantic_test_source": builder["semantic_test"],
        "semantic_test_on_original": semantic_original,
        "original_suite_after_swap": after_swap,
        "semantic_test_after_swap": semantic_after,
        "changed_files": changed_files,
        "behavior_difference": {
            "original": semantic_original["stdout"],
            "swapped": semantic_after["stdout"] + semantic_after["stderr"],
        },
    }
    trajectory.append({"time": utc_now(), "actor": "pipeline", "action": "verify_swap", "result": evidence})

    generated_gold, checker_usage = call_deepseek(api_key, checker_prompt(record, builder, evidence), timeout)
    trajectory.append({
        "time": utc_now(),
        "actor": "deepseek_checker",
        "action": "label_swap",
        "output": generated_gold,
    })
    gold = prepare_generated_gold(generated_gold)
    sanitize_verification_commands(evidence)

    sample_number = next_sample_number(output_root)
    output_dir = output_root / str(sample_number)
    case = {
        "sample_number": sample_number,
        "case_id": instance_id,
        "status": "validated",
        "source": {
            "dataset": "AweAI-Team/DeNovoSWE",
            "instance_id": instance_id,
            "repository_url": record["github_url"],
            "repository_path": f"data/{sample_number}/repository",
            "parent_commit": record["parent_commit"],
            "image_url": record["image_url"],
            "license_spdx_id": record["license_spdx_id"],
        },
        "test_environment": {
            "source_image": record["image_url"],
            "local_python": run([str(venv / "bin" / "python"), "--version"], original_repo)["stdout"].strip(),
            "test_command": shlex.join(test_command),
        },
        "mapping": {
            "document_text": builder["original_document_text"],
            "target_file": builder["target_file"],
            "semantic_test": "verification_evidence.json#semantic_test_source",
        },
        "provenance": {
            "construction": "controlled_injection",
            "model": MODEL,
            "thinking_enabled": True,
            "builder_usage": builder_usage,
            "checker_usage": checker_usage,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    shutil.move(str(original_repo), str(output_dir / "repository"))
    (output_dir / "original_document.md").write_text(record["document"], encoding="utf-8")
    (output_dir / "swap.patch").write_text(swap_patch, encoding="utf-8")
    with (output_dir / "construction_trajectory.jsonl").open("w", encoding="utf-8") as handle:
        for event in trajectory:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    write_json(output_dir / "verification_evidence.json", evidence)
    write_json(output_dir / "gold.json", gold)
    write_json(case_path, case)
    print(f"[done] {instance_id}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data")
    parser.add_argument("--work", type=Path, default=ROOT / ".work" / "formal")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--timeout", type=int, default=300)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise PipelineError("DEEPSEEK_API_KEY is required")
    args.output.mkdir(parents=True, exist_ok=True)
    args.work.mkdir(parents=True, exist_ok=True)
    records = load_source_records({sample["instance_id"] for sample in SAMPLES})
    for config in SAMPLES:
        build_sample(
            records[config["instance_id"]], config, api_key,
            args.output, args.work, args.resume, args.timeout,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
