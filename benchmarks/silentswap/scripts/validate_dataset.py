#!/usr/bin/env python3
"""Validate SilentSwap gold-review metadata and generated path sanitization."""

from __future__ import annotations

import argparse
import ast
import io
import json
import re
import subprocess
import tokenize
from collections import Counter
from pathlib import Path
from typing import Any

import review_config as review
import apply_gold_reviews as adjudication
import code_locations as evaluation


ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data"
REVIEW_LOG = ROOT / "review" / "reviews.jsonl"
REVIEW_PLAN = ROOT / "review" / "review_plan.json"
GOLD_FIELDS = {
    "swap_type",
    "original_semantics",
    "swapped_semantics",
    "why_different",
    "evidence",
    "localization",
    "gold_source",
    "adjudication_status",
}
GOLD_SOURCES = {
    "model_generated",
    "model_draft_human_verified",
    "model_draft_human_revised",
    "human_rewritten",
}
ADJUDICATION_STATUSES = {
    "pending",
    "single_reviewed",
    "double_reviewed",
    "adjudicated_after_disagreement",
    "rejected",
}
SYMBOL_KINDS = {"function", "method", "class", "field", "module"}
SAMPLE_ENTRIES = {
    "case.json",
    "construction_trajectory.jsonl",
    "gold.json",
    "original_document.md",
    "repository",
    "swap.patch",
    "verification_evidence.json",
}
LOCAL_PATH = re.compile(
    r"/Users/[^/]+/(?:Desktop|Documents|Downloads)/|"
    r"[A-Za-z]:[\\\\]Users[\\\\][^\\\\]+[\\\\]"
)
INFRASTRUCTURE_FAILURE_MARKERS = (
    "ModuleNotFoundError",
    "ImportError",
    "SyntaxError",
    "IndentationError",
    "exec format error",
    "can't open file",
    "No such file or directory",
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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
    stderr = result.get("stderr", "")
    if any(marker in stderr for marker in INFRASTRUCTURE_FAILURE_MARKERS):
        raise ValueError("semantic test has an environment or code-loading error")
    stderr = stderr.strip()
    stdout = result.get("stdout", "").strip()
    output = stdout if stdout and warning_only_stderr(stderr) else stderr or stdout
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        raise ValueError("semantic test failed without diagnostic output")
    fingerprint = lines[-1]
    fingerprint = re.sub(
        r"/tmp/tmp[A-Za-z0-9._-]+(?=/|\s|$)",
        "<temporary-directory>",
        fingerprint,
    )
    fingerprint = re.sub(r"'[^'\n]*'", "'<value>'", fingerprint)
    fingerprint = re.sub(r'"[^"\n]*"', '"<value>"', fingerprint)
    return re.sub(
        r"(?<![A-Za-z_])-?\d+(?:\.\d+)?", "<number>", fingerprint
    )


def validate_no_infrastructure_error(
    sample_number: int,
    swap_number: int,
    section: str,
    result: dict[str, Any],
) -> None:
    stderr = result.get("stderr", "")
    if any(marker in stderr for marker in INFRASTRUCTURE_FAILURE_MARKERS):
        raise ValueError(
            f"sample {sample_number} swap {swap_number}: "
            f"{section} has an environment or code-loading error"
        )


def validate_sample_layout(sample_number: int) -> None:
    sample = DATA_ROOT / str(sample_number)
    entries = {path.name for path in sample.iterdir()}
    if entries != SAMPLE_ENTRIES:
        raise ValueError(f"sample {sample_number}: unexpected top-level entries")


def validate_repository(sample_number: int) -> None:
    sample = DATA_ROOT / str(sample_number)
    repository = sample / "repository"
    expected_origin = load_json(sample / "case.json")["source"]["repository_url"]
    origin = subprocess.run(
        ["git", "-C", str(repository), "config", "--get", "remote.origin.url"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if origin != expected_origin:
        raise ValueError(f"sample {sample_number}: repository origin differs")

    tracked_status = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "diff",
            "--name-only",
            "HEAD",
            "--",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    untracked_status = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "ls-files",
            "--others",
            "--exclude-standard",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if tracked_status or untracked_status:
        raise ValueError(f"sample {sample_number}: repository contains local artifacts")

    git_dir = repository / ".git"
    metadata_files = [git_dir / "config"] + [
        path for path in (git_dir / "logs").rglob("*") if path.is_file()
    ]
    for path in metadata_files:
        if LOCAL_PATH.search(path.read_text(encoding="utf-8")):
            raise ValueError(
                f"sample {sample_number}: local path remains in repository metadata"
            )


def added_python_explanations(
    original: str, swapped: str
) -> tuple[list[str], list[str]]:
    def comments(source: str) -> Counter[str]:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        return Counter(token.string for token in tokens if token.type == tokenize.COMMENT)

    def docstrings(source: str) -> Counter[str]:
        tree = ast.parse(source)
        nodes = (
            node
            for node in ast.walk(tree)
            if isinstance(
                node,
                (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
            )
        )
        return Counter(
            value
            for node in nodes
            if (value := ast.get_docstring(node, clean=False)) is not None
        )

    added_comments = list((comments(swapped) - comments(original)).elements())
    added_docstrings = list((docstrings(swapped) - docstrings(original)).elements())
    return added_comments, added_docstrings


def validate_patch_documentation_neutrality(sample_number: int) -> None:
    sample = DATA_ROOT / str(sample_number)
    overlay = evaluation.load_swapped_overlay(sample)
    for relative, swapped in overlay.items():
        if not relative.endswith(".py"):
            continue
        original = (sample / "repository" / relative).read_text(encoding="utf-8")
        comments, docstrings = added_python_explanations(original, swapped)
        if comments or docstrings:
            raise ValueError(
                f"sample {sample_number}: {relative} adds explanatory comments or docstrings"
            )


def validate_gold(sample_number: int, require_adjudicated: bool) -> None:
    sample = DATA_ROOT / str(sample_number)
    gold_document = load_json(sample / "gold.json")
    evidence = load_json(sample / "verification_evidence.json")
    gold_items = gold_document.get("swaps")
    if gold_document.get("swap_count") != 5 or not isinstance(gold_items, list) or len(gold_items) != 5:
        raise ValueError(f"sample {sample_number}: expected five Gold swaps")
    evidence_items = evidence.get("swaps")
    if not isinstance(evidence_items, list) or len(evidence_items) != 5:
        raise ValueError(f"sample {sample_number}: expected five evidence swaps")
    for index, (gold, swap_evidence) in enumerate(
        zip(gold_items, evidence_items, strict=True), 1
    ):
        validate_gold_item(
            sample_number,
            gold,
            [swap_evidence["changed_files"][0]],
            require_adjudicated,
            f" swap {index}",
        )
    validate_localization_targets(sample, gold_items)
    expected_changed = sorted(item["localization"]["file"] for item in gold_items)
    if sorted(evidence["changed_files"]) != sorted(set(expected_changed)):
        raise ValueError(f"sample {sample_number}: aggregate changed files differ")
    occupied_lines: dict[str, set[int]] = {}
    for index, item in enumerate(gold_items, 1):
        localization = item["localization"]
        lines = {
            line
            for line_range in localization["line_ranges"]
            for line in range(line_range["start"], line_range["end"] + 1)
        }
        overlap = occupied_lines.setdefault(localization["file"], set()) & lines
        if overlap:
            raise ValueError(
                f"sample {sample_number} swap {index}: localization overlaps another swap"
            )
        occupied_lines[localization["file"]].update(lines)


def validate_localization_targets(
    sample: Path, gold_items: list[dict[str, Any]]
) -> None:
    sample_number = int(sample.name)
    overlay = evaluation.load_swapped_overlay(sample)
    trees: dict[str, tuple[ast.AST, dict[ast.AST, ast.AST]]] = {}
    for swap_number, gold in enumerate(gold_items, 1):
        localization = gold["localization"]
        target_file = localization["file"]
        if target_file not in overlay:
            raise ValueError(
                f"sample {sample_number} swap {swap_number}: "
                "localization file is absent from the swapped overlay"
            )
        swapped = overlay[target_file]
        line_numbers = {
            line
            for line_range in localization["line_ranges"]
            for line in range(line_range["start"], line_range["end"] + 1)
        }
        if max(line_numbers) > len(swapped.splitlines()):
            raise ValueError(
                f"sample {sample_number} swap {swap_number}: "
                "localization line exceeds the swapped file"
            )
        if not target_file.endswith(".py"):
            if localization["symbol"] != {"kind": "module", "qualified_name": []}:
                raise ValueError(
                    f"sample {sample_number} swap {swap_number}: "
                    "non-Python localization must use the module symbol"
                )
            continue

        if target_file not in trees:
            tree = ast.parse(swapped, filename=target_file)
            trees[target_file] = (tree, evaluation._ast_parents(tree))
        tree, parents = trees[target_file]
        symbols = [
            evaluation.symbol_for_line(tree, parents, line)[0]
            for line in sorted(line_numbers)
        ]
        expected = (
            symbols[0]
            if all(symbol == symbols[0] for symbol in symbols[1:])
            else evaluation.common_symbol_for_lines(tree, parents, line_numbers)
        )
        if localization["symbol"] != expected:
            raise ValueError(
                f"sample {sample_number} swap {swap_number}: "
                f"localization symbol differs from its lines: expected {expected}"
            )


def validate_gold_item(
    sample_number: int,
    gold: dict[str, Any],
    changed_files: list[str],
    require_adjudicated: bool,
    label: str,
) -> None:
    if set(gold) != GOLD_FIELDS:
        raise ValueError(f"sample {sample_number}{label}: unexpected gold fields")
    if gold["swap_type"] not in review.SAMPLES_BY_SWAP_TYPE:
        raise ValueError(f"sample {sample_number}{label}: unsupported swap_type")
    if gold["gold_source"] not in GOLD_SOURCES:
        raise ValueError(f"sample {sample_number}{label}: unsupported gold_source")
    if gold["adjudication_status"] not in ADJUDICATION_STATUSES:
        raise ValueError(f"sample {sample_number}{label}: unsupported adjudication_status")
    localization = gold["localization"]
    if set(localization) != {"file", "symbol", "line_ranges"}:
        raise ValueError(f"sample {sample_number}{label}: invalid localization fields")
    if changed_files != [localization["file"]]:
        raise ValueError(f"sample {sample_number}{label}: localization file differs")
    symbol = localization["symbol"]
    if (
        set(symbol) != {"kind", "qualified_name"}
        or symbol["kind"] not in SYMBOL_KINDS
        or not isinstance(symbol["qualified_name"], list)
        or any(
            not isinstance(name, str) or not name for name in symbol["qualified_name"]
        )
        or (symbol["kind"] != "module" and not symbol["qualified_name"])
    ):
        raise ValueError(f"sample {sample_number}{label}: invalid localization symbol")
    line_ranges = localization["line_ranges"]
    if not isinstance(line_ranges, list) or not line_ranges:
        raise ValueError(f"sample {sample_number}{label}: missing localization lines")
    for line_range in line_ranges:
        if not isinstance(line_range, dict):
            raise ValueError(f"sample {sample_number}{label}: invalid localization range")
        start = line_range.get("start")
        end = line_range.get("end")
        if (
            set(line_range) != {"start", "end"}
            or isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or start < 1
            or end < start
        ):
            raise ValueError(f"sample {sample_number}{label}: invalid localization range")
    if require_adjudicated and gold["adjudication_status"] in {"pending", "rejected"}:
        raise ValueError(f"sample {sample_number}{label}: gold is not release-ready")
    if require_adjudicated and gold["gold_source"] == "model_generated":
        raise ValueError(f"sample {sample_number}{label}: gold still has a model-only source")
    if (
        require_adjudicated
        and sample_number in review.DOUBLE_REVIEW_SAMPLES
        and gold["adjudication_status"]
        not in {"double_reviewed", "adjudicated_after_disagreement"}
    ):
        raise ValueError(f"sample {sample_number}{label}: double review is incomplete")
    if (
        require_adjudicated
        and sample_number not in review.DOUBLE_REVIEW_SAMPLES
        and gold["adjudication_status"] != "single_reviewed"
    ):
        raise ValueError(f"sample {sample_number}{label}: unexpected single-review status")


def validate_commands(sample_number: int) -> None:
    sample = DATA_ROOT / str(sample_number)
    evidence = load_json(sample / "verification_evidence.json")
    validate_multi_swap_commands(sample_number, sample, evidence)


def validate_multi_swap_commands(
    sample_number: int, sample: Path, evidence: dict[str, Any]
) -> None:
    if evidence.get("swap_count") != 5 or len(evidence.get("swaps", [])) != 5:
        raise ValueError(f"sample {sample_number}: expected five verified swaps")
    if evidence["original_suite_on_original"]["exit_code"] != 0:
        raise ValueError(f"sample {sample_number}: aggregate original suite failed")
    if evidence["original_suite_after_all_swaps"]["exit_code"] != 0:
        raise ValueError(f"sample {sample_number}: aggregate final suite failed")
    if evidence["docker_image_lifecycle"]["remove_after_validation"]["exit_code"] != 0:
        raise ValueError(f"sample {sample_number}: Docker image cleanup failed")

    case = load_json(sample / "case.json")
    mappings = case.get("mapping", {}).get("swaps", [])
    if len(mappings) != 5:
        raise ValueError(f"sample {sample_number}: expected five document mappings")
    document_targets = [mapping.get("document_text") for mapping in mappings]
    if any(not isinstance(target, str) or not target for target in document_targets):
        raise ValueError(f"sample {sample_number}: missing document target")
    if len(set(document_targets)) != 5:
        raise ValueError(f"sample {sample_number}: duplicate document targets")
    original_document = (sample / "original_document.md").read_text(encoding="utf-8")
    spans = []
    for index, (mapping, target, swap) in enumerate(
        zip(mappings, document_targets, evidence["swaps"], strict=True), 1
    ):
        if mapping.get("swap_number") != index or swap.get("swap_number") != index:
            raise ValueError(f"sample {sample_number}: invalid swap numbering")
        if original_document.count(target) != 1:
            raise ValueError(
                f"sample {sample_number} swap {index}: document target is not unique"
            )
        start = original_document.index(target)
        spans.append((start, start + len(target)))
        if swap.get("changed_files") != [mapping.get("target_file")]:
            raise ValueError(
                f"sample {sample_number} swap {index}: mapped target file differs"
            )
    for previous, current in zip(sorted(spans), sorted(spans)[1:]):
        if previous[1] > current[0]:
            raise ValueError(f"sample {sample_number}: document targets overlap")

    semantic_sources = [swap.get("semantic_test_source") for swap in evidence["swaps"]]
    if any(not isinstance(source, str) or not source.strip() for source in semantic_sources):
        raise ValueError(f"sample {sample_number}: missing semantic test source")
    if len(set(semantic_sources)) != 5:
        raise ValueError(f"sample {sample_number}: duplicate semantic tests")

    for swap in evidence["swaps"]:
        swap_number = swap["swap_number"]
        counterfactual = swap.get("counterfactual_attribution", {})
        exit_codes = (
            swap["semantic_test_on_original"]["exit_code"],
            counterfactual["only_this_swap"]["exit_code"],
            counterfactual["other_four_without_this_swap"]["exit_code"],
            counterfactual["all_five_swaps"]["exit_code"],
        )
        if exit_codes[0] != 0 or exit_codes[1] == 0 or exit_codes[2] != 0 or exit_codes[3] == 0:
            raise ValueError(
                f"sample {sample_number} swap {swap_number}: invalid counterfactual matrix"
            )
        if swap["original_suite_on_original"]["exit_code"] != 0:
            raise ValueError(f"sample {sample_number}: original suite failed")
        if swap["original_suite_after_swap"]["exit_code"] != 0:
            raise ValueError(f"sample {sample_number}: final suite failed")
        if swap["semantic_test_after_swap"]["exit_code"] == 0:
            raise ValueError(
                f"sample {sample_number} swap {swap_number}: final semantic test passed"
            )
        if swap.get("quality_checker", {}).get("verdict") != "accept":
            raise ValueError(
                f"sample {sample_number} swap {swap_number}: quality checker did not accept"
            )

        individual = swap.get("individual_validation", {})
        if swap.get("semantic_test_source") != individual.get("semantic_test_source"):
            raise ValueError(
                f"sample {sample_number} swap {swap_number}: "
                "combined and individual semantic test sources differ"
            )
        individual_codes = (
            individual.get("original_suite_on_original", {}).get("exit_code"),
            individual.get("semantic_test_on_original", {}).get("exit_code"),
            individual.get("original_suite_after_swap", {}).get("exit_code"),
            individual.get("semantic_test_after_swap", {}).get("exit_code"),
        )
        if not (
            individual_codes[0] == 0
            and individual_codes[1] == 0
            and individual_codes[2] == 0
            and individual_codes[3] not in (None, 0)
        ):
            raise ValueError(
                f"sample {sample_number} swap {swap_number}: invalid individual validation"
            )
        for section in ("semantic_test_on_original", "semantic_test_after_swap"):
            validate_no_infrastructure_error(
                sample_number,
                swap_number,
                f"individual_validation.{section}",
                individual[section],
            )
        individual_expected = individual.get("expected_failure_fingerprint")
        if individual_expected is not None:
            individual_fingerprint = failure_fingerprint(
                individual["semantic_test_after_swap"]
            )
            expected_individual_fingerprint = failure_fingerprint(
                {"stdout": individual_expected, "stderr": ""}
            )
            if individual_fingerprint != expected_individual_fingerprint:
                raise ValueError(
                    f"sample {sample_number} swap {swap_number}: "
                    "individual failure fingerprint differs"
                )

        results = {
            "semantic_test_on_original": swap["semantic_test_on_original"],
            "only_this_swap": counterfactual["only_this_swap"],
            "other_four_without_this_swap": counterfactual[
                "other_four_without_this_swap"
            ],
            "all_five_swaps": counterfactual["all_five_swaps"],
        }
        for section, result in results.items():
            validate_no_infrastructure_error(
                sample_number, swap_number, section, result
            )

        expected = counterfactual.get("expected_failure_fingerprint")
        if not isinstance(expected, str) or not expected.strip():
            raise ValueError(
                f"sample {sample_number} swap {swap_number}: missing failure fingerprint"
            )
        expected_fingerprint = failure_fingerprint(
            {"stdout": expected, "stderr": ""}
        )
        for section in ("only_this_swap", "all_five_swaps"):
            if failure_fingerprint(counterfactual[section]) != expected_fingerprint:
                raise ValueError(
                    f"sample {sample_number} swap {swap_number}: "
                    f"{section} failure fingerprint differs"
                )
        if failure_fingerprint(swap["semantic_test_after_swap"]) != expected_fingerprint:
            raise ValueError(
                f"sample {sample_number} swap {swap_number}: final failure fingerprint differs"
            )

        final_result = swap["semantic_test_after_swap"]
        all_five_result = counterfactual["all_five_swaps"]
        if any(
            final_result[field] != all_five_result[field]
            for field in ("exit_code", "stdout", "stderr")
        ):
            raise ValueError(
                f"sample {sample_number} swap {swap_number}: final and all-five results differ"
            )

    def validate_paths(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "command" and isinstance(item, str):
                    if LOCAL_PATH.search(item):
                        raise ValueError(
                            f"sample {sample_number}: local path remains in command"
                        )
                    if "--volume" in item and review.PORTABLE_ROOT not in item:
                        raise ValueError(
                            f"sample {sample_number}: portable root missing from command"
                        )
                else:
                    validate_paths(item)
        elif isinstance(value, list):
            for item in value:
                validate_paths(item)

    validate_paths(evidence)
    verify_events = [
        json.loads(line)
        for line in (sample / "construction_trajectory.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if json.loads(line).get("action") == "verify_all_five_swaps"
    ]
    if len(verify_events) != 1 or verify_events[0]["result"] != evidence:
        raise ValueError(f"sample {sample_number}: combined verification event differs")


def validate_review_log(require_adjudicated: bool = False) -> None:
    records = [
        json.loads(line) for line in REVIEW_LOG.read_text(encoding="utf-8").splitlines()
    ]
    primary = [
        (record["sample_number"], record["swap_number"])
        for record in records
        if record["reviewer_role"] == "primary"
    ]
    secondary = [
        (record["sample_number"], record["swap_number"])
        for record in records
        if record["reviewer_role"] == "secondary"
    ]
    expected_primary = [
        (sample_number, swap_number)
        for sample_number in range(1, 101)
        for swap_number in range(1, 6)
    ]
    expected_secondary = [
        (sample_number, swap_number)
        for sample_number in review.DOUBLE_REVIEW_SAMPLES
        for swap_number in range(1, 6)
    ]
    if primary != expected_primary:
        raise ValueError(
            "review log must contain one ordered primary record per swap"
        )
    if secondary != expected_secondary:
        raise ValueError(
            "review log secondary records do not match the double-review swaps"
        )

    by_key = {
        (record["sample_number"], record["swap_number"], record["reviewer_role"]): record
        for record in records
    }
    if len(by_key) != len(records):
        raise ValueError("review log contains duplicate assignments")
    if any(
        not isinstance(record.get("reviewer_id"), str) or not record["reviewer_id"]
        for record in records
    ):
        raise ValueError("review log contains an empty reviewer identity")
    for sample_number in range(1, 101):
        gold_items = load_json(DATA_ROOT / str(sample_number) / "gold.json")["swaps"]
        for swap_number, gold in enumerate(gold_items, 1):
            roles = (
                ("primary", "secondary")
                if sample_number in review.DOUBLE_REVIEW_SAMPLES
                else ("primary",)
            )
            reviewers = [
                by_key[(sample_number, swap_number, role)]["reviewer_id"]
                for role in roles
            ]
            if len(reviewers) == 2 and reviewers[0] == reviewers[1]:
                raise ValueError(
                    f"sample {sample_number} swap {swap_number}: double review is not independent"
                )
            primary = by_key[(sample_number, swap_number, "primary")]
            if (
                gold["adjudication_status"] == "pending"
                and not require_adjudicated
                and primary.get("review_status") != "completed"
            ):
                continue
            final, expected_source, expected_status = adjudication.resolve_swap(
                sample_number, swap_number, by_key
            )
            if gold["adjudication_status"] == "pending" and not require_adjudicated:
                continue
            if (
                gold["gold_source"] != expected_source
                or gold["adjudication_status"] != expected_status
            ):
                raise ValueError(
                    f"sample {sample_number} swap {swap_number}: review metadata differs from log"
                )
            if expected_status != "rejected" and adjudication.content(final) != adjudication.gold_content(gold):
                raise ValueError(
                    f"sample {sample_number} swap {swap_number}: review content differs from Gold"
                )

    plan = load_json(REVIEW_PLAN)
    if plan["double_review_sample_numbers"] != list(review.DOUBLE_REVIEW_SAMPLES):
        raise ValueError("review plan does not match the double-review sample")
    quotas = Counter(
        review.SWAP_TYPE_BY_SAMPLE[sample_number]
        for sample_number in review.DOUBLE_REVIEW_SAMPLES
    )
    if plan["stratum_quotas"] != dict(quotas):
        raise ValueError("review plan quotas do not match the taxonomy mapping")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-adjudicated", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sample_numbers = sorted(
        int(path.name)
        for path in DATA_ROOT.iterdir()
        if path.is_dir() and path.name.isdigit()
    )
    if sample_numbers != list(range(1, 101)):
        raise ValueError("data must contain sample directories 1 through 100")
    for sample_number in sample_numbers:
        validate_sample_layout(sample_number)
        validate_repository(sample_number)
        validate_patch_documentation_neutrality(sample_number)
        validate_gold(sample_number, args.require_adjudicated)
        validate_commands(sample_number)
    validate_review_log(args.require_adjudicated)
    print("Validated 100 samples, 625 primary/secondary reviews, and 8 adjudications.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
