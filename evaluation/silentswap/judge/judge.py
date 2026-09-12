"""Apply the frozen formal SilentSwap Judge rubric to a BEG prediction."""

from __future__ import annotations

import itertools
import json
import os
from pathlib import Path
from typing import Any, Mapping, Protocol


PROMPT_PATH = (
    Path(__file__).resolve().parents[3]
    / "prompts"
    / "judge"
    / "silentswap.txt"
)
SWAPS_PER_SAMPLE = 5
SYMBOL_KINDS = ("function", "method", "class", "field", "module")
GOLD_REFERENCE_FIELDS = (
    "swap_type",
    "original_semantics",
    "swapped_semantics",
    "evidence",
)
JUDGE_SCORE_NAMES = ("location_correct", "code_change_correct")
JUDGE_SCORE_VALUES = (0, 0.5, 1)
LOCATION_COMPONENTS = (
    "file_correct",
    "symbol_kind_correct",
    "qualified_name_correct",
    "line_range_acceptable",
)
CODE_CHANGE_COMPONENTS = (
    "concrete_operation_correct",
    "all_changed_operations_covered",
    "original_logic_correct",
    "swapped_logic_correct",
    "direction_correct",
    "no_material_contradiction",
)
JUDGE_SYSTEM_PROMPT = (
    "You are an independent benchmark judge. Compare behavioral meaning, apply the supplied rubric "
    "strictly, and return one JSON object. Use a one-to-one candidate alignment: each non-null "
    "matched_candidate_swap_number may appear at most once in location_checks, and "
    "code_change_checks must use the same alignment. If no unique unused candidate matches a "
    "reference swap, set matched_candidate_swap_number to null instead of reusing a number. "
    "Before returning, verify that all non-null matched candidate numbers are unique."
)


class JsonJudgeClient(Protocol):
    def complete(self, messages: list[dict[str, str]]) -> Mapping[str, Any]: ...


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _candidate_target(swap: Mapping[str, Any]) -> Mapping[str, Any]:
    """Use the formal target while retaining old BEG prediction compatibility."""

    target = swap.get("target")
    if isinstance(target, Mapping):
        return target
    localization = swap.get("localization")
    return localization if isinstance(localization, Mapping) else {}


def _data_root_candidates(formal_data_root: Path | None) -> list[Path]:
    candidates: list[Path] = []
    if formal_data_root is not None:
        candidates.append(formal_data_root)
    configured = os.environ.get("SILENTSWAP_DATA_ROOT")
    if configured:
        candidates.append(Path(configured).expanduser())
    for ancestor in Path(__file__).resolve().parents:
        candidates.extend(
            (
                ancestor / "SilentSwap" / "data",
                ancestor / "benchmark" / "silentswap" / "data",
            )
        )
    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def _official_sample_dir(
    gold: Mapping[str, Any], formal_data_root: Path | None
) -> Path:
    sample_id = str(gold.get("source_sample_id") or "")
    if not sample_id.isdigit():
        raise ValueError("SilentSwap Gold must contain a numeric source_sample_id")
    roots = _data_root_candidates(formal_data_root)
    for root in roots:
        sample = root / sample_id
        if all(
            (sample / name).is_file()
            for name in ("gold.json", "case.json", "verification_evidence.json")
        ):
            return sample
    raise FileNotFoundError(
        "cannot locate the formal SilentSwap reference for sample "
        f"{sample_id}; set SILENTSWAP_DATA_ROOT (searched: "
        + ", ".join(str(path) for path in roots)
        + ")"
    )


def _semantic_gold_reference(gold: Mapping[str, Any]) -> list[dict[str, Any]]:
    swaps = gold.get("swaps")
    if not isinstance(swaps, list) or len(swaps) != SWAPS_PER_SAMPLE:
        raise ValueError("SilentSwap Gold must contain exactly five swaps")
    return [
        {field: swap[field] for field in GOLD_REFERENCE_FIELDS}
        for swap in swaps
    ]


def build_judge_reference(
    gold: Mapping[str, Any], *, formal_data_root: Path | None = None
) -> dict[str, Any]:
    """Recreate ``evaluate_models.judge_reference`` without a lossy summary."""

    embedded = gold.get("official_judge_reference")
    if isinstance(embedded, Mapping):
        return dict(embedded)

    sample = _official_sample_dir(gold, formal_data_root)
    source_gold = _read_json(sample / "gold.json")
    case = _read_json(sample / "case.json")
    execution = _read_json(sample / "verification_evidence.json")

    compact_swaps = gold.get("swaps")
    source_swaps = source_gold.get("swaps")
    if not isinstance(compact_swaps, list) or not isinstance(source_swaps, list):
        raise ValueError("SilentSwap Gold swaps must be arrays")
    if len(compact_swaps) != SWAPS_PER_SAMPLE or len(source_swaps) != SWAPS_PER_SAMPLE:
        raise ValueError("SilentSwap Gold must contain exactly five swaps")
    for compact, source in zip(compact_swaps, source_swaps):
        for field in (*GOLD_REFERENCE_FIELDS, "localization"):
            if compact.get(field) != source.get(field):
                raise ValueError(
                    f"compact Gold differs from formal source field {field!r}"
                )

    gold_swaps = _semantic_gold_reference(source_gold)
    mappings = case["mapping"]["swaps"]
    execution_swaps = execution["swaps"]
    if not (
        len(gold_swaps)
        == len(mappings)
        == len(execution_swaps)
        == SWAPS_PER_SAMPLE
    ):
        raise ValueError("formal SilentSwap reference must contain exactly five swaps")
    return {
        "swaps": [
            {
                "swap_number": index,
                "gold": gold_swaps[index - 1],
                "localization": source_gold["swaps"][index - 1]["localization"],
                "document_target": mappings[index - 1]["document_text"],
                "executed_behavior_difference": {
                    "on_original": execution_swaps[index - 1][
                        "semantic_test_on_original"
                    ],
                    "after_all_five_swaps": execution_swaps[index - 1][
                        "semantic_test_after_swap"
                    ],
                },
            }
            for index in range(1, SWAPS_PER_SAMPLE + 1)
        ],
        "changed_files": execution["changed_files"],
    }


def _visible_source_total_lines(gold: Mapping[str, Any], relative: str) -> int:
    explicit = gold.get("source_line_counts")
    if isinstance(explicit, Mapping):
        total = explicit.get(relative)
        if isinstance(total, int) and not isinstance(total, bool) and total > 0:
            return total
    input_id = gold.get("input_id")
    candidates = [
        Path(__file__).resolve().parents[1]
        / "artifacts"
        / "visible_bundles"
        / str(input_id)
        / "repository"
        / relative
    ]
    candidates.extend(
        ancestor
        / "evaluation"
        / "silentswap"
        / "artifacts"
        / "visible_bundles"
        / str(input_id)
        / "repository"
        / relative
        for ancestor in Path(__file__).resolve().parents
    )
    for source in candidates:
        if source.is_file():
            return len(source.read_text(encoding="utf-8").splitlines())
    raise FileNotFoundError(
        "cannot count lines in swapped source; searched: "
        + ", ".join(str(path) for path in candidates)
    )


def _gold_localizations(gold: Mapping[str, Any]) -> list[dict[str, Any]]:
    swaps = gold.get("swaps")
    if not isinstance(swaps, list) or len(swaps) != SWAPS_PER_SAMPLE:
        raise ValueError("SilentSwap Gold must contain exactly five swaps")
    localizations = []
    for swap in swaps:
        localization = swap["localization"]
        line_numbers = {
            line
            for line_range in localization["line_ranges"]
            for line in range(line_range["start"], line_range["end"] + 1)
        }
        localizations.append(
            {
                **localization,
                "line_numbers": line_numbers,
                "total_lines": _visible_source_total_lines(gold, localization["file"]),
            }
        )
    return localizations


def _predicted_line_numbers(value: Any, *, total_lines: int) -> set[int]:
    if not isinstance(value, list) or not value:
        return set()
    lines: set[int] = set()
    for line_range in value:
        if not isinstance(line_range, Mapping):
            return set()
        start = line_range.get("start")
        end = line_range.get("end")
        if isinstance(start, str) and start.isdigit():
            start = int(start)
        if isinstance(end, str) and end.isdigit():
            end = int(end)
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or start < 1
            or end < start
            or end > total_lines
        ):
            return set()
        lines.update(range(start, end + 1))
    return lines


def score_localization_pair(
    target: Mapping[str, Any], gold: Mapping[str, Any]
) -> dict[str, Any]:
    predicted_symbol = target.get("symbol")
    symbol_valid = (
        isinstance(predicted_symbol, Mapping)
        and predicted_symbol.get("kind") in SYMBOL_KINDS
        and isinstance(predicted_symbol.get("qualified_name"), list)
        and all(
            isinstance(name, str)
            for name in predicted_symbol.get("qualified_name", [])
        )
    )
    file_exact = target.get("file") == gold["file"]
    symbol_exact = symbol_valid and predicted_symbol == gold["symbol"]
    gold_lines = gold["line_numbers"]
    if file_exact:
        predicted_lines = _predicted_line_numbers(
            target.get("line_ranges"), total_lines=gold["total_lines"]
        )
        if predicted_lines:
            overlap = len(predicted_lines & gold_lines)
            line_precision = overlap / len(predicted_lines)
            line_recall = overlap / len(gold_lines)
            line_f1 = (
                2 * line_precision * line_recall / (line_precision + line_recall)
                if line_precision + line_recall
                else 0.0
            )
            minimum_distance = min(
                abs(predicted - expected)
                for predicted in predicted_lines
                for expected in gold_lines
            )
        else:
            line_precision = 0.0
            line_recall = 0.0
            line_f1 = 0.0
            minimum_distance = None
    else:
        line_precision = 0.0
        line_recall = 0.0
        line_f1 = 0.0
        minimum_distance = None
    return {
        "file_exact": file_exact,
        "symbol_exact": symbol_exact,
        "line_precision": line_precision,
        "line_recall": line_recall,
        "line_f1": line_f1,
        "line_hit_at_0": minimum_distance == 0,
        "line_hit_at_3": minimum_distance is not None and minimum_distance <= 3,
        "minimum_line_distance": minimum_distance,
        "localization_score": float(file_exact and symbol_exact) * line_f1,
    }


def rule_based_score(
    gold: Mapping[str, Any], prediction: Mapping[str, Any]
) -> dict[str, Any]:
    references = _gold_localizations(gold)
    predicted_swaps = prediction.get("swaps")
    if not isinstance(predicted_swaps, list) or len(predicted_swaps) > SWAPS_PER_SAMPLE:
        raise ValueError("answer must contain a list of at most five swaps")
    pair_scores = [
        [
            score_localization_pair(
                _candidate_target(swap) if isinstance(swap, Mapping) else {},
                target,
            )
            for target in references
        ]
        for swap in predicted_swaps
    ]
    assignment = max(
        itertools.permutations(range(SWAPS_PER_SAMPLE), len(predicted_swaps)),
        key=lambda candidate: (
            sum(
                pair_scores[index][gold_index]["localization_score"]
                for index, gold_index in enumerate(candidate)
            ),
            sum(
                pair_scores[index][gold_index]["file_exact"]
                for index, gold_index in enumerate(candidate)
            ),
            sum(
                pair_scores[index][gold_index]["symbol_exact"]
                for index, gold_index in enumerate(candidate)
            ),
            tuple(-gold_index for gold_index in candidate),
        ),
    )
    matches = []
    for predicted_index, gold_index in enumerate(assignment):
        metrics = pair_scores[predicted_index][gold_index]
        matches.append(
            {
                "predicted_swap_number": predicted_index + 1,
                "gold_swap_number": gold_index + 1,
                **metrics,
                "gold_target": {
                    key: value
                    for key, value in references[gold_index].items()
                    if key not in {"line_numbers", "total_lines"}
                },
            }
        )

    def mean(field: str) -> float:
        # Unreported gold swaps contribute zero, including an empty answer.
        return sum(float(match[field]) for match in matches) / SWAPS_PER_SAMPLE

    complete = len(matches) == SWAPS_PER_SAMPLE
    return {
        "gold_swaps": SWAPS_PER_SAMPLE,
        "predicted_swaps": len(predicted_swaps),
        "file_exact": complete and all(match["file_exact"] for match in matches),
        "symbol_exact": complete and all(match["symbol_exact"] for match in matches),
        "line_precision": mean("line_precision"),
        "line_recall": mean("line_recall"),
        "line_f1": mean("line_f1"),
        "line_hit_at_0": complete and all(match["line_hit_at_0"] for match in matches),
        "line_hit_at_3": complete and all(match["line_hit_at_3"] for match in matches),
        "localization_score": mean("localization_score"),
        "matched_swaps": matches,
    }


def _judge_contract() -> dict[str, Any]:
    return {
        "location_checks": [
            {
                "reference_swap_number": number,
                "matched_candidate_swap_number": number,
                "file_correct": False,
                "symbol_kind_correct": False,
                "qualified_name_correct": False,
                "line_range_acceptable": False,
                "note": "candidate target and exact missing or incorrect component",
            }
            for number in range(1, SWAPS_PER_SAMPLE + 1)
        ],
        "code_change_checks": [
            {
                "reference_swap_number": number,
                "matched_candidate_swap_number": number,
                "concrete_operation_correct": False,
                "all_changed_operations_covered": False,
                "original_logic_correct": False,
                "swapped_logic_correct": False,
                "direction_correct": False,
                "no_material_contradiction": False,
                "note": "candidate quote and exact missing or incorrect component",
            }
            for number in range(1, SWAPS_PER_SAMPLE + 1)
        ],
        "score_notes": {
            name: "candidate quote and exact missing fact, if any"
            for name in JUDGE_SCORE_NAMES
        },
        "missing_or_incorrect": ["each omitted, vague, or wrong required fact"],
        "contradicted_claims": [
            "candidate claims contradicted by the reference evidence; diagnostic only"
        ],
        "rationale": "strict assessment without filling gaps in the candidate answer",
    }


def build_llm_messages(
    gold: Mapping[str, Any],
    prediction: Mapping[str, Any],
    *,
    formal_data_root: Path | None = None,
) -> list[dict[str, str]]:
    reference = build_judge_reference(gold, formal_data_root=formal_data_root)
    answer = {"swaps": prediction["swaps"]}
    user_prompt = (
        PROMPT_PATH.read_text(encoding="utf-8")
        .strip()
        .replace(
            "{{JSON_CONTRACT}}",
            json.dumps(_judge_contract(), ensure_ascii=False, indent=2),
        )
        .replace(
            "{{REFERENCE}}",
            json.dumps(reference, ensure_ascii=False, indent=2),
        )
        .replace(
            "{{CANDIDATE}}", json.dumps(answer, ensure_ascii=False, indent=2)
        )
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _evaluate_checks(
    checks: Any,
    components: tuple[str, ...],
    dimension: str,
    partial_minimum: int,
    candidate_count: int,
) -> tuple[list[dict[str, Any]], int | float]:
    if not isinstance(checks, list) or len(checks) != SWAPS_PER_SAMPLE:
        raise ValueError(f"judge must return exactly five {dimension} checks")
    if not all(isinstance(check, Mapping) for check in checks):
        raise ValueError(f"{dimension} checks must be objects")
    ordered = sorted(checks, key=lambda check: check["reference_swap_number"])
    if [check["reference_swap_number"] for check in ordered] != list(range(1, 6)):
        raise ValueError(f"{dimension} checks must cover reference swaps 1 through 5")
    matched_candidates = [
        check["matched_candidate_swap_number"]
        for check in ordered
        if check["matched_candidate_swap_number"] is not None
    ]
    if any(
        isinstance(number, bool)
        or not isinstance(number, int)
        or number not in range(1, candidate_count + 1)
        for number in matched_candidates
    ) or len(matched_candidates) != len(set(matched_candidates)):
        raise ValueError(f"{dimension} checks must use a one-to-one candidate alignment")

    normalized = []
    for check in ordered:
        component_values = {name: check[name] for name in components}
        if any(not isinstance(value, bool) for value in component_values.values()):
            raise ValueError(f"{dimension} check components must be booleans")
        if check["matched_candidate_swap_number"] is None and any(
            component_values.values()
        ):
            raise ValueError("a missing candidate swap cannot have correct components")
        normalized.append(
            {
                "reference_swap_number": check["reference_swap_number"],
                "matched_candidate_swap_number": check[
                    "matched_candidate_swap_number"
                ],
                **component_values,
                "fully_correct": all(component_values.values()),
                "note": check["note"],
            }
        )
    fully_correct_count = sum(check["fully_correct"] for check in normalized)
    score = (
        1
        if fully_correct_count == 5
        else 0.5
        if fully_correct_count >= partial_minimum
        else 0
    )
    return normalized, score


def validate_llm_response(
    value: Mapping[str, Any], *, candidate_count: int = SWAPS_PER_SAMPLE
) -> dict[str, Any]:
    location_checks, location_score = _evaluate_checks(
        value["location_checks"], LOCATION_COMPONENTS, "location", 3, candidate_count
    )
    code_checks, code_score = _evaluate_checks(
        value["code_change_checks"], CODE_CHANGE_COMPONENTS, "code change", 4,
        candidate_count,
    )
    location_alignment = {
        check["reference_swap_number"]: check["matched_candidate_swap_number"]
        for check in location_checks
    }
    code_alignment = {
        check["reference_swap_number"]: check["matched_candidate_swap_number"]
        for check in code_checks
    }
    if location_alignment != code_alignment:
        raise ValueError(
            "location and code change checks must use the same candidate alignment"
        )
    scores = {
        "location_correct": location_score,
        "code_change_correct": code_score,
    }
    if any(
        isinstance(score, bool) or score not in JUDGE_SCORE_VALUES
        for score in scores.values()
    ):
        raise ValueError("judge scores must be numeric values from: 0, 0.5, 1")
    if "scores" in value and value["scores"] != scores:
        raise ValueError("saved judge scores differ from the formal tier calculation")
    score_notes = value["score_notes"]
    location_prefix = (
        f"{sum(check['fully_correct'] for check in location_checks)}/5 swaps fully correct. "
    )
    code_prefix = (
        f"{sum(check['fully_correct'] for check in code_checks)}/5 swaps fully correct. "
    )
    location_note = score_notes["location_correct"]
    code_note = score_notes["code_change_correct"]
    return {
        "scores": scores,
        "location_checks": location_checks,
        "code_change_checks": code_checks,
        "score_notes": {
            "location_correct": (
                location_note
                if "scores" in value and location_note.startswith(location_prefix)
                else location_prefix + location_note
            ),
            "code_change_correct": (
                code_note
                if "scores" in value and code_note.startswith(code_prefix)
                else code_prefix + code_note
            ),
        },
        "missing_or_incorrect": value["missing_or_incorrect"],
        "contradicted_claims": value["contradicted_claims"],
        "rationale": value["rationale"],
    }


def judge_prediction(
    gold: Mapping[str, Any],
    prediction: Mapping[str, Any],
    client: JsonJudgeClient,
    *,
    formal_data_root: Path | None = None,
) -> dict[str, Any]:
    if gold.get("input_id") != prediction.get("input_id"):
        raise ValueError("Gold and prediction input_id values differ")
    return {
        "input_id": gold["input_id"],
        "benchmark": "silentswap",
        "rule_based": rule_based_score(gold, prediction),
        "llm_judge": validate_llm_response(
            client.complete(
                build_llm_messages(
                    gold, prediction, formal_data_root=formal_data_root
                )
            ),
            candidate_count=len(prediction["swaps"]),
        ),
    }
