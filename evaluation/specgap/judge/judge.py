"""Official-compatible SpecGap Judge for saved BEG predictions.

The transport remains outside this module.  The Judge input is reconstructed
from the frozen BEG sample identity and the original SpecGAP reference files so
the model receives the same semantic payload as the desktop benchmark.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Protocol, Sequence


SCHEMA_VERSION = "specgap-eval-1.12"
SOURCE_EVIDENCE_TYPES = {"implementation", "data_flow", "configuration"}
MAX_EVIDENCE_LINES = 160
PROMPT_PATH = Path(__file__).resolve().parents[3] / "prompts" / "judge" / "specgap.txt"
SPEC_GAP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FORMAL_DATA_ROOT = Path.home() / "Desktop" / "SpecGAP" / "SpecGAP"


@dataclass(frozen=True)
class CodeRange:
    path: str
    start_line: int
    end_line: int
    symbol: str | None = None
    evidence_id: str | None = None
    route: str | None = None

    def overlaps(self, other: "CodeRange") -> bool:
        return (
            self.path == other.path
            and self.start_line <= other.end_line
            and other.start_line <= self.end_line
        )


@dataclass(frozen=True)
class GoldCondition:
    condition_id: str
    condition_type: str
    importance: str
    normalized_condition: str
    source_text: str
    expected_verification_question: str
    why_important: str
    downstream_impact: str
    gold_source: str
    gold_explanation: str
    condition_by_condition: dict[str, Any]
    holistic_alignment: dict[str, Any]
    location_targets: tuple[tuple[CodeRange, ...], ...]


class JsonJudgeClient(Protocol):
    def complete(self, messages: list[dict[str, str]]) -> Mapping[str, Any]: ...


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def normalize_repo_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"invalid repository path: {value!r}")
    return path.as_posix()


def candidate_findings(prediction: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Normalize BEG and legacy predictions to the official finding shape."""

    native = prediction.get("findings")
    if isinstance(native, list):
        findings: list[dict[str, Any]] = []
        for index, value in enumerate(native, start=1):
            if not isinstance(value, Mapping):
                continue
            evidence = [
                {
                    "path": item.get("path"),
                    "symbol": item.get("symbol"),
                    "start_line": item.get("start_line"),
                    "end_line": item.get("end_line"),
                    "explanation": item.get("explanation"),
                }
                for item in value.get("code_evidence", [])
                if isinstance(item, Mapping)
            ]
            findings.append(
                {
                    "finding_id": str(value.get("finding_id") or f"F{index:03d}"),
                    "claim": value.get("claim"),
                    "verification_question": value.get("verification_question"),
                    "why_important": value.get("why_important"),
                    "downstream_impact": value.get("downstream_impact"),
                    "code_evidence": evidence,
                }
            )
        return findings

    legacy = prediction.get("conditions")
    if not isinstance(legacy, list):
        return []
    findings = []
    for index, condition in enumerate(legacy, start=1):
        if not isinstance(condition, Mapping):
            continue
        evidence = []
        for location in condition.get("implementation_locations", []):
            if not isinstance(location, Mapping):
                continue
            for line_range in location.get("line_ranges", []):
                if isinstance(line_range, Mapping):
                    evidence.append(
                        {
                            "path": location.get("file"),
                            "symbol": location.get("symbol"),
                            "start_line": line_range.get("start"),
                            "end_line": line_range.get("end"),
                        }
                    )
        findings.append(
            {
                "finding_id": f"F{index:03d}",
                "claim": condition.get("normalized_condition"),
                "verification_question": condition.get(
                    "expected_verification_question"
                ),
                "why_important": condition.get("why_important"),
                "downstream_impact": condition.get("downstream_impact"),
                "code_evidence": evidence,
            }
        )
    return findings


def compact_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    evidence = []
    for item in mapping.get("evidence", []):
        if item.get("evidence_type") not in SOURCE_EVIDENCE_TYPES:
            continue
        evidence.append(
            {
                "evidence_id": item.get("evidence_id"),
                "evidence_type": item.get("evidence_type"),
                "relation": item.get("relation"),
                "strength": item.get("strength"),
                "locations": item.get("locations"),
                "explanation": item.get("explanation"),
            }
        )
    return {
        "mapping_status": mapping.get("mapping_status"),
        "coverage": mapping.get("coverage"),
        "mapping_explanation": mapping.get("mapping_explanation"),
        "evidence": evidence,
    }


def gold_for_judge(condition: GoldCondition) -> dict[str, Any]:
    """Return the exact field projection used by the formal benchmark."""

    return {
        "condition_id": condition.condition_id,
        "type": condition.condition_type,
        "importance": condition.importance,
        "normalized_condition": condition.normalized_condition,
        "source_text": condition.source_text,
        "expected_verification_question": condition.expected_verification_question,
        "why_important": condition.why_important,
        "downstream_impact": condition.downstream_impact,
        "manual_gold_explanation": condition.gold_explanation,
        "manual_gold_source": condition.gold_source,
        "condition_by_condition": compact_mapping(
            condition.condition_by_condition
        ),
        "holistic_alignment": compact_mapping(condition.holistic_alignment),
    }


def direct_implementation_targets(
    mapping: Mapping[str, Any], route: str
) -> tuple[tuple[CodeRange, ...], ...]:
    targets: list[tuple[CodeRange, ...]] = []
    for evidence in mapping.get("evidence", []):
        if (
            evidence.get("evidence_type") != "implementation"
            or evidence.get("strength") != "direct"
            or evidence.get("relation") == "contradicts"
        ):
            continue
        evidence_id = str(evidence.get("evidence_id") or "") or None
        for location in evidence.get("locations", []):
            symbol_value = location.get("symbol") or {}
            symbol = str(symbol_value.get("qualified_name") or "") or None
            if symbol is None:
                continue
            target = tuple(
                CodeRange(
                    path=normalize_repo_path(str(location["file_path"])),
                    start_line=int(line_range["start"]),
                    end_line=int(line_range["end"]),
                    symbol=symbol,
                    evidence_id=evidence_id,
                    route=route,
                )
                for line_range in location.get("line_ranges", [])
            )
            if target:
                targets.append(target)
    unique: dict[
        tuple[tuple[str, int, int, str | None], ...], tuple[CodeRange, ...]
    ] = {}
    for target in targets:
        key = tuple(
            (item.path, item.start_line, item.end_line, item.symbol)
            for item in target
        )
        unique.setdefault(key, target)
    return tuple(unique.values())


def gold_location_targets(
    condition_by_condition: Mapping[str, Any],
    holistic_alignment: Mapping[str, Any],
    gold_source: str,
) -> tuple[tuple[CodeRange, ...], ...]:
    branches = {
        "condition_by_condition": condition_by_condition,
        "holistic_alignment": holistic_alignment,
    }
    routes = (
        (gold_source,)
        if gold_source in branches
        else ("condition_by_condition", "holistic_alignment")
    )
    unique: dict[
        tuple[tuple[str, int, int, str | None], ...], tuple[CodeRange, ...]
    ] = {}
    for route in routes:
        for target in direct_implementation_targets(branches[route], route):
            key = tuple(
                (item.path, item.start_line, item.end_line, item.symbol)
                for item in target
            )
            unique.setdefault(key, target)
    return tuple(unique.values())


def _formal_sample_root(gold: Mapping[str, Any], data_root: Path) -> Path:
    source_sample_id = str(gold.get("source_sample_id") or "")
    if not source_sample_id:
        raise ValueError("SpecGap hidden Gold lacks source_sample_id")
    root = data_root.resolve()
    if (root / "2_deleted_parts.json").is_file() and (
        root / "4_code_mapping.json"
    ).is_file():
        return root
    matches = sorted(
        path
        for path in root.glob(f"*_{source_sample_id}")
        if path.is_dir()
    )
    if len(matches) != 1:
        raise ValueError(
            f"expected one formal SpecGap sample for {source_sample_id!r}, "
            f"found {len(matches)} under {root}"
        )
    return matches[0]


def load_gold_conditions(
    gold: Mapping[str, Any],
    *,
    formal_data_root: Path | None = None,
) -> list[GoldCondition]:
    """Load the selected conditions from the original formal reference files."""

    sample_root = _formal_sample_root(
        gold, formal_data_root or DEFAULT_FORMAL_DATA_ROOT
    )
    deleted = _read_json(sample_root / "2_deleted_parts.json")["deleted_parts"]
    mappings = _read_json(sample_root / "4_code_mapping.json")["code_mappings"]
    deleted_by_id = {str(item["condition_id"]): item for item in deleted}
    items = {
        str(item["condition_id"]): item
        for item in mappings["comparison_items"]
    }
    labels = {
        str(item["condition_id"]): item for item in mappings["gold_labels"]
    }
    conditions: list[GoldCondition] = []
    for hidden in gold.get("conditions", []):
        condition_id = str(hidden["condition_id"])
        try:
            part = deleted_by_id[condition_id]
            item = items[condition_id]
            label = labels[condition_id]
        except KeyError as error:
            raise ValueError(
                f"formal Gold lacks selected condition {condition_id}"
            ) from error
        for field in ("normalized_condition", "source_text"):
            if hidden.get(field) != part.get(field):
                raise ValueError(
                    f"hidden and formal Gold differ for {condition_id}.{field}"
                )
        cbc = item["condition_by_condition"]
        holistic = item["holistic_alignment"]
        gold_source = str(label["gold_source"])
        conditions.append(
            GoldCondition(
                condition_id=condition_id,
                condition_type=str(part["type"]),
                importance=str(part["importance"]),
                normalized_condition=str(part["normalized_condition"]),
                source_text=str(part["source_text"]),
                expected_verification_question=str(
                    part["expected_verification_question"]
                ),
                why_important=str(part["why_important"]),
                downstream_impact=str(part["downstream_impact"]),
                gold_source=gold_source,
                gold_explanation=str(label["mapping_explanation"]),
                condition_by_condition=cbc,
                holistic_alignment=holistic,
                location_targets=gold_location_targets(cbc, holistic, gold_source),
            )
        )
    return conditions


def _default_repository_root(gold: Mapping[str, Any]) -> Path:
    input_id = str(gold.get("input_id") or "")
    if not input_id or input_id != Path(input_id).name:
        raise ValueError(f"invalid SpecGap input_id: {input_id!r}")
    return SPEC_GAP_ROOT / "artifacts" / "visible_bundles" / input_id / "repository"


def _read_repository_excerpt(
    repository_root: Path, path_value: Any, start_value: Any, end_value: Any
) -> str:
    relative = normalize_repo_path(str(path_value))
    root = repository_root.resolve()
    path = (root / relative).resolve()
    if root not in path.parents or not path.is_file():
        raise ValueError(f"path is not an indexed repository file: {relative}")
    start_line, end_line = int(start_value), int(end_value)
    if start_line < 1 or end_line < start_line:
        raise ValueError("read_file requires 1 <= start_line <= end_line")
    if end_line - start_line + 1 > MAX_EVIDENCE_LINES:
        raise ValueError(
            f"read_file is limited to {MAX_EVIDENCE_LINES} lines per action"
        )
    lines = path.read_text(encoding="utf-8").splitlines()
    if start_line > len(lines):
        raise ValueError(f"start_line exceeds file length ({len(lines)})")
    actual_end = min(end_line, len(lines))
    return "\n".join(
        f"{number:>6} | {lines[number - 1]}"
        for number in range(start_line, actual_end + 1)
    )


def resolve_candidate_evidence(
    prediction: Mapping[str, Any], repository_root: Path
) -> list[dict[str, Any]]:
    """Attach the same resolved source excerpts used by the formal Judge."""

    resolved = []
    for finding in candidate_findings(prediction):
        evidence_items = []
        for evidence in finding["code_evidence"]:
            try:
                excerpt = _read_repository_excerpt(
                    repository_root,
                    evidence["path"],
                    evidence["start_line"],
                    evidence["end_line"],
                )
                evidence_items.append(
                    {**evidence, "resolved": True, "excerpt": excerpt}
                )
            except (KeyError, TypeError, ValueError) as error:
                evidence_items.append(
                    {**evidence, "resolved": False, "error": str(error)}
                )
        resolved.append({**finding, "code_evidence": evidence_items})
    return resolved


def build_llm_messages(
    gold: Mapping[str, Any],
    prediction: Mapping[str, Any],
    *,
    document_after: str | None = None,
    repository_root: Path | None = None,
    formal_data_root: Path | None = None,
) -> list[dict[str, str]]:
    conditions = load_gold_conditions(gold, formal_data_root=formal_data_root)
    resolved_findings = resolve_candidate_evidence(
        prediction, repository_root or _default_repository_root(gold)
    )
    judge_input = {
        "document_after": document_after,
        "gold_conditions": [gold_for_judge(condition) for condition in conditions],
        "candidate_findings": resolved_findings,
    }
    return [
        {"role": "system", "content": PROMPT_PATH.read_text(encoding="utf-8").strip()},
        {"role": "user", "content": json.dumps(judge_input, ensure_ascii=False)},
    ]


def validate_llm_response(
    value: Mapping[str, Any],
    *,
    finding_ids: set[str],
    condition_ids: set[str],
) -> dict[str, Any]:
    matches = value.get("matches")
    unmatched = value.get("unmatched_findings")
    if not isinstance(matches, list) or not isinstance(unmatched, list):
        raise ValueError("judge output requires matches and unmatched_findings arrays")
    matched_findings: set[str] = set()
    matched_conditions: set[str] = set()
    for item in matches:
        finding_id = str(item.get("finding_id") or "")
        condition_id = str(item.get("condition_id") or "")
        if finding_id not in finding_ids or finding_id in matched_findings:
            raise ValueError(f"invalid or duplicate matched finding: {finding_id}")
        if condition_id not in condition_ids or condition_id in matched_conditions:
            raise ValueError(f"invalid or duplicate matched condition: {condition_id}")
        if item.get("match_score") not in {0.5, 1.0}:
            raise ValueError(f"invalid match_score for {finding_id}")
        if item.get("question_score") not in {0.0, 0.5, 1.0}:
            raise ValueError(f"invalid question_score for {finding_id}")
        matched_findings.add(finding_id)
        matched_conditions.add(condition_id)
    unmatched_ids: set[str] = set()
    for item in unmatched:
        finding_id = str(item.get("finding_id") or "")
        if (
            finding_id not in finding_ids
            or finding_id in matched_findings
            or finding_id in unmatched_ids
        ):
            raise ValueError(f"invalid unmatched finding: {finding_id}")
        unmatched_ids.add(finding_id)
    if matched_findings | unmatched_ids != finding_ids:
        raise ValueError("judge did not account for every candidate finding exactly once")
    return {"matches": matches, "unmatched_findings": unmatched}


def judgment_correction_prompt(error: ValueError) -> str:
    return f"""Your previous JSON failed validation: {error}

Return a complete corrected JSON object only. Preserve the substantive assessment, but
fix the structure so it follows every rule from the original instructions. In particular:
- matches may use only match_score 0.5 or 1.0;
- a finding assessed as match_score 0.0 belongs in unmatched_findings, not matches;
- every candidate finding must appear exactly once across the two arrays;
- matched findings and matched conditions must remain one-to-one."""


def finding_evidence(finding: Mapping[str, Any]) -> tuple[CodeRange, ...]:
    return tuple(
        CodeRange(
            path=normalize_repo_path(str(value["path"])),
            start_line=int(value["start_line"]),
            end_line=int(value["end_line"]),
            symbol=str(value.get("symbol") or "").strip() or None,
        )
        for value in finding["code_evidence"]
    )


def symbol_compatible(candidate: str | None, gold: str | None) -> bool:
    if not candidate or not gold:
        return False
    if candidate == gold:
        return True
    candidate_parts = candidate.split(".")
    gold_parts = gold.split(".")
    return (
        len(candidate_parts) >= 2
        and len(gold_parts) >= 2
        and candidate_parts[-2:] == gold_parts[-2:]
    )


def target_score(evidence: CodeRange, target: Sequence[CodeRange]) -> dict[str, Any]:
    first = target[0]
    file_exact = evidence.path == first.path
    symbol_match = file_exact and symbol_compatible(evidence.symbol, first.symbol)
    line_overlap = file_exact and any(evidence.overlaps(item) for item in target)
    if symbol_match and line_overlap:
        score, match_basis = 1.0, "symbol_and_line"
    elif symbol_match or line_overlap:
        score, match_basis = 0.5, "symbol" if symbol_match else "line"
    else:
        score, match_basis = 0.0, "none"
    return {
        "score": score,
        "file_exact": file_exact,
        "symbol_compatible": symbol_match,
        "line_overlap": line_overlap,
        "match_basis": match_basis,
    }


def pair_score(
    evidence: Sequence[CodeRange], condition: GoldCondition
) -> dict[str, Any]:
    choices = [
        (target_score(item, target), evidence_index, target_index)
        for evidence_index, item in enumerate(evidence)
        for target_index, target in enumerate(condition.location_targets)
    ]
    if not choices:
        return {"score": 0.0}
    score, evidence_index, target_index = max(
        choices,
        key=lambda choice: (
            choice[0]["score"],
            choice[0]["symbol_compatible"],
            choice[0]["line_overlap"],
            -choice[1],
            -choice[2],
        ),
    )
    item = evidence[evidence_index]
    target = condition.location_targets[target_index]
    return {
        **score,
        "evidence_location": {
            "path": item.path,
            "symbol": item.symbol,
            "start_line": item.start_line,
            "end_line": item.end_line,
        },
        "gold_target": {
            "path": target[0].path,
            "symbol": target[0].symbol,
            "line_ranges": [
                {"start_line": value.start_line, "end_line": value.end_line}
                for value in target
            ],
        },
    }


def location_matches(
    conditions: Sequence[GoldCondition], prediction: Mapping[str, Any]
) -> list[dict[str, Any]]:
    findings = candidate_findings(prediction)
    evidence = [finding_evidence(finding) for finding in findings]
    matrix = [
        [pair_score(finding_ranges, condition) for condition in conditions]
        for finding_ranges in evidence
    ]
    size = max(len(findings), len(conditions))
    potentials_by_row = [0.0] * (size + 1)
    potentials_by_column = [0.0] * (size + 1)
    row_by_column = [0] * (size + 1)
    previous_column = [0] * (size + 1)
    for row in range(1, size + 1):
        row_by_column[0] = row
        minimum_cost = [float("inf")] * (size + 1)
        used = [False] * (size + 1)
        column = 0
        while True:
            used[column] = True
            active_row = row_by_column[column]
            delta, next_column = float("inf"), 0
            for candidate_column in range(1, size + 1):
                if used[candidate_column]:
                    continue
                score = (
                    matrix[active_row - 1][candidate_column - 1]["score"]
                    if active_row <= len(findings)
                    and candidate_column <= len(conditions)
                    else 0.0
                )
                cost = (
                    -score
                    - potentials_by_row[active_row]
                    - potentials_by_column[candidate_column]
                )
                if cost < minimum_cost[candidate_column]:
                    minimum_cost[candidate_column] = cost
                    previous_column[candidate_column] = column
                if minimum_cost[candidate_column] < delta:
                    delta = minimum_cost[candidate_column]
                    next_column = candidate_column
            for candidate_column in range(size + 1):
                if used[candidate_column]:
                    potentials_by_row[row_by_column[candidate_column]] += delta
                    potentials_by_column[candidate_column] -= delta
                else:
                    minimum_cost[candidate_column] -= delta
            column = next_column
            if row_by_column[column] == 0:
                break
        while True:
            previous = previous_column[column]
            row_by_column[column] = row_by_column[previous]
            column = previous
            if column == 0:
                break
    assignment = [-1] * len(findings)
    for condition_index in range(len(conditions)):
        finding_index = row_by_column[condition_index + 1] - 1
        if (
            0 <= finding_index < len(findings)
            and matrix[finding_index][condition_index]["score"] > 0.0
        ):
            assignment[finding_index] = condition_index
    matches = []
    for finding_index, condition_index in enumerate(assignment):
        if condition_index < 0:
            continue
        details = matrix[finding_index][condition_index]
        matches.append(
            {
                "finding_id": str(findings[finding_index]["finding_id"]),
                "condition_id": conditions[condition_index].condition_id,
                "score": round(details["score"], 6),
                "file_exact": details["file_exact"],
                "symbol_compatible": details["symbol_compatible"],
                "line_overlap": details["line_overlap"],
                "match_basis": details["match_basis"],
                "evidence_location": details["evidence_location"],
                "gold_target": details["gold_target"],
            }
        )
    return matches


def _ratio(numerator: float, denominator: float, *, empty: float = 0.0) -> float:
    return round(numerator / denominator, 6) if denominator else empty


def rule_based_score(
    gold: Mapping[str, Any],
    prediction: Mapping[str, Any],
    *,
    formal_data_root: Path | None = None,
) -> dict[str, Any]:
    conditions = load_gold_conditions(gold, formal_data_root=formal_data_root)
    scorable_conditions = [
        condition for condition in conditions if condition.location_targets
    ]
    matches = location_matches(scorable_conditions, prediction)
    prediction_count = len(candidate_findings(prediction))
    score_sum = sum(item["score"] for item in matches)
    precision = _ratio(score_sum, prediction_count)
    recall = _ratio(score_sum, len(scorable_conditions))
    return {
        "schema_version": SCHEMA_VERSION,
        "scorer": "location",
        "location_source": "all_code_evidence",
        "gold_labels": len(scorable_conditions),
        "excluded_gold_labels": len(conditions) - len(scorable_conditions),
        "predicted_findings": prediction_count,
        "matched_labels": len(matches),
        "matched_score": round(score_sum, 6),
        "location_precision": precision,
        "location_recall": recall,
        "location_f1": _ratio(2 * precision * recall, precision + recall),
        "matches": matches,
    }


def llm_metrics(
    gold: Mapping[str, Any],
    prediction: Mapping[str, Any],
    judgment: Mapping[str, Any],
) -> dict[str, float]:
    matches = judgment["matches"]
    prediction_count = len(candidate_findings(prediction))
    matched_score = sum(float(item["match_score"]) for item in matches)
    recall = _ratio(matched_score, len(gold.get("conditions", [])), empty=1.0)
    precision = _ratio(matched_score, prediction_count, empty=1.0)
    return {
        "gold_precision": precision,
        "recall": recall,
        "f1": _ratio(2 * precision * recall, precision + recall),
        "question_quality": _ratio(
            sum(
                float(item["match_score"]) * float(item["question_score"])
                for item in matches
            ),
            matched_score,
        ),
    }


def judge_prediction(
    gold: Mapping[str, Any],
    prediction: Mapping[str, Any],
    client: JsonJudgeClient,
    *,
    document_after: str | None = None,
    repository_root: Path | None = None,
    formal_data_root: Path | None = None,
) -> dict[str, Any]:
    if gold.get("input_id") != prediction.get("input_id"):
        raise ValueError("Gold and prediction input_id values differ")
    findings = candidate_findings(prediction)
    messages = build_llm_messages(
        gold,
        prediction,
        document_after=document_after,
        repository_root=repository_root,
        formal_data_root=formal_data_root,
    )
    judgment = validate_llm_response(
        client.complete(messages),
        finding_ids={item["finding_id"] for item in findings},
        condition_ids={str(item["condition_id"]) for item in gold["conditions"]},
    )
    return {
        "input_id": gold["input_id"],
        "benchmark": "specgap",
        "rule_based": rule_based_score(
            gold, prediction, formal_data_root=formal_data_root
        ),
        "llm_judge": {
            "judgment": judgment,
            "metrics": llm_metrics(gold, prediction, judgment),
        },
    }
