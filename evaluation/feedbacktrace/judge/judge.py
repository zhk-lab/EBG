"""Latest formal FeedbackTrace joint Judge contract.

The scoring rules are migrated from the desktop FeedbackTrace evaluator. Model
transport and batch orchestration remain outside this module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol


PROMPT_PATH = (
    Path(__file__).resolve().parents[3]
    / "prompts"
    / "judge"
    / "feedbacktrace.txt"
)
JUDGE_RESPONSE_KEYS = {
    "verification_point_relation",
    "evidence_case_evaluation",
}
VERIFICATION_POINT_RELATIONS = {"equivalent", "partial", "different"}
JUDGE_EVIDENCE_EVALUATION_KEYS = {"is_evidence", "direct", "sufficient"}
EVIDENCE_ID_RELATIONS = {
    "disjoint",
    "exact",
    "superset",
    "proper_subset",
    "partial_overlap",
}
EVIDENCE_CASE_PROMPT_MARKER = "{{EVIDENCE_ID_RELATION_INSTRUCTION}}"
JUDGE_RUBRIC_ID = "feedbacktrace_evaluation_rubric_case_routed"
JUDGE_EVIDENCE_EVENT_KEYS = (
    "evidence_id",
    "event_type",
    "content",
    "turn_number",
    "tool_name",
    "tool_result_present",
    "tool_result_turn_number",
)


class JsonJudgeClient(Protocol):
    def complete(self, messages: list[dict[str, str]]) -> Mapping[str, Any]: ...


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )

def evidence_map(model_input: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    events = model_input.get("events")
    if not isinstance(events, list):
        raise ValueError(f"{model_input.get('input_id')}: events must be a list")
    for event in events:
        if not isinstance(event, dict):
            raise ValueError(f"{model_input.get('input_id')}: event must be an object")
        evidence_id = event.get("evidence_id")
        if evidence_id is None:
            continue
        if not isinstance(evidence_id, str) or not evidence_id:
            raise ValueError(f"{model_input.get('input_id')}: invalid evidence ID")
        if evidence_id in result:
            raise ValueError(f"{model_input.get('input_id')}: duplicate evidence ID {evidence_id}")
        result[evidence_id] = event
    return result

def evidence_location_score_from_evaluation(evaluation: Any) -> float:
    if not isinstance(evaluation, dict):
        raise ValueError("evidence evaluation must be an object")
    if set(evaluation) != JUDGE_EVIDENCE_EVALUATION_KEYS:
        raise ValueError("evidence evaluation fields are invalid")
    if any(
        type(evaluation[key]) is not bool
        for key in JUDGE_EVIDENCE_EVALUATION_KEYS
    ):
        raise ValueError("evidence evaluation values must be booleans")
    if not evaluation["is_evidence"]:
        if evaluation["direct"] or evaluation["sufficient"]:
            raise ValueError("non-evidence cannot be direct or sufficient")
        return 0.0
    if evaluation["direct"] and evaluation["sufficient"]:
        return 1.0
    return 0.5


def verification_point_alignment_from_relation(relation: Any) -> float:
    if (
        not isinstance(relation, str)
        or relation not in VERIFICATION_POINT_RELATIONS
    ):
        raise ValueError("verification point relation is invalid")
    return {
        "equivalent": 1.0,
        "partial": 0.5,
        "different": 0.0,
    }[relation]


def evidence_id_relation(
    gold_evidence_ids: Iterable[str],
    candidate_evidence_ids: Iterable[str],
) -> str:
    gold = set(gold_evidence_ids)
    candidate = set(candidate_evidence_ids)
    if not gold:
        raise ValueError("Gold evidence set must be non-empty")
    if not gold & candidate:
        return "disjoint"
    if candidate == gold:
        return "exact"
    if gold < candidate:
        return "superset"
    if candidate < gold:
        return "proper_subset"
    return "partial_overlap"


def evidence_case_response_keys(
    relation: str,
    *,
    force_false: bool = False,
) -> set[str]:
    if relation not in EVIDENCE_ID_RELATIONS:
        raise ValueError("invalid Evidence ID relation")
    if force_false or relation in {"exact", "proper_subset"}:
        return set()
    if relation == "superset":
        return {"material_contradiction"}
    if relation == "partial_overlap":
        return {"replacement_sufficient"}
    return set(JUDGE_EVIDENCE_EVALUATION_KEYS)


def evidence_evaluation_from_case(
    relation: str,
    case_evaluation: dict[str, Any],
    *,
    force_false: bool = False,
) -> dict[str, bool]:
    expected_keys = evidence_case_response_keys(relation, force_false=force_false)
    if set(case_evaluation) != expected_keys:
        raise ValueError("Evidence case evaluation fields are invalid")
    if any(type(value) is not bool for value in case_evaluation.values()):
        raise ValueError("Evidence case evaluation values must be booleans")
    if force_false:
        return {"is_evidence": False, "direct": False, "sufficient": False}
    if relation == "exact":
        return {"is_evidence": True, "direct": True, "sufficient": True}
    if relation == "proper_subset":
        return {"is_evidence": True, "direct": True, "sufficient": False}
    if relation == "superset":
        value = not case_evaluation["material_contradiction"]
        return {"is_evidence": value, "direct": value, "sufficient": value}
    if relation == "partial_overlap":
        return {
            "is_evidence": True,
            "direct": True,
            "sufficient": case_evaluation["replacement_sufficient"],
        }
    return {
        key: case_evaluation[key]
        for key in ("is_evidence", "direct", "sufficient")
    }


def validate_judge_response(
    value: Any,
    *,
    evidence_id_relation_value: str = "disjoint",
    prediction_verdict: str | None = None,
    evidence_ids_valid: bool = True,
) -> list[str]:
    if not isinstance(value, dict):
        return ["judge_not_object"]
    errors: list[str] = []
    if set(value) != JUDGE_RESPONSE_KEYS:
        errors.append("judge_fields_not_exact")
    relation = value.get("verification_point_relation")
    if (
        not isinstance(relation, str)
        or relation not in VERIFICATION_POINT_RELATIONS
    ):
        errors.append("judge_verification_point_relation_invalid")

    force_false = prediction_verdict != "KEY" or not evidence_ids_valid
    expected_case_keys = evidence_case_response_keys(
        evidence_id_relation_value,
        force_false=force_false,
    )
    evaluation = value.get("evidence_case_evaluation")
    if not isinstance(evaluation, dict):
        errors.append("judge_evidence_case_evaluation_not_object")
    else:
        if set(evaluation) != expected_case_keys:
            errors.append("judge_evidence_case_evaluation_fields_not_exact")
        boolean_values_valid = all(
            type(evaluation.get(key)) is bool for key in expected_case_keys
        )
        if not boolean_values_valid:
            errors.append("judge_evidence_case_evaluation_values_invalid")
        elif (
            expected_case_keys == JUDGE_EVIDENCE_EVALUATION_KEYS
            and not evaluation["is_evidence"]
            and (evaluation["direct"] or evaluation["sufficient"])
        ):
            errors.append("judge_non_evidence_cannot_be_direct_or_sufficient")

    if prediction_verdict != "KEY":
        if relation != "different":
            errors.append("judge_forced_different_relation_invalid")
    return sorted(set(errors))


def render_judge_prompt(
    judge_prompt: str,
    *,
    evidence_id_relation_value: str,
    force_false_evidence: bool,
) -> str:
    if judge_prompt.count(EVIDENCE_CASE_PROMPT_MARKER) != 1:
        raise ValueError("Judge prompt must contain the Evidence case marker once")
    if evidence_id_relation_value not in EVIDENCE_ID_RELATIONS:
        raise ValueError("invalid Evidence ID relation")

    if force_false_evidence:
        instruction = f"""### Applicable Evidence ID-set case

The program computed evidence_id_relation={evidence_id_relation_value}. The
Prediction has an invalid verdict or invalid Evidence IDs, so the program will set
is_evidence, direct, and sufficient to false. Do not perform an Evidence
semantic judgment. Return evidence_case_evaluation as an empty object.

Return exactly one JSON object with exactly these fields and no explanation:

{{
  \"verification_point_relation\": \"<chosen_relation>\",
  \"evidence_case_evaluation\": {{}}
}}"""
    elif evidence_id_relation_value == "disjoint":
        instruction = """### Applicable Evidence ID-set case: disjoint

Candidate Evidence and Gold Evidence have no IDs in common. Gold is minimal,
not exhaustive, so all three Evidence booleans may still be true. Judge
is_evidence, direct, and sufficient using their definitions above.

Return exactly one JSON object with exactly these fields and no explanation:

{
  \"verification_point_relation\": \"<chosen_relation>\",
  \"evidence_case_evaluation\": {
    \"is_evidence\": false,
    \"direct\": false,
    \"sufficient\": false
  }
}"""
    elif evidence_id_relation_value == "exact":
        instruction = """### Applicable Evidence ID-set case: exact

Candidate Evidence is exactly the Gold Evidence set. The program will set
is_evidence, direct, and sufficient to true. Do not perform an Evidence
semantic judgment. Return evidence_case_evaluation as an empty object.

Return exactly one JSON object with exactly these fields and no explanation:

{
  \"verification_point_relation\": \"<chosen_relation>\",
  \"evidence_case_evaluation\": {}
}"""
    elif evidence_id_relation_value == "superset":
        instruction = """### Applicable Evidence ID-set case: superset

Candidate Evidence contains the complete Gold Evidence set and has additional
items. Judge only whether any additional Candidate item materially contradicts
Gold. An irrelevant item or an item about another issue is not a contradiction.
The program will set all three Evidence booleans to true when there is no
material contradiction and all three to false when there is one.

Return exactly one JSON object with exactly these fields and no explanation:

{
  \"verification_point_relation\": \"<chosen_relation>\",
  \"evidence_case_evaluation\": {
    \"material_contradiction\": false
  }
}"""
    elif evidence_id_relation_value == "proper_subset":
        instruction = """### Applicable Evidence ID-set case: proper_subset

Candidate Evidence is a non-empty proper subset of Gold Evidence. Because Gold
is minimal, the program will set is_evidence=true, direct=true, and
sufficient=false. Do not perform an Evidence semantic judgment. Return
evidence_case_evaluation as an empty object.

Return exactly one JSON object with exactly these fields and no explanation:

{
  \"verification_point_relation\": \"<chosen_relation>\",
  \"evidence_case_evaluation\": {}
}"""
    else:
        instruction = """### Applicable Evidence ID-set case: partial_overlap

Candidate Evidence overlaps Gold Evidence, but neither set contains the other.
The matching Gold items already establish is_evidence=true and direct=true.
Judge only whether the different Candidate items fully replace the proof effect
of every missing Gold item and introduce no material contradiction.

Return exactly one JSON object with exactly these fields and no explanation:

{
  \"verification_point_relation\": \"<chosen_relation>\",
  \"evidence_case_evaluation\": {
    \"replacement_sufficient\": false
  }
}"""
    return judge_prompt.replace(EVIDENCE_CASE_PROMPT_MARKER, instruction)

def judge_payload(
    *,
    model_input: dict[str, Any],
    annotation: dict[str, Any],
    prediction: dict[str, Any],
) -> dict[str, Any]:
    selectable_evidence = evidence_map(model_input)

    gold_evidence_ids = annotation.get("gold_evidence_ids")
    if not isinstance(gold_evidence_ids, list) or not gold_evidence_ids:
        raise ValueError(f"{model_input['input_id']}: Gold evidence IDs are invalid")
    if (
        not all(isinstance(evidence_id, str) for evidence_id in gold_evidence_ids)
        or len(gold_evidence_ids) != len(set(gold_evidence_ids))
        or any(
            evidence_id not in selectable_evidence
            for evidence_id in gold_evidence_ids
        )
    ):
        raise ValueError(
            f"{model_input['input_id']}: Gold evidence IDs are not selectable"
        )

    predicted_evidence_ids = prediction.get("supporting_evidence_ids")
    predicted_verdict = prediction.get("verdict")
    evidence_ids_valid = (
        isinstance(predicted_evidence_ids, list)
        and all(
            isinstance(evidence_id, str) and evidence_id in selectable_evidence
            for evidence_id in predicted_evidence_ids
        )
        and len(predicted_evidence_ids) == len(set(predicted_evidence_ids))
        and (
            (predicted_verdict == "KEY" and 1 <= len(predicted_evidence_ids) <= 2)
        )
    )

    def selected_events(evidence_ids: list[str]) -> list[dict[str, Any]]:
        return [
            {
                key: selectable_evidence[evidence_id][key]
                for key in JUDGE_EVIDENCE_EVENT_KEYS
                if key in selectable_evidence[evidence_id]
            }
            for evidence_id in evidence_ids
            if evidence_id in selectable_evidence
        ]

    candidate_ids_for_relation = (
        [
            evidence_id
            for evidence_id in predicted_evidence_ids
            if isinstance(evidence_id, str)
        ]
        if isinstance(predicted_evidence_ids, list)
        else []
    )
    return {
        "evidence_id_relation": evidence_id_relation(
            gold_evidence_ids,
            candidate_ids_for_relation,
        ),
        "gold": {
            "verdict": annotation["verdict"],
            "gold_verification_point": annotation["gold_verification_point"],
            "evidence": selected_events(gold_evidence_ids),
        },
        "prediction": {
            "verdict": predicted_verdict,
            "verification_point": prediction["verification_point"],
            "evidence_ids_valid": evidence_ids_valid,
            "evidence": selected_events(
                predicted_evidence_ids
                if isinstance(predicted_evidence_ids, list)
                else []
            ),
        },
    }


def build_llm_messages(
    annotation: Mapping[str, Any],
    prediction: Mapping[str, Any],
    *,
    model_input: Mapping[str, Any],
) -> list[dict[str, str]]:
    """Build the latest case-routed Judge request."""

    payload = judge_payload(
        model_input=dict(model_input),
        annotation=dict(annotation),
        prediction=dict(prediction),
    )
    relation = str(payload["evidence_id_relation"])
    evidence_ids_valid = bool(payload["prediction"]["evidence_ids_valid"])
    force_false = prediction.get("verdict") != "KEY" or not evidence_ids_valid
    rendered_prompt = render_judge_prompt(
        PROMPT_PATH.read_text(encoding="utf-8").strip(),
        evidence_id_relation_value=relation,
        force_false_evidence=force_false,
    )
    return [
        {"role": "system", "content": rendered_prompt},
        {"role": "user", "content": canonical_json(payload)},
    ]


def validate_llm_response(
    value: Mapping[str, Any],
    *,
    evidence_id_relation_value: str,
    prediction_verdict: str | None,
    evidence_ids_valid: bool,
) -> dict[str, Any]:
    """Validate and normalize one latest-format Judge response."""

    errors = validate_judge_response(
        value,
        evidence_id_relation_value=evidence_id_relation_value,
        prediction_verdict=prediction_verdict,
        evidence_ids_valid=evidence_ids_valid,
    )
    if errors:
        raise ValueError(", ".join(errors))
    force_false = prediction_verdict != "KEY" or not evidence_ids_valid
    evidence_evaluation = evidence_evaluation_from_case(
        evidence_id_relation_value,
        dict(value["evidence_case_evaluation"]),
        force_false=force_false,
    )
    relation = value["verification_point_relation"]
    return {
        "verification_point_relation": relation,
        "verification_point_alignment": (
            verification_point_alignment_from_relation(relation)
        ),
        "evidence_evaluation": evidence_evaluation,
        "evidence_location_score": evidence_location_score_from_evaluation(
            evidence_evaluation
        ),
    }


def judge_prediction(
    annotation: Mapping[str, Any],
    prediction: Mapping[str, Any],
    client: JsonJudgeClient,
    *,
    model_input: Mapping[str, Any],
) -> dict[str, Any]:
    """Run the transport-independent latest joint Judge for one sample."""

    if model_input.get("input_id") != prediction.get("input_id"):
        raise ValueError("model input and prediction input_id values differ")
    if annotation.get("verdict") != "KEY":
        raise ValueError("semantic Judge accepts only Gold KEY samples")
    payload = judge_payload(
        model_input=dict(model_input),
        annotation=dict(annotation),
        prediction=dict(prediction),
    )
    response = client.complete(
        build_llm_messages(annotation, prediction, model_input=model_input)
    )
    normalized = validate_llm_response(
        response,
        evidence_id_relation_value=str(payload["evidence_id_relation"]),
        prediction_verdict=prediction.get("verdict"),
        evidence_ids_valid=bool(payload["prediction"]["evidence_ids_valid"]),
    )
    return {
        "input_id": model_input["input_id"],
        "benchmark": "feedbacktrace",
        **normalized,
        "judge_rubric": JUDGE_RUBRIC_ID,
    }


def verification_point_alignment_score(
    results: list[Mapping[str, Any]],
) -> float:
    if not results:
        return 0.0
    values = [result.get("verification_point_alignment") for result in results]
    if any(type(value) not in {int, float} or value not in {0, 0.5, 1} for value in values):
        raise ValueError("invalid verification-point alignment")
    return round(sum(float(value) for value in values) / len(values), 6)


def evidence_location_score(results: list[Mapping[str, Any]]) -> float:
    if not results:
        return 0.0
    values = [result.get("evidence_location_score") for result in results]
    if any(type(value) not in {int, float} or value not in {0, 0.5, 1} for value in values):
        raise ValueError("invalid evidence-location score")
    return round(sum(float(value) for value in values) / len(values), 6)
