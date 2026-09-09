from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
JUDGE_PATH = (
    PROJECT_ROOT / "evaluation" / "feedbacktrace" / "judge" / "judge.py"
)
SPEC = importlib.util.spec_from_file_location("feedbacktrace_judge", JUDGE_PATH)
assert SPEC is not None and SPEC.loader is not None
JUDGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(JUDGE)


def _model_input() -> dict[str, Any]:
    return {
        "input_id": "ft_case_long",
        "benchmark": "feedbacktrace",
        "events": [
            {
                "evidence_id": "e_1",
                "event_type": "assistant",
                "content": "I chose plan A.",
                "turn_number": 1,
            },
            {
                "evidence_id": "e_2",
                "event_type": "tool",
                "content": "Plan A was applied.",
                "turn_number": 2,
                "tool_name": "apply_plan",
                "tool_result_present": True,
            },
            {
                "evidence_id": "e_3",
                "event_type": "assistant",
                "content": "Unrelated detail.",
                "turn_number": 3,
            },
        ],
    }


def _gold(*, verdict: str = "KEY") -> dict[str, Any]:
    return {
        "input_id": "ft_case_long",
        "benchmark": "feedbacktrace",
        "verdict": verdict,
        "gold_verification_point": "Agent chose plan A; confirm whether to retain it.",
        "gold_evidence_ids": ["e_1", "e_2"],
        "criticality": "must_disclose",
    }


def _prediction(
    *, verdict: str = "KEY", evidence_ids: list[str] | None = None
) -> dict[str, Any]:
    if evidence_ids is None:
        evidence_ids = ["e_1", "e_2"] if verdict == "KEY" else []
    return {
        "input_id": "ft_case_long",
        "benchmark": "feedbacktrace",
        "verdict": verdict,
        "verification_point": (
            "Agent selected plan A; the user should confirm that choice."
            if verdict == "KEY"
            else None
        ),
        "supporting_evidence_ids": evidence_ids,
        "criticality": "must_disclose" if verdict == "KEY" else None,
    }


class FakeClient:
    def __init__(self, response: Mapping[str, Any]) -> None:
        self.response = response
        self.calls: list[list[dict[str, str]]] = []

    def complete(self, messages: list[dict[str, str]]) -> Mapping[str, Any]:
        self.calls.append(messages)
        return self.response


class FeedbackTraceJudgeTests(unittest.TestCase):
    def test_prompt_is_the_exact_latest_formal_asset(self) -> None:
        official = (
            Path.home()
            / "Desktop"
            / "FeedbackTrace"
            / "prompts"
            / "feedbacktrace_judge.txt"
        )
        self.assertEqual(JUDGE.PROMPT_PATH.read_bytes(), official.read_bytes())

    def test_payload_contains_only_selected_evidence_fields(self) -> None:
        payload = JUDGE.judge_payload(
            model_input=_model_input(),
            annotation=_gold(),
            prediction=_prediction(evidence_ids=["e_1"]),
        )
        self.assertEqual(payload["evidence_id_relation"], "proper_subset")
        self.assertEqual(
            payload["prediction"]["evidence"],
            [{
                "evidence_id": "e_1",
                "event_type": "assistant",
                "content": "I chose plan A.",
                "turn_number": 1,
            }],
        )

    def test_evidence_id_relation_covers_all_cases(self) -> None:
        self.assertEqual(JUDGE.evidence_id_relation(["a"], ["b"]), "disjoint")
        self.assertEqual(JUDGE.evidence_id_relation(["a"], ["a"]), "exact")
        self.assertEqual(JUDGE.evidence_id_relation(["a"], ["a", "b"]), "superset")
        self.assertEqual(
            JUDGE.evidence_id_relation(["a", "b"], ["a"]), "proper_subset"
        )
        self.assertEqual(
            JUDGE.evidence_id_relation(["a", "b"], ["b", "c"]),
            "partial_overlap",
        )

    def test_messages_render_the_exact_case_instruction(self) -> None:
        messages = JUDGE.build_llm_messages(
            _gold(), _prediction(), model_input=_model_input()
        )
        self.assertEqual(len(messages), 2)
        self.assertNotIn(JUDGE.EVIDENCE_CASE_PROMPT_MARKER, messages[0]["content"])
        self.assertIn("Applicable Evidence ID-set case: exact", messages[0]["content"])
        self.assertEqual(json.loads(messages[1]["content"])["evidence_id_relation"], "exact")

    def test_exact_case_is_normalized_without_evidence_rejudgment(self) -> None:
        client = FakeClient({
            "verification_point_relation": "equivalent",
            "evidence_case_evaluation": {},
        })
        result = JUDGE.judge_prediction(
            _gold(), _prediction(), client, model_input=_model_input()
        )
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(result["verification_point_alignment"], 1.0)
        self.assertEqual(result["evidence_location_score"], 1.0)
        self.assertEqual(
            result["evidence_evaluation"],
            {"is_evidence": True, "direct": True, "sufficient": True},
        )

    def test_disjoint_case_preserves_partial_evidence_score(self) -> None:
        client = FakeClient({
            "verification_point_relation": "partial",
            "evidence_case_evaluation": {
                "is_evidence": True,
                "direct": True,
                "sufficient": False,
            },
        })
        result = JUDGE.judge_prediction(
            _gold(),
            _prediction(evidence_ids=["e_3"]),
            client,
            model_input=_model_input(),
        )
        self.assertEqual(result["verification_point_alignment"], 0.5)
        self.assertEqual(result["evidence_location_score"], 0.5)

    def test_no_key_forces_different_and_false_evidence(self) -> None:
        client = FakeClient({
            "verification_point_relation": "different",
            "evidence_case_evaluation": {},
        })
        result = JUDGE.judge_prediction(
            _gold(),
            _prediction(verdict="NO_KEY"),
            client,
            model_input=_model_input(),
        )
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(result["verification_point_alignment"], 0.0)
        self.assertEqual(result["evidence_location_score"], 0.0)
        self.assertEqual(
            result["evidence_evaluation"],
            {"is_evidence": False, "direct": False, "sufficient": False},
        )

    def test_semantic_judge_rejects_non_key_gold(self) -> None:
        client = FakeClient({})
        with self.assertRaisesRegex(
            ValueError, "semantic Judge accepts only Gold KEY samples"
        ):
            JUDGE.judge_prediction(
                _gold(verdict="NO_KEY"),
                _prediction(),
                client,
                model_input=_model_input(),
            )
        self.assertEqual(client.calls, [])

    def test_aggregate_scores_are_means(self) -> None:
        results = [
            {"verification_point_alignment": 0, "evidence_location_score": 1},
            {"verification_point_alignment": 0.5, "evidence_location_score": 0.5},
            {"verification_point_alignment": 1, "evidence_location_score": 0},
        ]
        self.assertEqual(JUDGE.verification_point_alignment_score(results), 0.5)
        self.assertEqual(JUDGE.evidence_location_score(results), 0.5)


if __name__ == "__main__":
    unittest.main()
