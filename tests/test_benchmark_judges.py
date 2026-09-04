from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, relative_path: str) -> ModuleType:
    path = PROJECT_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SPECGAP = _load_module(
    "beg_specgap_judge", "evaluation/specgap/judge/judge.py"
)
SILENTSWAP = _load_module(
    "beg_silentswap_judge", "evaluation/silentswap/judge/judge.py"
)


class FakeJudgeClient:
    def __init__(self, response: Mapping[str, Any]) -> None:
        self.response = response
        self.messages: list[dict[str, str]] | None = None

    def complete(self, messages: list[dict[str, str]]) -> Mapping[str, Any]:
        self.messages = messages
        return self.response


class SpecGapJudgeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gold = {
            "input_id": "sg_test",
            "conditions": [
                {
                    "condition_id": "kc_001",
                    "normalized_condition": "Public calls return one.",
                    "implementation_locations": [
                        {
                            "file": "pkg/api.py",
                            "symbol": "public",
                            "line_ranges": [{"start": 10, "end": 11}],
                        }
                    ],
                }
            ],
        }
        self.finding = {
            "finding_id": "F001",
            "claim": "Public calls return one.",
            "verification_question": "Should public calls return one?",
            "why_important": "Callers observe the result.",
            "downstream_impact": "The API value changes.",
            "code_evidence": [
                {
                    "path": "pkg/api.py",
                    "symbol": "public",
                    "start_line": 10,
                    "end_line": 11,
                    "explanation": "The return statement fixes the value.",
                }
            ],
        }

    def test_native_beg_finding_is_scored_without_lossy_conversion(self) -> None:
        prediction = {
            "input_id": "sg_test",
            "benchmark": "specgap",
            "findings": [self.finding],
        }
        client = FakeJudgeClient(
            {
                "matches": [
                    {
                        "finding_id": "F001",
                        "condition_id": "kc_001",
                        "match_score": 1.0,
                        "question_score": 1.0,
                        "reason": "same behavior",
                    }
                ],
                "unmatched_findings": [],
            }
        )

        result = SPECGAP.judge_prediction(
            self.gold,
            prediction,
            client,
            document_after="Visible document.",
        )

        self.assertEqual(result["rule_based"]["location_f1"], 1.0)
        self.assertEqual(result["llm_judge"]["metrics"]["f1"], 1.0)
        payload = json.loads(client.messages[1]["content"])
        self.assertEqual(payload["candidate_findings"][0], self.finding)
        self.assertEqual(payload["document_after"], "Visible document.")

    def test_legacy_condition_contract_remains_scoreable(self) -> None:
        prediction = {
            "input_id": "sg_test",
            "benchmark": "specgap",
            "conditions": [
                {
                    "normalized_condition": self.finding["claim"],
                    "expected_verification_question": self.finding[
                        "verification_question"
                    ],
                    "why_important": self.finding["why_important"],
                    "downstream_impact": self.finding["downstream_impact"],
                    "implementation_locations": self.gold["conditions"][0][
                        "implementation_locations"
                    ],
                }
            ],
        }

        score = SPECGAP.rule_based_score(self.gold, prediction)

        self.assertEqual(score["predicted_findings"], 1)
        self.assertEqual(score["location_f1"], 1.0)


def _silent_gold() -> dict[str, Any]:
    return {
        "input_id": "ss_test",
        "swaps": [
            {
                "swap_type": "parsing_matching",
                "original_semantics": f"before {number}",
                "swapped_semantics": f"after {number}",
                "localization": {
                    "file": f"pkg/file_{number}.py",
                    "symbol": {
                        "kind": "function",
                        "qualified_name": [f"function_{number}"],
                    },
                    "line_ranges": [{"start": number, "end": number}],
                },
            }
            for number in range(1, 6)
        ],
    }


def _silent_prediction(*, legacy: bool = False) -> dict[str, Any]:
    swaps = []
    for number, reference in enumerate(_silent_gold()["swaps"], start=1):
        swap = {
            "swap_type": "parsing_matching",
            "code_change": f"operation {number} changes direction",
            "trigger_condition": f"input {number}",
            "behavioral_effect": {
                "before": f"before {number}",
                "after": f"after {number}",
            },
            ("localization" if legacy else "target"): reference["localization"],
        }
        swaps.append(swap)
    return {
        "input_id": "ss_test",
        "benchmark": "silentswap",
        "swaps": swaps,
    }


def _perfect_silent_response() -> dict[str, Any]:
    return {
        "location_checks": [
            {
                "reference_swap_number": number,
                "matched_candidate_swap_number": number,
                "file_correct": True,
                "symbol_kind_correct": True,
                "qualified_name_correct": True,
                "line_range_acceptable": True,
                "note": "correct",
            }
            for number in range(1, 6)
        ],
        "code_change_checks": [
            {
                "reference_swap_number": number,
                "matched_candidate_swap_number": number,
                "concrete_operation_correct": True,
                "all_changed_operations_covered": True,
                "original_logic_correct": True,
                "swapped_logic_correct": True,
                "direction_correct": True,
                "no_material_contradiction": True,
                "note": "correct",
            }
            for number in range(1, 6)
        ],
        "score_notes": {},
        "missing_or_incorrect": [],
        "contradicted_claims": [],
        "rationale": "all five match",
    }


class SilentSwapJudgeTests(unittest.TestCase):
    def test_native_beg_targets_receive_full_rule_and_judge_scores(self) -> None:
        prediction = _silent_prediction()
        client = FakeJudgeClient(_perfect_silent_response())

        result = SILENTSWAP.judge_prediction(_silent_gold(), prediction, client)

        self.assertEqual(result["rule_based"]["localization_score"], 1.0)
        self.assertEqual(
            result["llm_judge"]["scores"],
            {"location_correct": 1.0, "code_change_correct": 1.0},
        )
        self.assertIn('"target"', client.messages[1]["content"])

    def test_legacy_localization_contract_remains_scoreable(self) -> None:
        score = SILENTSWAP.rule_based_score(
            _silent_gold(), _silent_prediction(legacy=True)
        )

        self.assertEqual(score["predicted_swaps"], 5)
        self.assertEqual(score["localization_score"], 1.0)

    def test_duplicate_candidate_alignment_is_rejected(self) -> None:
        response = _perfect_silent_response()
        response["location_checks"][1]["matched_candidate_swap_number"] = 1

        with self.assertRaisesRegex(ValueError, "one-to-one"):
            SILENTSWAP.validate_llm_response(response)


if __name__ == "__main__":
    unittest.main()
