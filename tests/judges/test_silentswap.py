from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any


HERE = Path(__file__).resolve().parent
STAGING_ROOT = HERE.parents[1]


def _load_judge() -> ModuleType:
    path = STAGING_ROOT / "evaluation" / "silentswap" / "judge" / "judge.py"
    spec = importlib.util.spec_from_file_location("aligned_silentswap_judge", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


JUDGE = _load_judge()


def _location(number: int) -> dict[str, Any]:
    return {
        "file": f"pkg/file_{number}.py",
        "symbol": {
            "kind": "function",
            "qualified_name": [f"function_{number}"],
        },
        "line_ranges": [{"start": number, "end": number}],
    }


def _gold() -> dict[str, Any]:
    swaps = [
        {
            "swap_type": "parsing_matching",
            "original_semantics": f"before {number}",
            "swapped_semantics": f"after {number}",
            "evidence": [f"evidence {number}"],
            "localization": _location(number),
        }
        for number in range(1, 6)
    ]
    return {
        "input_id": "ss_test",
        "swaps": swaps,
        "source_line_counts": {
            f"pkg/file_{number}.py": 100 for number in range(1, 6)
        },
        "official_judge_reference": {
            "swaps": [
                {
                    "swap_number": number,
                    "gold": {
                        field: swaps[number - 1][field]
                        for field in JUDGE.GOLD_REFERENCE_FIELDS
                    },
                    "localization": swaps[number - 1]["localization"],
                    "document_target": f"document {number}",
                    "executed_behavior_difference": {
                        "on_original": {"stdout": f"before {number}"},
                        "after_all_five_swaps": {"stdout": f"after {number}"},
                    },
                }
                for number in range(1, 6)
            ],
            "changed_files": [f"pkg/file_{number}.py" for number in range(1, 6)],
        },
    }


def _prediction() -> dict[str, Any]:
    return {
        "input_id": "ss_test",
        "benchmark": "silentswap",
        "swaps": [
            {
                "target": _location(number),
                "swap_type": "parsing_matching",
                "code_change": f"operation {number}: before to after",
                "trigger_condition": f"input {number}",
                "behavioral_effect": {
                    "before": f"before {number}",
                    "after": f"after {number}",
                },
            }
            for number in range(1, 6)
        ],
    }


def _response(*, location_correct: int, code_correct: int) -> dict[str, Any]:
    return {
        "location_checks": [
            {
                "reference_swap_number": number,
                "matched_candidate_swap_number": number,
                "file_correct": number <= location_correct,
                "symbol_kind_correct": number <= location_correct,
                "qualified_name_correct": number <= location_correct,
                "line_range_acceptable": number <= location_correct,
                "note": "correct" if number <= location_correct else "incorrect",
            }
            for number in range(1, 6)
        ],
        "code_change_checks": [
            {
                "reference_swap_number": number,
                "matched_candidate_swap_number": number,
                "concrete_operation_correct": number <= code_correct,
                "all_changed_operations_covered": number <= code_correct,
                "original_logic_correct": number <= code_correct,
                "swapped_logic_correct": number <= code_correct,
                "direction_correct": number <= code_correct,
                "no_material_contradiction": number <= code_correct,
                "note": "correct" if number <= code_correct else "incorrect",
            }
            for number in range(1, 6)
        ],
        "score_notes": {
            "location_correct": "location tier details",
            "code_change_correct": "code tier details",
        },
        "missing_or_incorrect": [],
        "contradicted_claims": [],
        "rationale": "strict assessment",
    }


class FormalJudgeAlignmentTests(unittest.TestCase):
    def test_native_prediction_receives_full_rule_and_judge_scores(self) -> None:
        client = SimpleNamespace(
            complete=lambda messages: _response(location_correct=5, code_correct=5)
        )
        result = JUDGE.judge_prediction(_gold(), _prediction(), client)
        self.assertEqual(result["rule_based"]["localization_score"], 1.0)
        self.assertEqual(
            result["llm_judge"]["scores"],
            {"location_correct": 1.0, "code_change_correct": 1.0},
        )

    def test_legacy_localization_preserves_location_scores(self) -> None:
        prediction = _prediction()
        for swap in prediction["swaps"]:
            swap["localization"] = swap.pop("target")
        score = JUDGE.rule_based_score(_gold(), prediction)
        self.assertEqual(score, JUDGE.rule_based_score(_gold(), _prediction()))
        self.assertEqual(score["predicted_swaps"], 5)

    def test_ss001_full_message_matches_frozen_formal_runner(self) -> None:
        gold_path = (
            STAGING_ROOT
            / "evaluation/silentswap/artifacts/hidden_gold/ss_001.json"
        )
        gold = json.loads(gold_path.read_text(encoding="utf-8"))
        messages = JUDGE.build_llm_messages(gold, {"swaps": []})
        scripts = Path.home() / "Desktop" / "SilentSwap" / "scripts"
        sys.path.insert(0, str(scripts))
        try:
            import evaluate_deepseek as formal_benchmark
            import evaluate_models as formal_runner
        finally:
            sys.path.pop(0)
        sample = Path.home() / "Desktop" / "SilentSwap" / "data" / "1"
        reference = formal_runner.judge_reference(sample)
        self.assertEqual(
            messages,
            [
                {"role": "system", "content": formal_runner.JUDGE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": formal_benchmark.judge_prompt(
                        reference, {"swaps": []}
                    ),
                },
            ],
        )
        reference = JUDGE.build_judge_reference(gold)
        self.assertEqual(
            set(reference["swaps"][0]),
            {
                "swap_number",
                "gold",
                "localization",
                "document_target",
                "executed_behavior_difference",
            },
        )
        self.assertIn("changed_files", reference)

    def test_objective_localization_matches_formal_assignment(self) -> None:
        prediction = _prediction()
        prediction["swaps"] = list(reversed(prediction["swaps"]))

        score = JUDGE.rule_based_score(_gold(), prediction)

        self.assertEqual(score["localization_score"], 1.0)
        self.assertTrue(score["file_exact"])
        self.assertTrue(score["symbol_exact"])
        self.assertEqual(
            [item["gold_swap_number"] for item in score["matched_swaps"]],
            [5, 4, 3, 2, 1],
        )

    def test_semantic_tiers_and_notes_match_formal_runner(self) -> None:
        partial = JUDGE.validate_llm_response(
            _response(location_correct=3, code_correct=4)
        )
        failed = JUDGE.validate_llm_response(
            _response(location_correct=2, code_correct=3)
        )
        perfect = JUDGE.validate_llm_response(
            _response(location_correct=5, code_correct=5)
        )

        self.assertEqual(
            partial["scores"],
            {"location_correct": 0.5, "code_change_correct": 0.5},
        )
        self.assertEqual(
            failed["scores"],
            {"location_correct": 0, "code_change_correct": 0},
        )
        self.assertEqual(
            perfect["scores"],
            {"location_correct": 1, "code_change_correct": 1},
        )
        self.assertEqual(
            partial["score_notes"]["location_correct"],
            "3/5 swaps fully correct. location tier details",
        )
        self.assertEqual(
            partial["score_notes"]["code_change_correct"],
            "4/5 swaps fully correct. code tier details",
        )
        self.assertEqual(JUDGE.validate_llm_response(partial), partial)

    def test_duplicate_alignment_is_rejected(self) -> None:
        response = _response(location_correct=5, code_correct=5)
        response["location_checks"][1]["matched_candidate_swap_number"] = 1
        with self.assertRaisesRegex(ValueError, "one-to-one"):
            JUDGE.validate_llm_response(response)

    def test_partial_localization_matches_any_gold_and_keeps_denominator(self) -> None:
        for count in range(6):
            with self.subTest(count=count):
                prediction = _prediction()
                prediction["swaps"] = list(reversed(prediction["swaps"]))[:count]
                score = JUDGE.rule_based_score(_gold(), prediction)
                self.assertEqual(score["predicted_swaps"], count)
                self.assertEqual(score["gold_swaps"], 5)
                for metric in ("localization_score", "line_precision", "line_recall", "line_f1"):
                    self.assertEqual(score[metric], count / 5)
                for metric in ("file_exact", "symbol_exact", "line_hit_at_0", "line_hit_at_3"):
                    self.assertEqual(score[metric], count == 5)
                self.assertEqual(
                    [item["gold_swap_number"] for item in score["matched_swaps"]],
                    list(range(5, 5 - count, -1)),
                )

    def test_localization_rejects_more_than_five_candidates(self) -> None:
        prediction = _prediction()
        prediction["swaps"].append(prediction["swaps"][0])
        with self.assertRaisesRegex(ValueError, "at most five"):
            JUDGE.rule_based_score(_gold(), prediction)

    def test_partial_answers_keep_five_gold_judge_tiers(self) -> None:
        for count in range(6):
            with self.subTest(count=count):
                prediction = _prediction()
                prediction["swaps"] = prediction["swaps"][:count]
                response = _response(location_correct=count, code_correct=count)
                for dimension in ("location_checks", "code_change_checks"):
                    for check in response[dimension][count:]:
                        check["matched_candidate_swap_number"] = None
                client = SimpleNamespace(complete=lambda messages: response)
                result = JUDGE.judge_prediction(_gold(), prediction, client)
                judgment = result["llm_judge"]
                self.assertEqual(judgment["scores"], {
                    "location_correct": 1 if count == 5 else 0.5 if count >= 3 else 0,
                    "code_change_correct": 1 if count == 5 else 0.5 if count == 4 else 0,
                })
                self.assertEqual(
                    JUDGE.validate_llm_response(judgment, candidate_count=count), judgment
                )

    def test_judge_rejects_alignment_to_an_unreported_candidate(self) -> None:
        prediction = _prediction()
        prediction["swaps"] = prediction["swaps"][:4]
        client = SimpleNamespace(
            complete=lambda messages: _response(location_correct=5, code_correct=5)
        )
        with self.assertRaisesRegex(ValueError, "one-to-one"):
            JUDGE.judge_prediction(_gold(), prediction, client)


if __name__ == "__main__":
    unittest.main()
