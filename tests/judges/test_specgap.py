from __future__ import annotations

import ast
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
JUDGE_PATH = Path(
    os.environ.get(
        "SPECGAP_JUDGE_PATH",
        PROJECT_ROOT / "evaluation/specgap/judge/judge.py",
    )
)


def _load_judge() -> ModuleType:
    spec = importlib.util.spec_from_file_location("aligned_specgap_judge", JUDGE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {JUDGE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


JUDGE = _load_judge()


class FakeJudgeClient:
    def __init__(self, response: Mapping[str, Any]) -> None:
        self.response = response
        self.messages: list[dict[str, str]] | None = None

    def complete(self, messages: list[dict[str, str]]) -> Mapping[str, Any]:
        self.messages = messages
        return self.response


def _part(number: int) -> dict[str, Any]:
    return {
        "condition_id": f"kc_{number:03d}",
        "type": "behavior",
        "importance": "must_ask",
        "normalized_condition": f"Condition {number} normalized.",
        "source_text": f"Condition {number} source.",
        "expected_verification_question": f"Does condition {number} hold?",
        "why_important": f"Condition {number} matters.",
        "downstream_impact": f"Condition {number} changes callers.",
    }


def _mapping(
    number: int, *, path: str, symbol: str, start: int, end: int
) -> dict[str, Any]:
    return {
        "mapping_status": "direct",
        "coverage": "full",
        "mapping_explanation": f"Mapping {number}.",
        "evidence": [
            {
                "evidence_id": f"kc_{number:03d}_ev001",
                "evidence_type": "implementation",
                "relation": "implements",
                "strength": "direct",
                "locations": [
                    {
                        "file_path": path,
                        "symbol": {
                            "kind": "function",
                            "qualified_name": symbol,
                        },
                        "line_ranges": [{"start": start, "end": end}],
                        "excerpt": f"gold excerpt {number}",
                    }
                ],
                "explanation": f"Implementation {number}.",
                "confidence": "high",
            },
            {
                "evidence_id": f"kc_{number:03d}_test",
                "evidence_type": "test",
                "relation": "validates",
                "strength": "indirect",
                "locations": [],
                "explanation": "This must be excluded by compact_mapping.",
            },
        ],
    }


class SpecGapJudgeAlignmentTests(unittest.TestCase):
    def test_legacy_conditions_preserve_location_scores(self) -> None:
        legacy = {
            "input_id": self.prediction["input_id"],
            "benchmark": "specgap",
            "conditions": [
                {
                    "normalized_condition": finding["claim"],
                    "expected_verification_question": finding["verification_question"],
                    "why_important": finding["why_important"],
                    "downstream_impact": finding["downstream_impact"],
                    "implementation_locations": [
                        {
                            "file": evidence["path"],
                            "symbol": evidence["symbol"],
                            "line_ranges": [{
                                "start": evidence["start_line"],
                                "end": evidence["end_line"],
                            }],
                        }
                        for evidence in finding["code_evidence"]
                    ],
                }
                for finding in self.prediction["findings"]
            ],
        }
        native_score = JUDGE.rule_based_score(
            self.gold, self.prediction, formal_data_root=self.formal_root
        )
        legacy_score = JUDGE.rule_based_score(
            self.gold, legacy, formal_data_root=self.formal_root
        )
        self.assertEqual(legacy_score, native_score)
        self.assertEqual(legacy_score["predicted_findings"], 2)

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.formal_root = self.root / "formal"
        sample = self.formal_root / "1_sample"
        sample.mkdir(parents=True)
        self.repository_root = self.root / "repository"
        (self.repository_root / "pkg").mkdir(parents=True)
        (self.repository_root / "pkg/api.py").write_text(
            "header\ndef public():\n    return 1\n",
            encoding="utf-8",
        )
        (self.repository_root / "pkg/other.py").write_text(
            "one\ntwo\nthree\nfour\nreturn 2\n",
            encoding="utf-8",
        )

        parts = [_part(1), _part(2)]
        mappings = [
            _mapping(1, path="pkg/api.py", symbol="public", start=2, end=3),
            _mapping(2, path="pkg/other.py", symbol="other", start=5, end=5),
        ]
        (sample / "2_deleted_parts.json").write_text(
            json.dumps({"deleted_parts": parts}), encoding="utf-8"
        )
        (sample / "4_code_mapping.json").write_text(
            json.dumps(
                {
                    "code_mappings": {
                        "comparison_items": [
                            {
                                "condition_id": f"kc_{number:03d}",
                                "condition_by_condition": mapping,
                                "holistic_alignment": {
                                    "mapping_status": "no_direct_mapping",
                                    "coverage": "none",
                                    "mapping_explanation": f"HA {number}.",
                                    "evidence": [],
                                },
                            }
                            for number, mapping in enumerate(mappings, start=1)
                        ],
                        "gold_labels": [
                            {
                                "condition_id": f"kc_{number:03d}",
                                "gold_source": "condition_by_condition",
                                "mapping_explanation": f"Manual gold {number}.",
                            }
                            for number in (1, 2)
                        ],
                    }
                }
            ),
            encoding="utf-8",
        )
        self.gold = {
            "input_id": "sg_test",
            "source_sample_id": "sample",
            "conditions": [
                {
                    "condition_id": part["condition_id"],
                    "normalized_condition": part["normalized_condition"],
                    "source_text": part["source_text"],
                }
                for part in parts
            ],
        }
        self.prediction = {
            "input_id": "sg_test",
            "benchmark": "specgap",
            "findings": [
                {
                    "finding_id": "F001",
                    "claim": "Condition 1 normalized.",
                    "verification_question": "Does condition 1 hold?",
                    "why_important": "Condition 1 matters.",
                    "downstream_impact": "Condition 1 changes callers.",
                    "code_evidence": [
                        {
                            "path": "pkg/api.py",
                            "symbol": "public",
                            "start_line": 2,
                            "end_line": 3,
                            "explanation": "The return fixes the value.",
                        }
                    ],
                },
                {
                    "finding_id": "F002",
                    "claim": "Condition 2 normalized.",
                    "verification_question": "Does condition 2 hold?",
                    "why_important": "Condition 2 matters.",
                    "downstream_impact": "Condition 2 changes callers.",
                    "code_evidence": [
                        {
                            "path": "pkg/other.py",
                            "symbol": "different",
                            "start_line": 5,
                            "end_line": 5,
                            "explanation": "The line fixes the value.",
                        }
                    ],
                },
            ],
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_bundled_gold_preserves_messages_and_location_scores(self) -> None:
        from scripts.package_specgap_gold import bundle_gold
        bundled = bundle_gold(self.gold, self.formal_root)
        kwargs = {"document_after": "Visible document.", "repository_root": self.repository_root}
        original = JUDGE.build_llm_messages(self.gold, self.prediction,
                                            formal_data_root=self.formal_root, **kwargs)
        self.assertEqual(JUDGE.build_llm_messages(bundled, self.prediction, **kwargs), original)
        self.assertEqual(JUDGE.rule_based_score(bundled, self.prediction),
                         JUDGE.rule_based_score(self.gold, self.prediction, formal_data_root=self.formal_root))


    def test_golden_message_has_formal_gold_and_resolved_candidate_source(self) -> None:
        gold = {**self.gold, "conditions": self.gold["conditions"][:1]}
        prediction = {**self.prediction, "findings": self.prediction["findings"][:1]}
        messages = JUDGE.build_llm_messages(
            gold,
            prediction,
            document_after="Visible document.",
            repository_root=self.repository_root,
            formal_data_root=self.formal_root,
        )
        expected_gold = {
            "condition_id": "kc_001",
            "type": "behavior",
            "importance": "must_ask",
            "normalized_condition": "Condition 1 normalized.",
            "source_text": "Condition 1 source.",
            "expected_verification_question": "Does condition 1 hold?",
            "why_important": "Condition 1 matters.",
            "downstream_impact": "Condition 1 changes callers.",
            "manual_gold_explanation": "Manual gold 1.",
            "manual_gold_source": "condition_by_condition",
            "condition_by_condition": {
                "mapping_status": "direct",
                "coverage": "full",
                "mapping_explanation": "Mapping 1.",
                "evidence": [
                    {
                        "evidence_id": "kc_001_ev001",
                        "evidence_type": "implementation",
                        "relation": "implements",
                        "strength": "direct",
                        "locations": [
                            {
                                "file_path": "pkg/api.py",
                                "symbol": {
                                    "kind": "function",
                                    "qualified_name": "public",
                                },
                                "line_ranges": [{"start": 2, "end": 3}],
                                "excerpt": "gold excerpt 1",
                            }
                        ],
                        "explanation": "Implementation 1.",
                    }
                ],
            },
            "holistic_alignment": {
                "mapping_status": "no_direct_mapping",
                "coverage": "none",
                "mapping_explanation": "HA 1.",
                "evidence": [],
            },
        }
        expected_finding = {
            **prediction["findings"][0],
            "code_evidence": [
                {
                    **prediction["findings"][0]["code_evidence"][0],
                    "resolved": True,
                    "excerpt": "     2 | def public():\n     3 |     return 1",
                }
            ],
        }
        expected_payload = {
            "document_after": "Visible document.",
            "gold_conditions": [expected_gold],
            "candidate_findings": [expected_finding],
        }
        self.assertEqual(
            messages,
            [
                {
                    "role": "system",
                    "content": JUDGE.PROMPT_PATH.read_text(
                        encoding="utf-8"
                    ).strip(),
                },
                {
                    "role": "user",
                    "content": json.dumps(expected_payload, ensure_ascii=False),
                },
            ],
        )

    def test_golden_location_and_semantic_scores_match_formal_rules(self) -> None:
        response = {
            "matches": [
                {
                    "finding_id": "F001",
                    "condition_id": "kc_001",
                    "match_score": 1.0,
                    "question_score": 1.0,
                    "reason": "full",
                },
                {
                    "finding_id": "F002",
                    "condition_id": "kc_002",
                    "match_score": 0.5,
                    "question_score": 0.5,
                    "reason": "partial",
                },
            ],
            "unmatched_findings": [],
        }
        client = FakeJudgeClient(response)
        result = JUDGE.judge_prediction(
            self.gold,
            self.prediction,
            client,
            document_after="Visible document.",
            repository_root=self.repository_root,
            formal_data_root=self.formal_root,
        )
        self.assertEqual(
            result["rule_based"],
            {
                "schema_version": "specgap-eval-1.12",
                "scorer": "location",
                "location_source": "all_code_evidence",
                "gold_labels": 2,
                "excluded_gold_labels": 0,
                "predicted_findings": 2,
                "matched_labels": 2,
                "matched_score": 1.5,
                "location_precision": 0.75,
                "location_recall": 0.75,
                "location_f1": 0.75,
                "matches": result["rule_based"]["matches"],
            },
        )
        self.assertEqual(
            result["llm_judge"]["metrics"],
            {
                "gold_precision": 0.75,
                "recall": 0.75,
                "f1": 0.75,
                "question_quality": 0.833333,
            },
        )
        self.assertEqual(
            [item["score"] for item in result["rule_based"]["matches"]],
            [1.0, 0.5],
        )

    def test_zero_score_in_matches_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid match_score"):
            JUDGE.validate_llm_response(
                {
                    "matches": [
                        {
                            "finding_id": "F001",
                            "condition_id": "kc_001",
                            "match_score": 0.0,
                            "question_score": 0.0,
                        }
                    ],
                    "unmatched_findings": [{"finding_id": "F002"}],
                },
                finding_ids={"F001", "F002"},
                condition_ids={"kc_001", "kc_002"},
            )


class DuplicateConditionCorrectionTests(unittest.TestCase):
    def test_competing_findings_require_one_match_and_one_unmatched(self) -> None:
        partial = {"finding_id": "F002", "condition_id": "kc_004",
                   "match_score": 0.5, "question_score": 0.5}
        full = {"finding_id": "F003", "condition_id": "kc_004",
                "match_score": 1.0, "question_score": 1.0}
        ids = {"finding_ids": {"F002", "F003"}, "condition_ids": {"kc_004"}}
        with self.assertRaisesRegex(ValueError, "duplicate matched condition") as caught:
            JUDGE.validate_llm_response(
                {"matches": [partial, full], "unmatched_findings": []}, **ids
            )
        correction = JUDGE.judgment_correction_prompt(caught.exception)
        self.assertIn("keep only the best semantic", correction)
        self.assertIn("unmatched_findings", correction)
        corrected = {"matches": [full], "unmatched_findings": [
            {"finding_id": "F002", "reason": "Another finding better covers this condition."}
        ]}
        self.assertEqual(JUDGE.validate_llm_response(corrected, **ids), corrected)


if __name__ == "__main__":
    unittest.main()
