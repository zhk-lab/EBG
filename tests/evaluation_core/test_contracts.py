from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluation_core.contracts import (
    EvidenceSpan,
    RepoPredictionContract,
    PredictionFormatError,
    PredictionGroundingError,
    load_prediction_schema,
    model_prediction_schema,
    validate_repo_prediction,
)


class PredictionTests(unittest.TestCase):
    def test_feedbacktrace_model_schema_omits_runner_owned_verdict(self) -> None:
        schema = load_prediction_schema(PROJECT_ROOT / "schemas", "feedbacktrace")
        visible = model_prediction_schema(schema)

        self.assertNotIn("input_id", visible["properties"])
        self.assertNotIn("benchmark", visible["properties"])
        self.assertNotIn("verdict", visible["properties"])
        self.assertNotIn("verdict", visible["required"])

    def test_specgap_does_not_add_finding_id_uniqueness_constraint(self) -> None:
        schema = load_prediction_schema(PROJECT_ROOT / "schemas", "specgap")
        finding = {
            "finding_id": "F001",
            "claim": "The implementation returns the public value.",
            "verification_question": "Should this value be returned?",
            "why_important": "Callers observe it.",
            "downstream_impact": "It changes the API result.",
            "code_evidence": [
                {
                    "path": "pkg/api.py",
                    "start_line": 1,
                    "end_line": 1,
                    "symbol": "public",
                    "explanation": "The line returns the value.",
                }
            ],
        }

        prediction = validate_repo_prediction(
            {"findings": [finding, dict(finding)]},
            input_id="sg_schema",
            benchmark="specgap",
            schema=schema,
            observed_spans=[EvidenceSpan("pkg/api.py", "public", 1, 1)],
        )

        self.assertEqual(len(prediction["findings"]), 2)

    def test_silentswap_formal_schema_accepts_five_grounded_swaps(self) -> None:
        schema = load_prediction_schema(PROJECT_ROOT / "schemas", "silentswap")
        visible = model_prediction_schema(schema)
        self.assertNotIn("input_id", visible["properties"])
        self.assertNotIn("benchmark", visible["properties"])
        swaps = []
        for index in range(5):
            swaps.append(
                {
                    "target": {
                        "file": "pkg/api.py",
                        "symbol": {
                            "kind": "function",
                            "qualified_name": ["public"],
                        },
                        "line_ranges": [{"start": index + 1, "end": index + 1}],
                    },
                    "swap_type": "parsing_matching",
                    "code_change": f"change {index}",
                    "trigger_condition": f"input {index}",
                    "behavioral_effect": {
                        "before": f"before {index}",
                        "after": f"after {index}",
                    },
                }
            )
        swaps[0]["target"]["file"] = "pkg\\api.py"
        swaps[0]["target"]["symbol"]["kind"] = "property"
        swaps[1]["target"]["symbol"] = {
            "kind": "module",
            "qualified_name": "",
        }

        prediction = validate_repo_prediction(
            {"swaps": swaps},
            input_id="ss_schema",
            benchmark="silentswap",
            schema=schema,
            observed_spans=[EvidenceSpan("pkg/api.py", "public", 1, 5)],
        )

        self.assertEqual(prediction["benchmark"], "silentswap")
        self.assertEqual(len(prediction["swaps"]), 5)
        self.assertEqual(prediction["swaps"][0]["target"]["file"], "pkg/api.py")
        self.assertEqual(
            prediction["swaps"][0]["target"]["symbol"]["kind"],
            "method",
        )
        self.assertEqual(
            prediction["swaps"][1]["target"]["symbol"]["qualified_name"],
            [],
        )

        with self.assertRaises(PredictionFormatError):
            validate_repo_prediction(
                {"input_id": "wrong", "swaps": swaps},
                input_id="ss_schema",
                benchmark="silentswap",
                schema=schema,
                observed_spans=[EvidenceSpan("pkg/api.py", "public", 1, 5)],
            )

    def test_schema_and_grounding_failures_are_distinct(self) -> None:
        schema = load_prediction_schema(PROJECT_ROOT / "schemas", "specgap")
        with self.assertRaises(PredictionFormatError):
            validate_repo_prediction(
                {},
                input_id="sg_schema",
                benchmark="specgap",
                schema=schema,
                observed_spans=[],
            )

        finding = {
            "finding_id": "F001",
            "claim": "A public behavior exists.",
            "verification_question": "Should it be documented?",
            "why_important": "Callers observe it.",
            "downstream_impact": "It affects callers.",
            "code_evidence": [
                {
                    "path": "pkg/api.py",
                    "start_line": 1,
                    "end_line": 1,
                    "symbol": "public",
                    "explanation": "The line returns a value.",
                }
            ],
        }
        with self.assertRaises(PredictionGroundingError):
            validate_repo_prediction(
                {"findings": [finding]},
                input_id="sg_schema",
                benchmark="specgap",
                schema=schema,
                observed_spans=[],
            )

        reversed_range = {
            **finding,
            "code_evidence": [
                {
                    **finding["code_evidence"][0],
                    "start_line": 5,
                    "end_line": 1,
                }
            ],
        }
        with self.assertRaises(PredictionFormatError):
            validate_repo_prediction(
                {"findings": [reversed_range]},
                input_id="sg_schema",
                benchmark="specgap",
                schema=schema,
                observed_spans=[EvidenceSpan("pkg/api.py", "public", 1, 5)],
            )

    def test_citations_bridge_only_verified_blank_lines(self) -> None:
        schema = load_prediction_schema(PROJECT_ROOT / "schemas", "specgap")
        finding = {
            "finding_id": "F001", "claim": "Two functions return values.",
            "verification_question": "Are both return values documented?",
            "why_important": "Callers observe them.", "downstream_impact": "Affects callers.",
            "code_evidence": [{
                "path": "pkg/api.py", "start_line": 1, "end_line": 5,
                "symbol": "first; second", "explanation": "Both functions were read.",
            }],
        }
        spans = [EvidenceSpan("pkg/api.py", "first", 1, 2), EvidenceSpan("pkg/api.py", "second", 4, 5)]
        for gap in ("", " \t", "hidden_call()", "# hidden comment"):
            with self.subTest(gap=gap):
                contract = RepoPredictionContract("specgap", schema, source_texts={
                    "pkg/api.py": f"def first():\n    return 1\n{gap}\ndef second():\n    return 2\n",
                })
                if not gap.strip():
                    result = contract.validate({"findings": [finding]}, input_id="sg_blank", observed_spans=spans)
                    self.assertEqual(result["findings"], [finding])
                else:
                    with self.assertRaises(PredictionGroundingError):
                        contract.validate({"findings": [finding]}, input_id="sg_blank", observed_spans=spans)
        for sources in (None, {}, {"pkg/api.py": "def first():\n    return 1\n"}):
            with self.subTest(sources=sources):
                contract = RepoPredictionContract("specgap", schema, source_texts=sources)
                with self.assertRaises(PredictionGroundingError):
                    contract.validate({"findings": [finding]}, input_id="sg_blank", observed_spans=spans)

    def test_adjacent_read_chunks_jointly_ground_one_reported_range(self) -> None:
        schema = load_prediction_schema(PROJECT_ROOT / "schemas", "specgap")
        finding = {
            "finding_id": "F001",
            "claim": "A public behavior spans adjacent source chunks.",
            "verification_question": "Should the behavior be documented?",
            "why_important": "Callers observe it.",
            "downstream_impact": "It affects callers.",
            "code_evidence": [
                {
                    "path": "pkg/api.py",
                    "start_line": 98,
                    "end_line": 103,
                    "symbol": "public",
                    "explanation": "Both adjacent chunks were read.",
                }
            ],
        }

        prediction = validate_repo_prediction(
            {"findings": [finding]},
            input_id="sg_schema",
            benchmark="specgap",
            schema=schema,
            observed_spans=[
                EvidenceSpan("pkg/api.py", "<module>", 1, 100),
                EvidenceSpan("pkg/api.py", "<module>", 101, 200),
            ],
        )
        self.assertEqual(prediction["findings"][0]["code_evidence"][0]["end_line"], 103)

        with self.assertRaises(PredictionGroundingError):
            validate_repo_prediction(
                {"findings": [finding]},
                input_id="sg_schema",
                benchmark="specgap",
                schema=schema,
                observed_spans=[
                    EvidenceSpan("pkg/api.py", "<module>", 1, 99),
                    EvidenceSpan("pkg/api.py", "<module>", 101, 200),
                ],
            )


if __name__ == "__main__":
    unittest.main()
