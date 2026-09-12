import copy
import json
import unittest
from unittest.mock import patch

from tests.agentloop.test_silentswap_source_review import ROOT, fixture, answer, directory
from agentloop.benchmark_configs import repo_benchmark_config
from agentloop.silentswap_source_review import build_review
from evaluation_core.contracts import PredictionError, RepoPredictionContract
from evaluation_core.contracts import model_prediction_schema
from agentloop.runner import _finish_format_example
from jsonschema import Draft202012Validator
from tests.judges.test_silentswap import JUDGE


class LocationStageTests(unittest.TestCase):
    def test_finish_repair_example_matches_stage_schema(self):
        for name in ["silentswap_location_prediction.schema.json",
                     "silentswap_prediction.schema.json"]:
            schema = json.loads((ROOT / "schemas" / name).read_text(encoding="utf-8"))
            prediction = json.loads(_finish_format_example("silentswap", schema))["prediction"]
            Draft202012Validator(model_prediction_schema(schema)).validate(prediction)
            expected = set(schema["$defs"]["swap"]["properties"])
            self.assertEqual(set(prediction["swaps"][0]), expected)

    def test_location_stage_accepts_only_five_grounded_targets(self):
        bundle, _, state = fixture()
        targets = {"swaps": [{"target": swap["target"]} for swap in answer()["swaps"]]}
        review = build_review(bundle, {**targets, "input_id": bundle.input_id}, directory(), state=state)
        config = repo_benchmark_config("silentswap")
        bound = config.bind_module7(
            ROOT / "schemas", input_id=bundle.input_id,
            task_document_name=config.task_document_filename,
            task_document=bundle.task_document.content, initial_index="BEG directory",
            source_texts=review.sources,
        )
        bound.finish_contract.validate(targets, input_id=bundle.input_id, observed_spans=review.spans)
        for candidate in [answer(), {"swaps": targets["swaps"][:4]}]:
            with self.assertRaises(PredictionError):
                bound.finish_contract.validate(candidate, input_id=bundle.input_id, observed_spans=review.spans)
        bad = copy.deepcopy(targets)
        bad["swaps"][0]["target"]["line_ranges"] = [{"start": 100, "end": 100}]
        with self.assertRaises(PredictionError):
            bound.finish_contract.validate(bad, input_id=bundle.input_id, observed_spans=review.spans)

        baseline = config.bind_module7(
            ROOT / "schemas", input_id=bundle.input_id,
            task_document_name=config.task_document_filename,
            task_document=bundle.task_document.content, initial_index="source files",
            source_texts=review.sources, prompt_variant="baseline",
        )
        for contract in [baseline.finish_contract,
                         RepoPredictionContract("silentswap", review.schema, source_texts=review.sources)]:
            contract.validate(answer(), input_id=bundle.input_id, observed_spans=review.spans)
            with self.assertRaises(PredictionError):
                contract.validate(targets, input_id=bundle.input_id, observed_spans=review.spans)

    def test_judge_passes_locations_without_inventing_semantics(self):
        targets = {"swaps": [{"target": swap["target"]} for swap in answer()["swaps"]]}
        with patch.object(JUDGE, "build_judge_reference", return_value=[]):
            messages = JUDGE.build_llm_messages({}, targets)
        self.assertIn('"target"', messages[-1]["content"])
        self.assertNotIn('"code_change":', messages[-1]["content"])


if __name__ == "__main__":
    unittest.main()
