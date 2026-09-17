from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.main.source_review import main
from tests.agentloop.test_runner import ScriptedClient
from tests.agentloop.test_silentswap_source_review import fixture, answer, directory
from tests.support import ProjectTemporaryDirectory


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class SourceReviewCommandTests(unittest.TestCase):
    def test_formal_experiment_judge_uses_stage_directories_and_saved_workers(self):
        with ProjectTemporaryDirectory() as root:
            ebg = root / "experiments/EBG/silentswap/sol"
            judge = "glm-5-2"
            write(ebg / "stage2/manifest.json", {
                "source_experiment": str((ebg / "stage1").resolve()),
                "workflow": "ebg_independent_source_review_v5",
            })
            write(ebg / "stage1/judges" / judge / "manifest.json", {
                "base_url": "http://127.0.0.1:28080/v1", "judge_model": judge, "phase": "full",
            })
            write(ebg / "stage2/judges" / judge / "manifest.json", {"workers": 20})
            with patch("scripts.main.source_review.ROOT", root), \
                 patch("scripts.main.source_review.aggregate", return_value={"status": "complete"}), \
                 patch("scripts.main.source_review.run_batch_judges") as judge_batch, \
                 patch("scripts.main.source_review.run_sample") as predict, patch("builtins.print"):
                self.assertEqual(main(["--experiment", "sol", "--judge", "--judge-model", judge]), 0)
            predict.assert_not_called()
            self.assertEqual(judge_batch.call_args.args[0].output_root, ebg / "stage2")
            self.assertEqual(judge_batch.call_args.args[0].workers, 20)

    def test_explicit_second_stage_model_is_saved_and_used(self):
        with ProjectTemporaryDirectory() as root:
            source, output = root / "source", root / "review"
            write(source / "manifest.json", {
                "model": "old-model", "benchmarks": ["silentswap"], "arms": ["graph"],
                "selected_ids": {"silentswap": ["ss_test"]}, "split_id": "full",
                "base_url": "https://provider.example/v1", "api_key_env": "TEST_API_KEY",
                "prediction_requests": {"silentswap": {}},
            })
            artifacts = root / "evaluation/silentswap/artifacts"
            write(artifacts / "behavior_directories/ss_test/ranked_directory.json", directory())
            bundle, prediction, state = fixture()
            args = ["--source", str(source), "--output", str(output),
                    "--artifact-root", str(artifacts), "--model", "new-model"]
            with patch("scripts.main.source_review.load_visible_bundle", return_value=bundle), \
                 patch("scripts.main.source_review.load_stage1", return_value=(prediction, state, source / "state.json")), \
                 patch("scripts.main.source_review._client", return_value=ScriptedClient([json.dumps(answer())])) as client, \
                 patch("builtins.print"):
                self.assertEqual(main(args), 0)
            saved = json.loads((output / "manifest.json").read_text())
            self.assertEqual(saved["model"], "new-model")
            self.assertEqual(saved["source_model"], "old-model")
            self.assertEqual(client.call_args.args[0].model, "new-model")
            self.assertEqual(json.loads((source / "manifest.json").read_text())["model"], "old-model")

    def test_prepare_predict_resume_and_combine_without_candidate_pairing(self):
        with ProjectTemporaryDirectory() as root:
            source, output = root / "source", root / "review"
            write(source / "manifest.json", {
                "schema_version": 4, "experiment_name": "source", "phase": "full",
                "benchmarks": ["silentswap"], "arms": ["graph"], "split_id": "full",
                "selected_ids": {"silentswap": ["ss_test"]},
                "model": "fake", "base_url": "http://127.0.0.1:28080/v1",
                "api_key_env": "TEST_API_KEY", "prediction_requests": {"silentswap": {}},
            })
            artifacts = root / "evaluation/silentswap/artifacts"
            directory_path = artifacts / "behavior_directories/ss_test/ranked_directory.json"
            write(directory_path, directory())
            args = ["--source", str(source), "--output", str(output), "--artifact-root", str(artifacts)]
            bundle, prediction, state = fixture()
            client = ScriptedClient([json.dumps(answer())])
            with patch("scripts.main.source_review.load_visible_bundle", return_value=bundle), \
                 patch("scripts.main.source_review.load_stage1", return_value=(prediction, state, source / "state.json")), \
                 patch("scripts.main.source_review._client", return_value=client) as make_client, \
                 patch("builtins.print"):
                self.assertEqual(main([*args, "--prepare-only"]), 0)
                make_client.assert_not_called()
                self.assertEqual(main(args), 0)
                self.assertEqual(main(args), 0)
                changed = directory()
                changed["entries"][0], changed["entries"][5] = changed["entries"][5], changed["entries"][0]
                write(directory_path, changed)
                self.assertEqual(main(args), 1)
            self.assertEqual(len(client.calls), 1)
            result = json.loads((output / "runs/ss_test/prediction.json").read_text())
            self.assertEqual(result["swaps"][0]["target"]["file"], "a.py")
            self.assertEqual(prediction["swaps"][0]["target"]["file"], "z.py")
            judge = "qwen3.7-max-2026-06-08"
            first = {"input_id": "ss_test", "benchmark": "silentswap",
                     "rule_based": {"localization_score": 0.8},
                     "llm_judge": {"scores": {"location_correct": 0.5, "code_change_correct": 0}}}
            second = {**first, "rule_based": {"localization_score": 0},
                      "llm_judge": {"scores": {"location_correct": 0, "code_change_correct": 1}}}
            write(source / "judges" / judge / "ss_test/result.json", first)
            with patch("builtins.print"):
                self.assertEqual(main([*args, "--aggregate"]), 1)
            summary_path = output / "combined" / judge / "summary.json"
            self.assertIsNone(json.loads(summary_path.read_text())["score_means"])
            write(output / "judges" / judge / "ss_test/result.json", second)
            with patch("builtins.print"):
                self.assertEqual(main([*args, "--aggregate"]), 0)
            summary = json.loads(summary_path.read_text())
            self.assertEqual(summary["score_means"], {
                "localization_score": 0.8, "location_correct": 0.5, "code_change_correct": 1,
            })
            write(source / "judges" / judge / "manifest.json", {
                "judge_model": judge, "phase": "full", "base_url": "https://judge.example/v1",
                "api_key_env": "SOURCE_JUDGE_KEY", "request_options": {"enable_thinking": False},
                "retry_policy": {"network_retries": 1, "json_format_repairs": 0, "complete_sample_reruns": 0},
            })
            with patch("builtins.print"), patch("scripts.main.source_review.run_batch_judges") as judge_batch:
                self.assertEqual(main([*args, "--judge"]), 0)
            config = judge_batch.call_args.args[0]
            self.assertEqual(config.base_url, "https://judge.example/v1")
            self.assertEqual(config.request_options, {"enable_thinking": False})
            self.assertEqual(config.output_root, output)
            self.assertEqual(config.network_retries, 1)
            self.assertEqual(config.format_repairs, 0)
            self.assertEqual(config.deferred_retries, 0)
            for status in ("complete", "failed"):
                with patch("builtins.print"), \
                     patch("scripts.main.source_review.run_sample", return_value={
                         "input_id": "ss_test", "status": status,
                     }), patch("scripts.main.source_review.run_batch_judges") as judge_batch:
                    self.assertEqual(main([*args, "--then-judge"]), int(status == "failed"))
                    self.assertEqual(judge_batch.call_count, int(status == "complete"))


if __name__ == "__main__":
    unittest.main()
