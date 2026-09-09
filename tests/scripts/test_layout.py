from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts.main.layout import migrate_experiment, sample_directory, update_summary
from tests.support import ProjectTemporaryDirectory


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class ExperimentLayoutTests(unittest.TestCase):
    def test_summary_updates_preserve_prediction_and_each_judge(self) -> None:
        with ProjectTemporaryDirectory() as root:
            update_summary(root, prediction={"totals": {"samples": 2}})
            update_summary(root, judge_model="qwen", judgment={"totals": {"completed": 1}})
            update_summary(root, judge_model="other", judgment={"totals": {"completed": 2}})
            update_summary(root, prediction={"totals": {"samples": 3}})
            saved = json.loads((root / "summary.json").read_text())
            self.assertEqual(saved["prediction"]["totals"]["samples"], 3)
            self.assertEqual(set(saved["judges"]), {"qwen", "other"})
            self.assertEqual(len(list(root.rglob("summary.json"))), 1)

    def test_migration_preserves_records_and_is_resumable(self) -> None:
        with ProjectTemporaryDirectory() as root:
            write(root / "manifest.json", {
                "arms": ["graph"], "phase": "full", "selected_ids": {"specgap": ["sg_001"]},
            })
            old_prediction = root / "runs/full/specgap/graph/sg_001"
            old_judge = root / "judges/qwen/specgap/graph/sg_001"
            prediction = {"findings": [], "input_id": "sg_001"}
            write(old_prediction / "prediction.json", prediction)
            write(old_prediction / "attempt_1/state.json", {"status": "complete"})
            write(old_judge / "retry_1/attempts/initial_001.json", {"content": "original response"})
            write(old_judge / "judge_input.json", {"prediction_path": str(old_prediction / "prediction.json"), "messages": []})
            write(root / "summary.json", {"totals": {"samples": 1}})
            write(root / "judges/qwen/summary.json", {"totals": {"completed": 1}})
            migrate_experiment(root)
            self.assertEqual(json.loads((root / "runs/sg_001/prediction.json").read_text()), prediction)
            self.assertTrue((root / "runs/sg_001/attempt_1/state.json").exists())
            self.assertTrue((root / "judges/qwen/sg_001/retry_1/attempts/initial_001.json").exists())
            saved_input = json.loads((root / "judges/qwen/sg_001/judge_input.json").read_text())
            self.assertEqual(saved_input["prediction_path"], str(root / "runs/sg_001/prediction.json"))
            summary = (root / "summary.json").read_bytes()
            migrate_experiment(root)
            self.assertEqual((root / "summary.json").read_bytes(), summary)
            self.assertFalse((root / "runs/full").exists())
            self.assertFalse((root / "judges/qwen/specgap").exists())
            self.assertFalse((root / "judges/qwen/summary.json").exists())

    def test_collision_is_rejected_before_moving_any_sample(self) -> None:
        with ProjectTemporaryDirectory() as root:
            write(root / "manifest.json", {"arms": ["raw"], "selected_ids": {"specgap": ["sg_001", "sg_002"]}})
            for sid in ["sg_001", "sg_002"]:
                write(root / f"runs/raw/{sid}/prediction.json", {"source": "old"})
            write(root / "runs/sg_002/prediction.json", {"source": "new"})
            with self.assertRaisesRegex(ValueError, "collision"):
                migrate_experiment(root)
            self.assertTrue((root / "runs/raw/sg_001/prediction.json").exists())
            self.assertFalse((root / "runs/sg_001").exists())

    def test_multiple_arms_keep_only_the_necessary_disambiguation(self) -> None:
        with ProjectTemporaryDirectory() as root:
            write(root / "manifest.json", {"arms": ["raw", "graph"]})
            self.assertEqual(sample_directory(root, "runs", "raw", "sg_001"), root / "runs/raw/sg_001")
            self.assertEqual(sample_directory(root, "judges/qwen", "graph", "sg_001"), root / "judges/qwen/graph/sg_001")
