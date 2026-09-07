from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.main.predict import (
    BatchConfig,
    BatchExperimentError,
    collect_response_usage,
    load_split,
    run_batch,
)
from tests.support import ProjectTemporaryDirectory


class BatchPredictionTests(unittest.TestCase):
    def test_split_validation_rejects_overlap_and_duplicates(self) -> None:
        with ProjectTemporaryDirectory() as temporary:
            split_path = temporary / "split.json"
            self._write_split(split_path)
            loaded = load_split(split_path)
            self.assertEqual(loaded["split_id"], "test_split")

            value = json.loads(split_path.read_text(encoding="utf-8"))
            value["benchmarks"]["specgap"]["formal"] = ["sg_dev"]
            split_path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(BatchExperimentError):
                load_split(split_path)

            self._write_split(split_path)
            value = json.loads(split_path.read_text(encoding="utf-8"))
            value["benchmarks"]["silentswap"]["development"] = ["ss_dev", "ss_dev"]
            split_path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(BatchExperimentError):
                load_split(split_path)

    def test_manifest_prevents_configuration_drift(self) -> None:
        with ProjectTemporaryDirectory() as temporary:
            split_path = temporary / "split.json"
            self._write_split(split_path)
            config = self._config(temporary, split_path)
            run_batch(config, run_one=lambda args: self._complete(args), report=lambda _: None)
            run_batch(config, run_one=lambda args: self._complete(args), report=lambda _: None)
            manifest = json.loads(
                (config.output_root / "manifest.json").read_text(encoding="utf-8")
            )

            self.assertEqual(manifest["schema_version"], 4)
            self.assertEqual(
                manifest["prompt_variants"],
                {"raw": "baseline", "graph": "BEG"},
            )
            self.assertEqual(
                manifest["prediction_requests"]["specgap"],
                {"reasoning_effort": "none"},
            )

            changed = BatchConfig(
                **{
                    **self._config_values(temporary, split_path),
                    "model": "different-model",
                }
            )
            with self.assertRaises(BatchExperimentError):
                run_batch(
                    changed,
                    run_one=lambda args: self._complete(args),
                    report=lambda _: None,
                )

    def test_response_usage_accepts_both_common_field_names(self) -> None:
        with ProjectTemporaryDirectory() as temporary:
            response_root = temporary / "run" / "responses"
            failure_root = temporary / "run" / "provider_failures"
            response_root.mkdir(parents=True)
            self._write_response(
                response_root / "turn_001.json",
                {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
            )
            self._write_response(
                response_root / "turn_002.json",
                {"input_tokens": 20, "output_tokens": 6},
            )
            self._write_response(
                response_root / "turn_002_format_01.json",
                {"prompt_tokens": 5, "completion_tokens": 2},
            )
            self._write_response(
                failure_root / "turn_002_retry_0.json",
                {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
            )

            usage = collect_response_usage(temporary / "run")

        self.assertEqual(
            usage,
            {"calls": 4, "input_tokens": 42, "output_tokens": 15, "total_tokens": 57},
        )

    def test_reentry_uses_the_same_sample_directory_without_double_counting(self) -> None:
        with ProjectTemporaryDirectory() as temporary:
            split_path = temporary / "split.json"
            self._write_split(split_path)
            config = self._config(temporary, split_path)
            calls: list[Path] = []

            def resumable(args):
                calls.append(args.output)
                response = args.output / "responses" / "turn_001.json"
                if not response.exists():
                    self._write_response(
                        response,
                        {"prompt_tokens": 7, "completion_tokens": 3},
                    )
                return {"status": "complete", "turns": 1, "failure": None}

            first = run_batch(config, run_one=resumable, report=lambda _: None)
            second = run_batch(config, run_one=resumable, report=lambda _: None)

        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0], calls[1])
        self.assertEqual(first, second)
        self.assertEqual(second["totals"]["usage"]["calls"], 1)
        self.assertEqual(second["totals"]["usage"]["total_tokens"], 10)

    @staticmethod
    def _write_split(path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "split_id": "test_split",
                    "selection": {"development_count": 1, "formal_count": 1},
                    "benchmarks": {
                        "specgap": {
                            "development": ["sg_dev"],
                            "formal": ["sg_formal"],
                        },
                        "silentswap": {
                            "development": ["ss_dev"],
                            "formal": ["ss_formal"],
                        },
                        "feedbacktrace": {
                            "development": ["ft_dev"],
                            "formal": ["ft_formal"],
                        },
                    },
                }
            ),
            encoding="utf-8",
            newline="",
        )

    @staticmethod
    def _write_response(path: Path, usage: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"usage": usage}),
            encoding="utf-8",
            newline="",
        )

    @staticmethod
    def _complete(args) -> dict:
        return {"status": "complete", "turns": 1, "failure": None}

    @staticmethod
    def _config_values(root: Path, split_path: Path) -> dict:
        return {
            "experiment_name": "pilot",
            "experiment_root": root / "experiments",
            "split_file": split_path,
            "phase": "development",
            "benchmarks": ("specgap",),
            "arms": ("graph",),
            "model": "gpt-test",
            "base_url": "https://example.test/v1",
            "workers": 1,
            "schema_root": PROJECT_ROOT / "schemas",
            "request_options": {"reasoning_effort": "none"},
        }

    @classmethod
    def _config(cls, root: Path, split_path: Path) -> BatchConfig:
        return BatchConfig(**cls._config_values(root, split_path))


if __name__ == "__main__":
    unittest.main()
