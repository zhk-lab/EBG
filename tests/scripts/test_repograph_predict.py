from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.main.predict import _run
from scripts.main.prepare import build_repograph_artifacts
from tests.support import ProjectTemporaryDirectory, make_repo_bundle


class RepoGraphPredictTests(unittest.TestCase):
    def test_prepare_only_loads_repograph_backend_and_prompt(self) -> None:
        with ProjectTemporaryDirectory() as temporary:
            artifact_root = temporary / "artifacts"
            bundle = make_repo_bundle(
                artifact_root / "visible_bundles",
                input_id="sg_repograph",
                benchmark="specgap",
                repository_files={
                    "pkg/api.py": (
                        "def normalize(value):\n"
                        "    return value.strip()\n\n"
                        "def public(value):\n"
                        "    return normalize(value)\n"
                    )
                },
                document="The public API normalizes its input.\n",
            )
            build_repograph_artifacts(
                bundle,
                artifact_root / "repographs" / "sg_repograph",
            )
            run_root = temporary / "run"
            args = SimpleNamespace(
                benchmark="specgap",
                input_id="sg_repograph",
                arm="repograph",
                artifact_root=artifact_root,
                output=run_root,
                schema_root=PROJECT_ROOT / "schemas",
                base_url="",
                model="",
                api_key_env="EBG_TEST_MISSING_KEY",
                timeout=30.0,
                prepare_only=True,
                request_options={},
            )

            result = _run(args)
            request = json.loads(
                (run_root / "requests" / "turn_001_retry_0.json").read_text("utf-8")
            )

        rendered = "\n".join(message["content"] for message in request["messages"])
        self.assertEqual(result["status"], "prepared")
        self.assertEqual(result["prompt_variant"], "RepoGraph")
        self.assertIn("[REPOGRAPH SYMBOLS]", rendered)
        self.assertIn("S0001", rendered)
        self.assertIn("RepoGraph `search_repo(symbol)`", rendered)
        self.assertNotIn("Each R Read ID", rendered)
        self.assertIn("Read accepts 1 to 32 distinct known IDs", rendered)
        self.assertNotIn("Read accepts 1 to 6 distinct known IDs", rendered)


if __name__ == "__main__":
    unittest.main()
