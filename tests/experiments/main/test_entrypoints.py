from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tests.support import ProjectTemporaryDirectory, make_repo_bundle


class EntrypointTests(unittest.TestCase):
    def test_prepare_build_resume_validate_and_predict_offline(self) -> None:
        with ProjectTemporaryDirectory() as root:
            artifacts = root / "evaluation/specgap/artifacts"
            make_repo_bundle(
                artifacts / "visible_bundles",
                input_id="sg_test",
                repository_files={"api.py": "def public():\n    return 1\n"},
                document="The public function returns one.\n",
            )
            common = [
                "--benchmark", "specgap",
                "--input-id", "sg_test",
                "--evaluation-root", str(root / "evaluation"),
                "--workers", "1",
            ]
            build = ["build", *common, "--checkpoint", str(root / "checkpoint.json")]
            first = self._run("prepare", build)
            self.assertEqual(first.returncode, 0, first.stderr + first.stdout)
            second = self._run("prepare", build)
            self.assertEqual(second.returncode, 0, second.stderr + second.stdout)
            self.assertEqual(json.loads(second.stdout.splitlines()[-1])["requested"], 0)
            valid = self._run("prepare", ["validate", *common])
            self.assertEqual(valid.returncode, 0, valid.stderr + valid.stdout)

            for arm in ("raw", "graph"):
                result = self._run("predict", [
                    "--benchmark", "specgap", "--input-id", "sg_test",
                    "--arm", arm, "--artifact-root", str(root / "evaluation"),
                    "--output", str(root / arm), "--prepare-only",
                    "--model", "offline", "--base-url", "http://127.0.0.1:1/v1",
                ])
                self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
                self.assertEqual(json.loads(result.stdout)["status"], "prepared")

            manifest = artifacts / "behavior_graphs/sg_test/build_manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            invalid = self._run("prepare", ["validate", *common])
            self.assertEqual(invalid.returncode, 1)
            self.assertEqual(manifest.read_text(encoding="utf-8"), "{}")

    @staticmethod
    def _run(script: str, arguments: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts/main" / f"{script}.py"), *arguments],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=60,
        )


if __name__ == "__main__":
    unittest.main()
