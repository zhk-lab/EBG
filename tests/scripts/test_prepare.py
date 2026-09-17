from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import jsonschema

from ebg.core.errors import GraphError
from scripts.main.prepare import (
    _read_token_metrics,
    build_graph_artifacts,
    validate_output,
)
from tests.support import ProjectTemporaryDirectory, make_repo_bundle


class PipelineTests(unittest.TestCase):
    def test_repeated_build_is_byte_stable_and_matches_schemas(self) -> None:
        with ProjectTemporaryDirectory() as temporary:
            bundle = make_repo_bundle(
                temporary,
                repository_files={
                    "src/service.py": (
                        "def normalize(value):\n"
                        "    if value is None:\n"
                        "        raise ValueError('missing')\n"
                        "    return str(value)\n"
                    )
                },
            )
            output = temporary / "output"
            first = build_graph_artifacts(bundle, output)
            before = {
                path.name: path.read_bytes()
                for path in output.iterdir()
                if path.is_file()
            }
            second = build_graph_artifacts(bundle, output)
            after = {
                path.name: path.read_bytes()
                for path in output.iterdir()
                if path.is_file()
            }
            self.assertTrue(first["changed"])
            self.assertEqual(second["changed"], [])
            self.assertEqual(before, after)
            self.assertTrue(validate_output(bundle, output)["valid"])

            graph = json.loads((output / "behavior_graph.json").read_text(encoding="utf-8"))
            artifacts = {
                "evidence.schema.json": [
                    json.loads(line)
                    for line in (output / "l3_evidence.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                ],
                "behaviors.schema.json": json.loads(
                    (output / "atomic_behaviors.json").read_text(encoding="utf-8")
                ),
                "edges.schema.json": json.loads(
                    (output / "behavior_edges.json").read_text(encoding="utf-8")
                ),
                "graph.schema.json": graph,
                "build_manifest.schema.json": json.loads(
                    (output / "build_manifest.json").read_text(encoding="utf-8")
                ),
            }
            for schema_name, artifact in artifacts.items():
                schema = json.loads(
                    (PROJECT_ROOT / "schemas" / schema_name).read_text(
                        encoding="utf-8"
                    )
                )
                jsonschema.Draft202012Validator.check_schema(schema)
                validator = jsonschema.Draft202012Validator(schema)
                if schema_name == "evidence.schema.json":
                    for node in artifact:
                        validator.validate(node)
                else:
                    validator.validate(artifact)
            serialized_graph = json.dumps(graph, ensure_ascii=False)
            self.assertNotIn("Complete task document", serialized_graph)
            self.assertEqual(
                (output / "task_document.md").read_text(encoding="utf-8"),
                "# Complete task document\n\nRequired behavior.\n",
            )

            source_path = bundle / "repository" / "src" / "service.py"
            source_path.write_text(
                source_path.read_text(encoding="utf-8").replace(
                    "return str(value)", "return repr(value)"
                ),
                encoding="utf-8",
                newline="",
            )
            with self.assertRaisesRegex(GraphError, "current visible input"):
                validate_output(bundle, output)

    def test_overloads_are_filtered_and_conditional_definitions_get_valid_ranges(self) -> None:
        source = """from typing import overload

@overload
def retry(value: int) -> int: ...

def retry(value):
    return value

if FEATURE_ENABLED:
    def normalize(value): return str(value)
"""
        with ProjectTemporaryDirectory() as temporary:
            bundle = make_repo_bundle(
                temporary,
                repository_files={"src/conditional.py": source},
            )
            output = temporary / "output"
            build_graph_artifacts(bundle, output)
            graph = json.loads(
                (output / "behavior_graph.json").read_text(encoding="utf-8")
            )

            self.assertFalse(
                any(item["symbol_lines"] == [4, 4] for item in graph["behaviors"])
            )
            conditional = next(
                item
                for item in graph["source_contexts"]
                if item["symbol"] == "normalize"
            )
            self.assertEqual(conditional["lines"], [10, 10])


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


class ReadBudgetTests(unittest.TestCase):
    def test_read_size_metrics_use_32k_and_64k_boundaries(self) -> None:
        metrics = _read_token_metrics([32_768, 32_769, 65_536, 65_537])

        self.assertEqual(metrics["read_units_over_32k"], 3)
        self.assertEqual(metrics["read_units_over_64k"], 1)
        self.assertNotIn("read_units_over_36k", metrics)


if __name__ == "__main__":
    unittest.main()
