from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import jsonschema

from beg.core.errors import GraphError
from scripts.graph_pipeline import build_graph_artifacts, validate_output
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


if __name__ == "__main__":
    unittest.main()
