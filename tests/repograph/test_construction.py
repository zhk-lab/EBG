from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ebg.evidence_intake import load_visible_bundle
from repograph.construction import build_repograph, render_symbol_directory
from scripts.main.prepare import (
    build_repograph_artifacts,
    validate_repograph_artifact,
)
from tests.support import ProjectTemporaryDirectory, make_repo_bundle


SOURCE = """class Service:
    def run(self, value):
        return normalize(value)

def normalize(value):
    return str(value)

def public(value):
    return Service().run(value)
"""


class RepoGraphConstructionTests(unittest.TestCase):
    def test_builds_definition_reference_contain_and_invoke_graph(self) -> None:
        with ProjectTemporaryDirectory() as temporary:
            bundle_root = make_repo_bundle(
                temporary,
                repository_files={"src/service.py": SOURCE},
            )
            graph = build_repograph(load_visible_bundle(bundle_root))

        self.assertEqual(
            [item["qualified_name"] for item in graph["symbols"]],
            ["Service", "Service.run", "normalize", "public"],
        )
        references = [node for node in graph["nodes"] if node["kind"] == "ref"]
        self.assertEqual([node["name"] for node in references], ["normalize", "Service", "run"])
        self.assertNotIn("str", [node["name"] for node in references])
        self.assertEqual(
            {edge["relation"] for edge in graph["edges"]},
            {"contain", "invoke"},
        )
        directory = render_symbol_directory(graph)
        self.assertIn("S0002  Service.run", directory)
        self.assertIn("S0003  normalize", directory)

    def test_artifact_build_is_byte_stable_and_detects_source_drift(self) -> None:
        with ProjectTemporaryDirectory() as temporary:
            bundle_root = make_repo_bundle(
                temporary,
                repository_files={"src/service.py": SOURCE},
            )
            output = temporary / "repograph"
            first = build_repograph_artifacts(bundle_root, output)
            before = {path.name: path.read_bytes() for path in output.iterdir()}
            second = build_repograph_artifacts(bundle_root, output)
            after = {path.name: path.read_bytes() for path in output.iterdir()}

            self.assertTrue(first["changed"])
            self.assertEqual(second["changed"], [])
            self.assertEqual(before, after)
            self.assertTrue(validate_repograph_artifact(bundle_root, output)["valid"])
            manifest = json.loads((output / "build_manifest.json").read_text("utf-8"))
            self.assertEqual(manifest["method"], "RepoGraph")
            self.assertEqual(manifest["retrieval"]["hop_depth"], 1)

            source = bundle_root / "repository" / "src" / "service.py"
            source.write_text(SOURCE.replace("str(value)", "repr(value)"), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "current visible input"):
                validate_repograph_artifact(bundle_root, output)

    def test_python2_print_fallback_preserves_symbol_lines(self) -> None:
        source = "def legacy(value):\n    print value\n    return value\n"
        with ProjectTemporaryDirectory() as temporary:
            bundle_root = make_repo_bundle(
                temporary,
                repository_files={"legacy.py": source},
            )
            graph = build_repograph(load_visible_bundle(bundle_root))

        self.assertEqual(graph["parse_errors"], [])
        symbol = graph["symbols"][0]
        self.assertEqual((symbol["line_start"], symbol["line_end"]), (1, 3))
        definition = next(node for node in graph["nodes"] if node["kind"] == "def")
        self.assertIn("2 |     print value", definition["source"])


if __name__ == "__main__":
    unittest.main()
