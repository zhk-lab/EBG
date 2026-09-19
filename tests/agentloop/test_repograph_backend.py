from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agentloop.evidence import render_tool_result
from agentloop.repograph_backend import RepoGraphBackend
from ebg.evidence_intake import load_visible_bundle
from repograph.construction import build_repograph
from tests.support import ProjectTemporaryDirectory, make_repo_bundle


class RepoGraphBackendTests(unittest.TestCase):
    def test_symbol_read_returns_one_hop_definition_and_reference_context(self) -> None:
        source = """def normalize(value):
    return value.strip()

def handler(raw):
    return normalize(raw)
"""
        with ProjectTemporaryDirectory() as temporary:
            root = make_repo_bundle(
                temporary,
                repository_files={"pkg/api.py": source},
            )
            bundle = load_visible_bundle(root)
            graph = build_repograph(bundle)
            backend = RepoGraphBackend(
                bundle,
                graph,
                index_budget=4096,
                count_tokens=lambda text: len(text.split()),
            )
            search = backend.search("normalize", limit=8)
            self.assertEqual(search.total_matches, 1)
            read_id = search.returned_ids[0]
            result = backend.read(
                [read_id],
                token_budget=4096,
                max_atomic_unit_tokens=8192,
                count_tokens=lambda text: len(text.split()),
            )

        rendered = render_tool_result(result)
        self.assertIn("[[REPOGRAPH EGO GRAPH]]", rendered)
        self.assertIn("Search symbol: normalize", rendered)
        self.assertIn("def normalize(value):", rendered)
        self.assertIn("return normalize(raw)", rendered)
        self.assertEqual(
            {(span.path, span.start, span.end) for span in result.displayed_spans},
            {
                ("pkg/api.py", 1, 2),
                ("pkg/api.py", 4, 5),
                ("pkg/api.py", 5, 5),
            },
        )

    def test_complete_directory_remains_searchable_when_index_is_tiny(self) -> None:
        source = "\n\n".join(
            f"def symbol_{index}():\n    return {index}" for index in range(30)
        )
        with ProjectTemporaryDirectory() as temporary:
            root = make_repo_bundle(
                temporary,
                repository_files={"many.py": source},
            )
            bundle = load_visible_bundle(root)
            backend = RepoGraphBackend(
                bundle,
                build_repograph(bundle),
                index_budget=55,
                count_tokens=lambda text: len(text.split()),
            )

        self.assertLess(len(backend.initial_ids), 30)
        found = backend.search("symbol_29", limit=8)
        self.assertEqual(found.total_matches, 1)
        self.assertEqual(found.hits[0].symbol, "symbol_29")

    def test_read_id_keeps_same_name_definitions_separate(self) -> None:
        with ProjectTemporaryDirectory() as temporary:
            root = make_repo_bundle(
                temporary,
                repository_files={
                    "first.py": "def shared():\n    return 'first'\n",
                    "second.py": "def shared():\n    return 'second'\n",
                },
            )
            bundle = load_visible_bundle(root)
            backend = RepoGraphBackend(
                bundle,
                build_repograph(bundle),
                index_budget=4096,
                count_tokens=lambda text: len(text.split()),
            )
            hits = backend.search("shared", limit=8)
            result = backend.read(
                [hits.returned_ids[0]],
                token_budget=4096,
                max_atomic_unit_tokens=8192,
                count_tokens=lambda text: len(text.split()),
            )

        rendered = render_tool_result(result)
        self.assertIn("return 'first'", rendered)
        self.assertNotIn("return 'second'", rendered)


if __name__ == "__main__":
    unittest.main()
