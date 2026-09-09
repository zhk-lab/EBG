from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from beg.behavior_atomization import build_behaviors
from beg.behavior_directory import build_ranked_directory
from beg.evidence_intake import build_evidence, load_visible_bundle
from beg.graph_assembly import build_graph
from beg.local_graph_retrieval import LocalGraphRetriever
from beg.relation_linking import build_edges


LOCAL_GRAPH_SCHEMA = json.loads(
    (PROJECT_ROOT / "schemas" / "local_graph.schema.json").read_text(
        encoding="utf-8"
    )
)
VALIDATE_LOCAL_GRAPH = Draft202012Validator(LOCAL_GRAPH_SCHEMA).validate


class RealSampleSmokeTests(unittest.TestCase):
    def test_one_repo_sample_from_each_benchmark_builds_a_local_graph(self) -> None:
        samples = {"specgap": "sg_001", "silentswap": "ss_001"}
        for benchmark, input_id in samples.items():
            with self.subTest(benchmark=benchmark):
                root = (
                    PROJECT_ROOT
                    / "evaluation"
                    / benchmark
                    / "artifacts"
                    / "visible_bundles"
                    / input_id
                )
                if not root.is_dir():
                    self.skipTest(f"{input_id} is not installed")
                bundle = load_visible_bundle(root)
                evidence = build_evidence(bundle)
                behaviors = build_behaviors(bundle, evidence)
                edges = build_edges(bundle, evidence, behaviors)
                graph = build_graph(bundle, evidence, behaviors, edges)
                directory = build_ranked_directory(bundle, graph)
                retriever = LocalGraphRetriever(bundle, graph, directory)
                indexed = next(
                    item
                    for item in directory["query_index"]
                    if item["root_symbols"]
                )
                local = retriever.read(indexed["read_id"])
                VALIDATE_LOCAL_GRAPH(local)
                self.assertTrue(local["graphs"])
                self.assertTrue(local["graphs"][0]["root"]["behaviors"])


if __name__ == "__main__":
    unittest.main()
