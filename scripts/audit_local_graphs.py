"""Build and validate module-six Local Graphs for visible Repo samples."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Sequence

from jsonschema import Draft202012Validator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from beg.behavior_atomization import build_behaviors
from beg.behavior_directory import build_ranked_directory
from beg.evidence_intake import build_evidence, load_visible_bundle
from beg.graph_assembly import build_graph
from beg.local_graph_retrieval import LocalGraphRetriever
from beg.relation_linking import build_edges
from scripts.graph_io import atomic_write, canonical_json_bytes


def audit(
    visible_roots: Sequence[Path],
    *,
    checkpoint: Path | None = None,
) -> dict[str, Any]:
    validator = Draft202012Validator(
        json.loads(
            (PROJECT_ROOT / "schemas" / "local_graph.schema.json").read_text(
                encoding="utf-8"
            )
        )
    )
    completed = _load_checkpoint(checkpoint)
    for root in visible_roots:
        for bundle_root in sorted(path for path in root.iterdir() if path.is_dir()):
            key = f"{root.parent.parent.name}/{bundle_root.name}"
            if key in completed:
                continue
            completed[key] = _audit_sample(bundle_root, validator)
            if checkpoint is not None:
                _save_checkpoint(checkpoint, completed)
    samples = [completed[key] for key in sorted(completed)]
    return {
        "samples": len(samples),
        "files": sum(item["files"] for item in samples),
        "readable_files": sum(item["readable_files"] for item in samples),
        "local_graphs": sum(item["local_graphs"] for item in samples),
        "paths": sum(item["paths"] for item in samples),
        "node_references": sum(item["node_references"] for item in samples),
        "average_local_graphs_per_sample": _mean(samples, "local_graphs"),
        "average_paths_per_sample": _mean(samples, "paths"),
        "median_paths_per_sample": (
            statistics.median(item["paths"] for item in samples) if samples else 0
        ),
    }


def _audit_sample(
    bundle_root: Path,
    validator: Draft202012Validator,
) -> dict[str, int]:
    bundle = load_visible_bundle(bundle_root)
    if bundle.benchmark not in {"specgap", "silentswap"}:
        raise ValueError(f"module six does not accept {bundle.benchmark}")
    evidence = build_evidence(bundle)
    behaviors = build_behaviors(bundle, evidence)
    edges = build_edges(bundle, evidence, behaviors)
    graph = build_graph(bundle, evidence, behaviors, edges)
    directory = build_ranked_directory(bundle, graph)
    retriever = LocalGraphRetriever(bundle, graph, directory)
    totals = {
        "files": len(directory["query_index"]),
        "readable_files": 0,
        "local_graphs": 0,
        "paths": 0,
        "node_references": 0,
    }
    for item in directory["query_index"]:
        local = retriever.read(str(item["read_id"]))
        validator.validate(local)
        if local["graphs"]:
            totals["readable_files"] += 1
        totals["local_graphs"] += len(local["graphs"])
        for graph_item in local["graphs"]:
            totals["paths"] += len(graph_item["paths"])
            totals["node_references"] += sum(
                "ref" in step["node"]
                for path in graph_item["paths"]
                for step in path["steps"]
            )
    return totals


def _load_checkpoint(path: Path | None) -> dict[str, dict[str, int]]:
    if path is None or not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("invalid module-six audit checkpoint")
    samples = value.get("samples")
    if not isinstance(samples, dict):
        raise ValueError("invalid module-six audit checkpoint samples")
    return samples


def _save_checkpoint(path: Path, samples: dict[str, dict[str, int]]) -> None:
    atomic_write(
        path,
        canonical_json_bytes({"schema_version": 1, "samples": samples}),
    )


def _mean(samples: list[dict[str, int]], field: str) -> float:
    return round(sum(item[field] for item in samples) / len(samples), 2) if samples else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("visible_roots", nargs="+", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            audit(args.visible_roots, checkpoint=args.checkpoint),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
