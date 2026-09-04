"""End-to-end deterministic execution of the BEG graph-building stages."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from beg.behavior_atomization import build_behaviors, validate_behaviors
from beg.core.errors import GraphError
from beg.evidence_intake import (
    build_evidence,
    load_visible_bundle,
    validate_evidence,
)
from beg.graph_assembly import (
    build_graph,
    validate_graph,
)
from beg.relation_linking import build_edges, validate_edges

from .graph_io import (
    atomic_write,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    load_json,
    load_jsonl,
)


REPO_SCHEMA_VERSION = 3
TRACE_SCHEMA_VERSION = 4
OUTPUT_FILES = {
    "l3_evidence": "l3_evidence.jsonl",
    "atomic_behaviors": "atomic_behaviors.json",
    "behavior_edges": "behavior_edges.json",
    "behavior_graph": "behavior_graph.json",
}


def build_graph_artifacts(
    bundle_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    bundle = load_visible_bundle(bundle_root)
    output = Path(output_root)

    l3_nodes = build_evidence(bundle)
    atomic_behaviors = build_behaviors(bundle, l3_nodes)
    behavior_edges = build_edges(bundle, l3_nodes, atomic_behaviors)
    graph = build_graph(bundle, l3_nodes, atomic_behaviors, behavior_edges)

    serialized = {
        "l3_evidence": canonical_jsonl_bytes(l3_nodes),
        "atomic_behaviors": canonical_json_bytes(atomic_behaviors),
        "behavior_edges": canonical_json_bytes(behavior_edges),
        "behavior_graph": canonical_json_bytes(graph),
    }
    changed: list[str] = []
    if bundle.task_document is not None:
        if atomic_write(
            output / "task_document.md",
            bundle.task_document.content.encode("utf-8"),
        ):
            changed.append("task_document.md")
    for key, filename in OUTPUT_FILES.items():
        if atomic_write(output / filename, serialized[key]):
            changed.append(filename)

    counts = {
        "visible_repository_files": bundle.visible_repository_files,
        "retained_production_artifacts": len(bundle.repo_artifacts),
        "excluded_repository_files": bundle.excluded_repository_files,
        "l3_facts": len(l3_nodes),
        "behaviors": len(atomic_behaviors),
        "edges": len(behavior_edges),
        "symbols": len(graph.get("symbols", [])),
        "graph_edges": len(graph["edges"]),
        "trace_events": len(bundle.trace_events),
    }
    manifest = {
        "schema_version": _schema_version(bundle.benchmark),
        "method": "BEG",
        "input_id": bundle.input_id,
        "benchmark": bundle.benchmark,
        "counts": counts,
        "outputs": [
            *(["task_document.md"] if bundle.task_document is not None else []),
            *OUTPUT_FILES.values(),
        ],
    }
    if atomic_write(output / "build_manifest.json", canonical_json_bytes(manifest)):
        changed.append("build_manifest.json")

    validate_output(
        bundle.root,
        output,
        expected=(l3_nodes, atomic_behaviors, behavior_edges, graph),
    )
    return {
        "input_id": bundle.input_id,
        "benchmark": bundle.benchmark,
        "output": str(output.resolve()),
        "changed": changed,
        "counts": counts,
    }


def validate_output(
    bundle_root: str | Path,
    output_root: str | Path,
    *,
    expected: tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
        dict[str, Any],
    ]
    | None = None,
) -> dict[str, Any]:
    bundle = load_visible_bundle(bundle_root)
    output = Path(output_root)
    required = [*OUTPUT_FILES.values(), "build_manifest.json"]
    if bundle.task_document is not None:
        required.append("task_document.md")
    missing = [filename for filename in required if not (output / filename).is_file()]
    if missing:
        raise GraphError(f"missing generated artifact: {missing[0]}")

    l3_nodes = load_jsonl(output / OUTPUT_FILES["l3_evidence"])
    atomic_behaviors = load_json(output / OUTPUT_FILES["atomic_behaviors"])
    behavior_edges = load_json(output / OUTPUT_FILES["behavior_edges"])
    graph = load_json(output / OUTPUT_FILES["behavior_graph"])
    manifest = load_json(output / "build_manifest.json")
    if not isinstance(atomic_behaviors, list) or not isinstance(behavior_edges, list) or not isinstance(graph, dict):
        raise GraphError("generated artifacts have invalid top-level JSON types")

    if expected is None:
        expected_l3 = build_evidence(bundle)
        expected_behaviors = build_behaviors(bundle, expected_l3)
        expected_edges = build_edges(
            bundle, expected_l3, expected_behaviors
        )
        expected_graph = build_graph(
            bundle, expected_l3, expected_behaviors, expected_edges
        )
    else:
        expected_l3, expected_behaviors, expected_edges, expected_graph = expected
    if l3_nodes != expected_l3:
        raise GraphError("l3_evidence does not match the current visible input")
    if atomic_behaviors != expected_behaviors:
        raise GraphError("atomic_behaviors does not match the current L3 evidence")
    if behavior_edges != expected_edges:
        raise GraphError("behavior_edges does not match the current behaviors")
    if graph != expected_graph:
        raise GraphError("behavior_graph does not match the upstream graph stages")

    validate_evidence(bundle, l3_nodes)
    validate_behaviors(bundle, l3_nodes, atomic_behaviors)
    validate_edges(bundle, l3_nodes, atomic_behaviors, behavior_edges)
    validate_graph(
        bundle,
        l3_nodes,
        atomic_behaviors,
        behavior_edges,
        graph,
    )

    expected_counts = {
        "visible_repository_files": bundle.visible_repository_files,
        "retained_production_artifacts": len(bundle.repo_artifacts),
        "excluded_repository_files": bundle.excluded_repository_files,
        "l3_facts": len(l3_nodes),
        "behaviors": len(atomic_behaviors),
        "edges": len(behavior_edges),
        "symbols": len(graph.get("symbols", [])),
        "graph_edges": len(graph["edges"]),
        "trace_events": len(bundle.trace_events),
    }
    expected_outputs = [
        *(["task_document.md"] if bundle.task_document is not None else []),
        *OUTPUT_FILES.values(),
    ]
    expected_manifest = {
        "schema_version": _schema_version(bundle.benchmark),
        "method": "BEG",
        "input_id": bundle.input_id,
        "benchmark": bundle.benchmark,
        "counts": expected_counts,
        "outputs": expected_outputs,
    }
    if manifest != expected_manifest:
        raise GraphError("build_manifest does not match the generated artifacts")
    if bundle.task_document is not None:
        expected_document = bundle.task_document.content.encode("utf-8")
        if (output / "task_document.md").read_bytes() != expected_document:
            raise GraphError("task document is not preserved exactly once outside the graph")
    elif (output / "task_document.md").exists():
        raise GraphError("FeedbackTrace output must not contain task_document.md")
    return {
        "input_id": bundle.input_id,
        "benchmark": bundle.benchmark,
        "behaviors": len(atomic_behaviors),
        "edges": len(behavior_edges),
        "valid": True,
    }


def _schema_version(benchmark: str) -> int:
    return TRACE_SCHEMA_VERSION if benchmark == "feedbacktrace" else REPO_SCHEMA_VERSION
