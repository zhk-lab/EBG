"""Graph Assembly: combine exact upstream facts into one static BEG graph."""

from __future__ import annotations

from typing import Any

from .behavior_atomization import validate_behaviors
from .core.errors import GraphError
from .core.model import VisibleBundle
from .evidence_intake import validate_evidence
from .relation_linking import validate_edges


def build_graph(
    bundle: VisibleBundle,
    evidence: list[dict[str, Any]],
    behaviors: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> dict[str, Any]:
    if bundle.benchmark == "feedbacktrace":
        graph = {
            "input_id": bundle.input_id,
            "benchmark": bundle.benchmark,
            "evidence": evidence,
            "behaviors": behaviors,
            "edges": edges,
        }
    else:
        graph = {
            "input_id": bundle.input_id,
            "benchmark": bundle.benchmark,
            "evidence": evidence,
            "behaviors": behaviors,
            "source_contexts": _build_source_contexts(bundle, behaviors),
            "edges": edges,
        }
    validate_graph(bundle, evidence, behaviors, edges, graph)
    return graph


def validate_graph(
    bundle: VisibleBundle,
    evidence: list[dict[str, Any]],
    behaviors: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    graph: dict[str, Any],
) -> None:
    validate_evidence(bundle, evidence)
    validate_behaviors(bundle, evidence, behaviors)
    validate_edges(bundle, evidence, behaviors, edges)
    if graph.get("input_id") != bundle.input_id:
        raise GraphError("Graph input_id does not match the visible bundle")
    if graph.get("benchmark") != bundle.benchmark:
        raise GraphError("Graph benchmark does not match the visible bundle")
    if graph.get("evidence") != evidence:
        raise GraphError("Graph must preserve the complete Evidence list exactly once")
    if graph.get("edges") != edges:
        raise GraphError("Graph edges differ from Relation Linking output")
    if bundle.benchmark == "feedbacktrace":
        _validate_trace_graph(behaviors, graph)
    else:
        _validate_repo_graph(bundle, behaviors, edges, graph)


def _build_source_contexts(
    bundle: VisibleBundle,
    behaviors: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    artifacts = {artifact.path: artifact for artifact in bundle.repo_artifacts}
    unique: dict[tuple[str, str, str, int, int], dict[str, Any]] = {}
    for behavior in behaviors:
        path = str(behavior["path"])
        symbol = str(behavior["symbol"])
        artifact_kind = str(behavior["artifact_kind"])
        start, end = (int(value) for value in behavior["symbol_lines"])
        artifact = artifacts.get(path)
        if artifact is None:
            raise GraphError(f"Behavior has no retained source artifact: {path}")
        if artifact.kind != artifact_kind:
            raise GraphError("Behavior artifact_kind differs from its retained artifact")
        source_lines = artifact.content.splitlines(keepends=True)
        if not 1 <= start <= end <= len(source_lines):
            raise GraphError("Behavior symbol_lines are outside the retained artifact")
        key = (path, symbol, artifact_kind, start, end)
        unique.setdefault(
            key,
            {
                "artifact_kind": artifact_kind,
                "path": path,
                "symbol": symbol,
                "lines": [start, end],
                "source": "".join(source_lines[start - 1 : end]),
            },
        )
    return [unique[key] for key in sorted(unique, key=_context_sort_key)]


def _context_sort_key(
    key: tuple[str, str, str, int, int],
) -> tuple[Any, ...]:
    path, symbol, artifact_kind, start, end = key
    return (
        path.casefold(),
        path,
        symbol.casefold(),
        symbol,
        start,
        end,
        artifact_kind,
    )


def _validate_repo_graph(
    bundle: VisibleBundle,
    behaviors: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    graph: dict[str, Any],
) -> None:
    if set(graph) != {
        "input_id",
        "benchmark",
        "evidence",
        "behaviors",
        "source_contexts",
        "edges",
    }:
        raise GraphError("Repo graph has unexpected top-level fields")
    if graph.get("behaviors") != behaviors:
        raise GraphError("Repo graph must preserve Behaviors exactly once")
    expected_contexts = _build_source_contexts(bundle, behaviors)
    if graph.get("source_contexts") != expected_contexts:
        raise GraphError("Repo source contexts are incomplete, duplicated, or changed")
    behavior_endpoints = {
        (str(item["path"]), str(item["symbol"])) for item in behaviors
    }
    context_endpoints = {
        (str(item["path"]), str(item["symbol"])) for item in expected_contexts
    }
    if behavior_endpoints != context_endpoints:
        raise GraphError("Every Repo Behavior must have one source context endpoint")
    for edge in edges:
        source = (str(edge["from"]["path"]), str(edge["from"]["symbol"]))
        target = (str(edge["to"]["path"]), str(edge["to"]["symbol"]))
        if source not in behavior_endpoints or target not in behavior_endpoints:
            raise GraphError("Repo edge endpoint has no Behavior")
        if source not in context_endpoints or target not in context_endpoints:
            raise GraphError("Repo edge endpoint has no source context")


def _validate_trace_graph(
    behaviors: list[dict[str, Any]],
    graph: dict[str, Any],
) -> None:
    if set(graph) != {
        "input_id",
        "benchmark",
        "evidence",
        "behaviors",
        "edges",
    }:
        raise GraphError("Trace graph has unexpected top-level fields")
    if graph.get("behaviors") != behaviors:
        raise GraphError("Trace graph must preserve Behaviors exactly once")
