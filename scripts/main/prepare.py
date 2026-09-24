"""Build and validate main experiment artifacts."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import tiktoken
from agentloop.config import DEFAULT_CONFIG
from agentloop.context import FastTokenCounter
from agentloop.evidence import render_tool_result
from agentloop.graph_backend import GraphBackend
from agentloop.raw_backend import RawBackend
from agentloop.repograph_backend import RepoGraphBackend
from ebg.behavior_atomization import build_behaviors, validate_behaviors
from ebg.behavior_directory import (
    DIRECTORY_ENCODING,
    build_ranked_directory,
    render_ranked_directory,
    validate_ranked_directory,
)
from ebg.core.errors import DirectoryError, GraphError
from ebg.evidence_intake import build_evidence, load_visible_bundle, validate_evidence
from ebg.graph_assembly import build_graph, validate_graph
from ebg.local_graph_retrieval import LocalGraphRetriever
from ebg.relation_linking import build_edges, validate_edges
from jsonschema import Draft202012Validator
from repograph.construction import (
    RepoGraphError,
    build_repograph,
    render_symbol_directory,
    validate_repograph,
)


def canonical_json_bytes(value: Any, *, pretty: bool = True) -> bytes:
    separators = None if pretty else (",", ":")
    text = json.dumps(
        value,
        ensure_ascii=False,
        indent=2 if pretty else None,
        separators=separators,
    )
    return (text + "\n").encode("utf-8")


def canonical_jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    lines = [
        json.dumps(row, ensure_ascii=False, separators=(",", ":"))
        for row in rows
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def atomic_write(path: Path, content: bytes) -> bool:
    """Write only when bytes differ and never expose a partial artifact."""

    if path.is_file() and path.read_bytes() == content:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(content)
    for attempt in range(6):
        try:
            os.replace(temporary, path)
            return True
        except PermissionError:
            if attempt == 5:
                temporary.unlink(missing_ok=True)
                raise
            time.sleep(0.2)
    return True


def write_json(path: Path, value: Any) -> bool:
    return atomic_write(path, canonical_json_bytes(value))


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> bool:
    return atomic_write(path, canonical_jsonl_bytes(rows))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row {line_number} must be an object")
            rows.append(value)
    return rows


REPO_SCHEMA_VERSION = 3

TRACE_SCHEMA_VERSION = 4

REPOGRAPH_OUTPUT_FILES = (
    "repograph.json",
    "symbol_directory.txt",
    "build_manifest.json",
)

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
        "method": "EBG",
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


def build_repograph_artifacts(
    bundle_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Build the independent RepoGraph baseline artifacts for one repo sample."""

    bundle = load_visible_bundle(bundle_root)
    graph = build_repograph(bundle)
    output = Path(output_root)
    directory = render_symbol_directory(graph)
    counts = {
        "definitions": len(graph["symbols"]),
        "references": sum(node["kind"] == "ref" for node in graph["nodes"]),
        "nodes": len(graph["nodes"]),
        "contain_edges": sum(edge["relation"] == "contain" for edge in graph["edges"]),
        "invoke_edges": sum(edge["relation"] == "invoke" for edge in graph["edges"]),
        "parse_errors": len(graph["parse_errors"]),
    }
    manifest = {
        "schema_version": 1,
        "method": "RepoGraph",
        "input_id": bundle.input_id,
        "benchmark": bundle.benchmark,
        "retrieval": {"strategy": "symbol_ego_graph", "hop_depth": 1},
        "counts": counts,
        "outputs": ["repograph.json", "symbol_directory.txt"],
    }
    changed: list[str] = []
    outputs = {
        "repograph.json": canonical_json_bytes(graph),
        "symbol_directory.txt": directory.encode("utf-8"),
        "build_manifest.json": canonical_json_bytes(manifest),
    }
    for filename, content in outputs.items():
        if atomic_write(output / filename, content):
            changed.append(filename)
    validate_repograph_artifact(bundle.root, output, expected=graph)
    return {
        "input_id": bundle.input_id,
        "benchmark": bundle.benchmark,
        "output": str(output.resolve()),
        "changed": changed,
        "counts": counts,
        "valid": True,
    }


def validate_repograph_artifact(
    bundle_root: str | Path,
    output_root: str | Path,
    *,
    expected: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate cached RepoGraph output against the current visible bundle."""

    bundle = load_visible_bundle(bundle_root)
    output = Path(output_root)
    missing = [name for name in REPOGRAPH_OUTPUT_FILES if not (output / name).is_file()]
    if missing:
        raise RepoGraphError(f"missing generated RepoGraph artifact: {missing[0]}")
    graph = load_json(output / "repograph.json")
    manifest = load_json(output / "build_manifest.json")
    if not isinstance(graph, dict) or not isinstance(manifest, dict):
        raise RepoGraphError("RepoGraph artifacts must contain JSON objects")
    current = expected if expected is not None else build_repograph(bundle)
    if graph != current:
        raise RepoGraphError("repograph.json does not match the current visible input")
    validate_repograph(bundle, graph)
    expected_directory = render_symbol_directory(graph).encode("utf-8")
    if (output / "symbol_directory.txt").read_bytes() != expected_directory:
        raise RepoGraphError("symbol_directory.txt does not match repograph.json")
    counts = {
        "definitions": len(graph["symbols"]),
        "references": sum(node["kind"] == "ref" for node in graph["nodes"]),
        "nodes": len(graph["nodes"]),
        "contain_edges": sum(edge["relation"] == "contain" for edge in graph["edges"]),
        "invoke_edges": sum(edge["relation"] == "invoke" for edge in graph["edges"]),
        "parse_errors": len(graph["parse_errors"]),
    }
    expected_manifest = {
        "schema_version": 1,
        "method": "RepoGraph",
        "input_id": bundle.input_id,
        "benchmark": bundle.benchmark,
        "retrieval": {"strategy": "symbol_ego_graph", "hop_depth": 1},
        "counts": counts,
        "outputs": ["repograph.json", "symbol_directory.txt"],
    }
    if manifest != expected_manifest:
        raise RepoGraphError("RepoGraph build manifest is invalid")
    return {
        "input_id": bundle.input_id,
        "benchmark": bundle.benchmark,
        "definitions": counts["definitions"],
        "edges": counts["contain_edges"] + counts["invoke_edges"],
        "valid": True,
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
        "method": "EBG",
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


JSON_OUTPUT = "ranked_directory.json"

TEXT_OUTPUT = "ranked_directory.txt"


def token_counter(encoding_name: str = DIRECTORY_ENCODING) -> Callable[[str], int]:
    try:
        encoding = tiktoken.get_encoding(encoding_name)
    except ValueError as error:
        raise DirectoryError(f"unknown tiktoken encoding: {encoding_name}") from error

    def count(text: str) -> int:
        return len(encoding.encode(text, disallowed_special=()))

    return count


def build_directory_artifact(
    bundle_root: str | Path,
    graph_source: str | Path,
    output_root: str | Path,
    *,
    encoding_name: str = DIRECTORY_ENCODING,
) -> dict[str, Any]:
    bundle = load_visible_bundle(bundle_root)
    graph = _load_graph(graph_source)
    directory = build_ranked_directory(
        bundle, graph, count_tokens=token_counter(encoding_name)
    )
    rendered = render_ranked_directory(directory)
    output = Path(output_root)
    changed: list[str] = []
    if atomic_write(output / JSON_OUTPUT, canonical_json_bytes(directory)):
        changed.append(JSON_OUTPUT)
    if atomic_write(output / TEXT_OUTPUT, rendered.encode("utf-8")):
        changed.append(TEXT_OUTPUT)
    validate_directory_artifact(
        bundle.root,
        graph_source,
        output,
        encoding_name=encoding_name,
    )
    return _summary(directory, output, changed)


def validate_directory_artifact(
    bundle_root: str | Path,
    graph_source: str | Path,
    output_root: str | Path,
    *,
    encoding_name: str = DIRECTORY_ENCODING,
) -> dict[str, Any]:
    bundle = load_visible_bundle(bundle_root)
    graph = _load_graph(graph_source)
    output = Path(output_root)
    json_path = output / JSON_OUTPUT
    text_path = output / TEXT_OUTPUT
    if not json_path.is_file() or not text_path.is_file():
        missing = JSON_OUTPUT if not json_path.is_file() else TEXT_OUTPUT
        raise DirectoryError(f"missing generated artifact: {missing}")
    directory = load_json(json_path)
    if not isinstance(directory, dict):
        raise DirectoryError("ranked_directory.json must contain an object")
    counter = token_counter(encoding_name)
    validate_ranked_directory(bundle, graph, directory, count_tokens=counter)
    expected_text = render_ranked_directory(directory).encode("utf-8")
    if text_path.read_bytes() != expected_text:
        raise DirectoryError("ranked_directory.txt does not match the structured index")
    return _summary(directory, output, [])


def _load_graph(source: str | Path) -> dict[str, Any]:
    path = Path(source)
    if path.is_dir():
        path = path / "behavior_graph.json"
    if not path.is_file():
        raise DirectoryError(f"missing module-four graph: {path}")
    graph = load_json(path)
    if not isinstance(graph, dict):
        raise DirectoryError("module-four graph must contain an object")
    return graph


def _summary(
    directory: dict[str, Any], output: Path, changed: list[str]
) -> dict[str, Any]:
    return {
        "input_id": directory["input_id"],
        "benchmark": directory["benchmark"],
        "directory_type": directory["directory_type"],
        "entries": len(directory["entries"]),
        "token_count": directory["token_count"],
        "output": str(output.resolve()),
        "changed": changed,
        "valid": True,
    }


def validate_repo_backends(
    graph_root: Path,
    directory_root: Path,
    visible_bundle_root: Path,
    *,
    arm: str,
    repograph_root: Path | None = None,
) -> dict[str, int | float]:
    counter = FastTokenCounter(DEFAULT_CONFIG.token_estimator)
    samples = 0
    listed_units = 0
    oversize_units = 0
    read_token_counts: list[int] = []
    source_root = repograph_root if arm == "repograph" and repograph_root else graph_root
    for graph_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
        input_id = graph_dir.name
        bundle = load_visible_bundle(visible_bundle_root / input_id)
        if arm == "graph":
            graph = load_json(graph_dir / "behavior_graph.json")
            directory = load_json(
                directory_root / input_id / "ranked_directory.json"
            )
            backend = GraphBackend(
                bundle,
                graph,
                directory,
                count_tokens=counter.count_text,
            )
            ids = sorted(backend.initial_ids)
        elif arm == "repograph":
            repograph = load_json(graph_dir / "repograph.json")
            backend = RepoGraphBackend(
                bundle,
                repograph,
                index_budget=DEFAULT_CONFIG.index_budget,
                count_tokens=counter.count_text,
            )
            ids = sorted(str(symbol["read_id"]) for symbol in repograph["symbols"])
        else:
            backend = RawBackend(
                bundle,
                index_budget=DEFAULT_CONFIG.index_budget,
                count_tokens=counter.count_text,
            )
            ids = []
            for artifact in bundle.repo_artifacts:
                exact = backend.search(
                    artifact.path,
                    limit=DEFAULT_CONFIG.max_search_hits,
                    exact_path_limit=DEFAULT_CONFIG.max_exact_path_hits,
                )
                if exact.total_matches != len(exact.returned_ids):
                    raise ValueError(
                        f"{input_id}: exact path search truncated {artifact.path}"
                    )
                ids.extend(exact.returned_ids)
            ids = sorted(set(ids))
        for unit_id in ids:
            exact = backend.search(unit_id, limit=DEFAULT_CONFIG.max_search_hits)
            if exact.returned_ids != (unit_id,):
                raise ValueError(f"{input_id}: exact search failed for {unit_id}")
            result = backend.read(
                [unit_id],
                token_budget=DEFAULT_CONFIG.tool_result_budget,
                max_atomic_unit_tokens=DEFAULT_CONFIG.physical_hard_limit,
                count_tokens=counter.count_text,
            )
            read_tokens = counter.count_text(render_tool_result(result))
            read_token_counts.append(read_tokens)
            oversize_units += int(result.oversize_unit)
        listed_units += len(ids)
        samples += 1
    return {
        "samples": samples,
        "listed_units": listed_units,
        "oversize_units": oversize_units,
        **_read_token_metrics(read_token_counts),
    }


def _read_token_metrics(values: list[int]) -> dict[str, int | float]:
    ordered_tokens = sorted(values)
    return {
        "median_read_tokens": (
            statistics.median(ordered_tokens) if ordered_tokens else 0
        ),
        "p95_read_tokens": _nearest_rank(ordered_tokens, 0.95),
        "max_read_tokens": ordered_tokens[-1] if ordered_tokens else 0,
        "read_units_over_32k": sum(value > 32_768 for value in ordered_tokens),
        "read_units_over_64k": sum(value > 65_536 for value in ordered_tokens),
    }


def _nearest_rank(ordered_values: list[int], percentile: float) -> int:
    """Return the nearest-rank percentile for an already sorted population."""

    if not ordered_values:
        return 0
    return ordered_values[math.ceil(percentile * len(ordered_values)) - 1]


def validate_local_graph_sample(
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


def _build_one(task: tuple[str, str, str | None, str | None]) -> dict[str, Any]:
    bundle, graph_output, directory_output, repograph_output = task
    graph = build_graph_artifacts(bundle, graph_output)
    directory_changed: list[str] = []
    repograph_changed: list[str] = []
    if directory_output is not None:
        directory = build_directory_artifact(bundle, graph_output, directory_output)
        directory_changed = directory["changed"]
    if repograph_output is not None:
        repograph = build_repograph_artifacts(bundle, repograph_output)
        repograph_changed = repograph["changed"]
    return {
        "input_id": graph["input_id"],
        "benchmark": graph["benchmark"],
        "graph_changed": graph["changed"],
        "directory_changed": directory_changed,
        "repograph_changed": repograph_changed,
    }


def _validate_one(task: tuple[str, str, str | None, str | None]) -> dict[str, Any]:
    bundle, graph_output, directory_output, repograph_output = task
    graph = validate_output(bundle, graph_output)
    if directory_output is not None:
        validate_directory_artifact(bundle, graph_output, directory_output)
    if repograph_output is not None:
        validate_repograph_artifact(bundle, repograph_output)
    return graph


def _load_checkpoint(path: Path | None) -> set[str]:
    if path is None or not path.is_file():
        return set()
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("invalid artifact checkpoint")
    completed = value.get("completed")
    if not isinstance(completed, list):
        raise ValueError("invalid artifact checkpoint")
    return {str(item) for item in completed}


def _save_checkpoint(path: Path | None, completed: set[str]) -> None:
    if path is None:
        return
    atomic_write(
        path,
        canonical_json_bytes(
            {"schema_version": 1, "completed": sorted(completed)}
        ),
    )


def _all_build_outputs_exist(
    artifact_root: Path,
    input_id: str,
    benchmark: str,
) -> bool:
    """Keep checkpoints resumable when a newer method adds required artifacts."""

    graph_root = artifact_root / "behavior_graphs" / input_id
    graph_files = [*OUTPUT_FILES.values(), "build_manifest.json"]
    if benchmark != "feedbacktrace":
        graph_files.append("task_document.md")
    required = [graph_root / name for name in graph_files]
    if benchmark != "feedbacktrace":
        directory_root = artifact_root / "behavior_directories" / input_id
        required.extend(directory_root / name for name in (JSON_OUTPUT, TEXT_OUTPUT))
        repograph_root = artifact_root / "repographs" / input_id
        required.extend(repograph_root / name for name in REPOGRAPH_OUTPUT_FILES)
    return all(path.is_file() for path in required)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build or validate main experiment artifacts.")
    parser.add_argument("command", choices=("build", "validate"))
    parser.add_argument(
        "--benchmark",
        action="append",
        choices=("specgap", "silentswap", "feedbacktrace"),
    )
    parser.add_argument("--evaluation-root", type=Path, default=PROJECT_ROOT / "data" / "prepared")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--input-id", help="Select one sample within --benchmark")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.input_id and (not args.benchmark or len(args.benchmark) != 1):
        parser.error("--input-id requires exactly one --benchmark")
    if args.command == "validate" and args.checkpoint is not None:
        parser.error("validation always checks current artifacts; --checkpoint is for build")

    benchmarks = args.benchmark or ["specgap", "silentswap", "feedbacktrace"]
    completed = _load_checkpoint(args.checkpoint)
    tasks: list[tuple[str, str, str | None, str | None]] = []
    for benchmark in benchmarks:
        artifact_root = args.evaluation_root / benchmark / "artifacts"
        visible_root = artifact_root / "visible_bundles"
        if args.input_id and not (visible_root / args.input_id).is_dir():
            parser.error(f"sample not found: {args.input_id}")
        for bundle in sorted(path for path in visible_root.iterdir() if path.is_dir()):
            if args.input_id and bundle.name != args.input_id:
                continue
            key = f"{benchmark}/{bundle.name}"
            if key in completed and _all_build_outputs_exist(
                artifact_root, bundle.name, benchmark
            ):
                continue
            completed.discard(key)
            tasks.append(
                (
                    str(bundle),
                    str(artifact_root / "behavior_graphs" / bundle.name),
                    (
                        None
                        if benchmark == "feedbacktrace"
                        else str(
                            artifact_root
                            / "behavior_directories"
                            / bundle.name
                        )
                    ),
                    (
                        None
                        if benchmark == "feedbacktrace"
                        else str(artifact_root / "repographs" / bundle.name)
                    ),
                )
            )

    failures: list[dict[str, str]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        process = _build_one if args.command == "build" else _validate_one
        future_tasks = {executor.submit(process, task): task for task in tasks}
        for future in as_completed(future_tasks):
            task = future_tasks[future]
            input_id = Path(task[0]).name
            benchmark = Path(task[0]).parent.parent.parent.name
            try:
                result = future.result()
            except Exception as error:
                failure = {
                    "input_id": input_id,
                    "error": f"{type(error).__name__}: {error}",
                }
                failures.append(failure)
                print(json.dumps({"event": "failed", **failure}, ensure_ascii=False), flush=True)
                continue
            completed.add(f"{result['benchmark']}/{result['input_id']}")
            _save_checkpoint(args.checkpoint, completed)
            print(
                json.dumps(
                    {
                        "event": "completed",
                        "input_id": result["input_id"],
                        "benchmark": result["benchmark"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    print(
        json.dumps(
            {
                "event": "summary",
                "requested": len(tasks),
                "completed_total": len(completed),
                "failed": len(failures),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
