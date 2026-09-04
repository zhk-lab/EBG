"""Offline Gold audit for module-five ranking and module-six coverage."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


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
    completed = _load_checkpoint(checkpoint)
    requested_keys: set[str] = set()
    for visible_root in visible_roots:
        benchmark = visible_root.parent.parent.name
        gold_root = visible_root.parent / "hidden_gold"
        for bundle_root in sorted(path for path in visible_root.iterdir() if path.is_dir()):
            key = f"{benchmark}/{bundle_root.name}"
            requested_keys.add(key)
            if key in completed:
                continue
            # Gold is deliberately loaded only after every retrieval artifact exists.
            bundle, graph, directory, retriever = _build_retrieval(bundle_root)
            gold = _load_gold(gold_root / f"{bundle.input_id}.json")
            completed[key] = _audit_sample(bundle, graph, directory, retriever, gold)
            if checkpoint is not None:
                _save_checkpoint(checkpoint, completed)
    samples = [completed[key] for key in sorted(requested_keys)]
    return {
        "gold_isolation": (
            "Gold is loaded only after Evidence, Behavior, Edge, Graph, and Ranked "
            "Directory construction; it never affects ranking or Local Graph output."
        ),
        "summary": _summarize(samples),
        "benchmarks": {
            benchmark: _summarize(
                [item for item in samples if item["benchmark"] == benchmark]
            )
            for benchmark in ("specgap", "silentswap")
        },
        "samples": samples,
    }


def _build_retrieval(bundle_root: Path) -> tuple[Any, dict, dict, LocalGraphRetriever]:
    bundle = load_visible_bundle(bundle_root)
    evidence = build_evidence(bundle)
    behaviors = build_behaviors(bundle, evidence)
    edges = build_edges(bundle, evidence, behaviors)
    graph = build_graph(bundle, evidence, behaviors, edges)
    directory = build_ranked_directory(bundle, graph)
    return bundle, graph, directory, LocalGraphRetriever(bundle, graph, directory)


def _audit_sample(
    bundle: Any,
    graph: dict[str, Any],
    directory: dict[str, Any],
    retriever: LocalGraphRetriever,
    gold: dict[str, Any],
) -> dict[str, Any]:
    locations = _gold_locations(gold)
    rank_by_path = {
        str(item["path"]): rank
        for rank, item in enumerate(directory["entries"], start=1)
    }
    index_by_path = {
        str(item["path"]): item for item in directory["query_index"]
    }
    default_nodes_by_path: dict[str, list[dict[str, Any]]] = {}
    exact_nodes_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    all_root_nodes = graph["source_contexts"]
    audited_locations: list[dict[str, Any]] = []
    for location in locations:
        path = location["path"]
        symbol = location["symbol"]
        index_item = index_by_path.get(path)
        if index_item is None:
            audited_locations.append(
                {
                    **location,
                    "read_id": None,
                    "directory_rank": None,
                    "default_source_covered": False,
                    "default_symbol_present": False,
                    "exact_root_source_covered": False,
                    "all_roots_source_covered": False,
                    "gold_symbol_available": False,
                }
            )
            continue
        read_id = str(index_item["read_id"])
        if path not in default_nodes_by_path:
            default_nodes_by_path[path] = _full_nodes(retriever.read(read_id))
        default_nodes = default_nodes_by_path[path]
        exact_key = (path, symbol)
        if exact_key not in exact_nodes_by_key:
            available = {
                endpoint[1]
                for endpoint in retriever.endpoints_by_path.get(path, [])
            }
            exact_nodes_by_key[exact_key] = (
                _full_nodes(retriever.read(read_id, root_symbols=[symbol]))
                if symbol in available
                else []
            )
        exact_nodes = exact_nodes_by_key[exact_key]
        audited_locations.append(
            {
                **location,
                "read_id": read_id,
                "directory_rank": rank_by_path.get(path),
                "default_source_covered": _source_covered(location, default_nodes),
                "default_symbol_present": any(
                    node["path"] == path and node["symbol"] == symbol
                    for node in default_nodes
                ),
                "exact_root_source_covered": _source_covered(location, exact_nodes),
                "all_roots_source_covered": _source_covered(
                    location, all_root_nodes
                ),
                "gold_symbol_available": any(
                    node["path"] == path and node["symbol"] == symbol
                    for node in all_root_nodes
                ),
            }
        )

    items: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for location in audited_locations:
        items[location["item_id"]].append(location)
    unique_files = sorted({location["path"] for location in audited_locations})
    return {
        "input_id": bundle.input_id,
        "benchmark": bundle.benchmark,
        "directory_entries": len(directory["entries"]),
        "gold_files": [
            {"path": path, "rank": rank_by_path.get(path)} for path in unique_files
        ],
        "locations": audited_locations,
        "items": [
            {
                "item_id": item_id,
                "locations": len(values),
                "default_any": any(item["default_source_covered"] for item in values),
                "default_all": all(item["default_source_covered"] for item in values),
                "exact_root_any": any(
                    item["exact_root_source_covered"] for item in values
                ),
                "exact_root_all": all(
                    item["exact_root_source_covered"] for item in values
                ),
                "all_roots_any": any(
                    item["all_roots_source_covered"] for item in values
                ),
                "all_roots_all": all(
                    item["all_roots_source_covered"] for item in values
                ),
            }
            for item_id, values in sorted(items.items())
        ],
    }


def _gold_locations(gold: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    if gold.get("benchmark") == "specgap":
        for condition in gold.get("conditions", []):
            for location in condition.get("implementation_locations", []):
                result.append(
                    _location(
                        str(condition["condition_id"]),
                        location["file"],
                        location["symbol"],
                        location.get("line_ranges", []),
                    )
                )
    elif gold.get("benchmark") == "silentswap":
        for index, swap in enumerate(gold.get("swaps", []), start=1):
            location = swap["localization"]
            qualified = location["symbol"]["qualified_name"]
            result.append(
                _location(
                    f"swap_{index:03d}",
                    location["file"],
                    ".".join(str(item) for item in qualified),
                    location.get("line_ranges", []),
                )
            )
    else:
        raise ValueError("Gold benchmark is not a Repo benchmark")
    return result


def _location(
    item_id: str,
    path: Any,
    symbol: Any,
    ranges: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "item_id": item_id,
        "path": str(path).replace("\\", "/"),
        "symbol": str(symbol),
        "line_ranges": [
            [int(item["start"]), int(item["end"])] for item in ranges
        ],
    }


def _full_nodes(local_graph: dict[str, Any]) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    for graph in local_graph["graphs"]:
        nodes.append(graph["root"])
        nodes.extend(
            step["node"]
            for path in graph["paths"]
            for step in path["steps"]
            if "ref" not in step["node"]
        )
    return nodes


def _source_covered(location: dict[str, Any], nodes: list[dict[str, Any]]) -> bool:
    ranges = location["line_ranges"]
    if not ranges:
        return any(
            node["path"] == location["path"]
            and node["symbol"] == location["symbol"]
            for node in nodes
        )
    return all(
        any(
            node["path"] == location["path"]
            and int(node["lines"][0]) <= start
            and end <= int(node["lines"][1])
            for node in nodes
        )
        for start, end in ranges
    )


def _summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    files = [item for sample in samples for item in sample["gold_files"]]
    locations = [item for sample in samples for item in sample["locations"]]
    items = [item for sample in samples for item in sample["items"]]
    listed_ranks = sorted(
        int(item["rank"]) for item in files if item["rank"] is not None
    )
    return {
        "samples": len(samples),
        "gold_files": len(files),
        "listed_gold_files": len(listed_ranks),
        "gold_file_listed_rate": _rate(len(listed_ranks), len(files)),
        "gold_file_rank": {
            "mean": round(statistics.mean(listed_ranks), 2) if listed_ranks else None,
            "median": statistics.median(listed_ranks) if listed_ranks else None,
            "p90": _nearest_rank(listed_ranks, 0.9),
            "max": listed_ranks[-1] if listed_ranks else None,
        },
        "gold_file_top_k_rate": {
            str(limit): _rate(
                sum(item["rank"] is not None and item["rank"] <= limit for item in files),
                len(files),
            )
            for limit in (1, 3, 5, 10, 20)
        },
        "gold_locations": len(locations),
        "default_source_covered": sum(
            item["default_source_covered"] for item in locations
        ),
        "default_source_coverage_rate": _rate(
            sum(item["default_source_covered"] for item in locations),
            len(locations),
        ),
        "default_symbol_present_rate": _rate(
            sum(item["default_symbol_present"] for item in locations),
            len(locations),
        ),
        "exact_root_source_coverage_rate": _rate(
            sum(item["exact_root_source_covered"] for item in locations),
            len(locations),
        ),
        "all_roots_source_coverage_rate": _rate(
            sum(item["all_roots_source_covered"] for item in locations),
            len(locations),
        ),
        "gold_symbol_available_rate": _rate(
            sum(item["gold_symbol_available"] for item in locations),
            len(locations),
        ),
        "gold_items": len(items),
        "default_item_any_rate": _rate(
            sum(item["default_any"] for item in items), len(items)
        ),
        "default_item_all_rate": _rate(
            sum(item["default_all"] for item in items), len(items)
        ),
        "exact_root_item_any_rate": _rate(
            sum(item["exact_root_any"] for item in items), len(items)
        ),
        "exact_root_item_all_rate": _rate(
            sum(item["exact_root_all"] for item in items), len(items)
        ),
        "all_roots_item_any_rate": _rate(
            sum(item["all_roots_any"] for item in items), len(items)
        ),
        "all_roots_item_all_rate": _rate(
            sum(item["all_roots_all"] for item in items), len(items)
        ),
    }


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _nearest_rank(values: list[int], percentile: float) -> int | None:
    return values[math.ceil(percentile * len(values)) - 1] if values else None


def _load_gold(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Gold must be an object: {path}")
    return value


def _load_checkpoint(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 2:
        raise ValueError("invalid Gold retrieval audit checkpoint")
    samples = value.get("samples")
    if not isinstance(samples, dict):
        raise ValueError("invalid Gold retrieval audit samples")
    return samples


def _save_checkpoint(path: Path, samples: dict[str, dict[str, Any]]) -> None:
    atomic_write(
        path,
        canonical_json_bytes({"schema_version": 2, "samples": samples}),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("visible_roots", nargs="+", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.visible_roots, checkpoint=args.checkpoint)
    if args.output is not None:
        atomic_write(args.output, canonical_json_bytes(result))
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(json.dumps(result["benchmarks"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
