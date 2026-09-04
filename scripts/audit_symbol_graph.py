"""Offline audit for the module-four symbol aggregation boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


UPSTREAM_FILES = (
    "l3_evidence.jsonl",
    "atomic_behaviors.json",
    "behavior_edges.json",
)


def audit(
    benchmark: str,
    old_graph_root: Path,
    new_graph_root: Path,
    gold_root: Path,
) -> dict[str, Any]:
    if benchmark not in {"specgap", "silentswap"}:
        raise ValueError("symbol graph audit supports only Repo benchmarks")
    sample_ids = sorted(path.name for path in new_graph_root.iterdir() if path.is_dir())
    if not sample_ids:
        raise ValueError("new graph root contains no samples")

    upstream_changed: list[dict[str, str]] = []
    old_covered = 0
    new_covered = 0
    locations = 0
    regressions: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for input_id in sample_ids:
        old_root = old_graph_root / input_id
        new_root = new_graph_root / input_id
        for filename in UPSTREAM_FILES:
            if (old_root / filename).read_bytes() != (new_root / filename).read_bytes():
                upstream_changed.append({"input_id": input_id, "file": filename})

        old_graph = _load(old_root / "behavior_graph.json")
        new_graph = _load(new_root / "behavior_graph.json")
        gold = _load(gold_root / f"{input_id}.json")
        old_units = _old_units(old_graph)
        new_units = _new_units(new_graph)
        for location in _gold_locations(benchmark, gold):
            locations += 1
            old_hit = _covered(location, old_units)
            new_hit = _covered(location, new_units)
            old_covered += old_hit
            new_covered += new_hit
            row = {
                "input_id": input_id,
                "file": location[0],
                "symbol": location[1],
                "ranges": [list(item) for item in location[2]],
            }
            if old_hit and not new_hit:
                regressions.append(row)
            if not new_hit:
                unresolved.append(row)

    return {
        "benchmark": benchmark,
        "samples": len(sample_ids),
        "upstream_files_compared": len(sample_ids) * len(UPSTREAM_FILES),
        "upstream_changes": upstream_changed,
        "gold_locations": locations,
        "old_covered_locations": old_covered,
        "new_covered_locations": new_covered,
        "coverage_regressions": regressions,
        "unresolved_locations": unresolved,
        "valid": not upstream_changed and not regressions,
    }


def _old_units(graph: dict[str, Any]) -> list[tuple[str, str, tuple[int, int]]]:
    return [
        (
            _path(item["location"]["path"]),
            str(item["location"]["symbol"]),
            tuple(item["location"]["lines"]),
        )
        for item in graph["behaviors"]
    ]


def _new_units(graph: dict[str, Any]) -> list[tuple[str, str, tuple[int, int]]]:
    return [
        (_path(item["path"]), str(item["symbol"]), tuple(item["lines"]))
        for item in graph["symbols"]
    ]


def _gold_locations(
    benchmark: str, gold: dict[str, Any]
) -> Iterable[tuple[str, str, tuple[tuple[int, int], ...]]]:
    if benchmark == "specgap":
        raw_locations = [
            location
            for condition in gold["conditions"]
            if condition.get("type") != "test_related"
            for location in condition.get("implementation_locations", [])
        ]
    else:
        raw_locations = [item["localization"] for item in gold["swaps"]]
    for location in raw_locations:
        symbol_value = location["symbol"]
        symbol = (
            symbol_value
            if isinstance(symbol_value, str)
            else ".".join(symbol_value["qualified_name"])
        )
        ranges = tuple(
            (int(item["start"]), int(item["end"]))
            for item in location.get("line_ranges", [])
        )
        yield _path(location["file"]), symbol, ranges


def _covered(
    location: tuple[str, str, tuple[tuple[int, int], ...]],
    units: list[tuple[str, str, tuple[int, int]]],
) -> bool:
    path, symbol, ranges = location
    same_path = [item for item in units if item[0] == path]
    exact = [item for item in same_path if item[1] == symbol]
    candidates = exact or same_path
    return bool(candidates) and (
        not ranges
        or any(_overlap(unit_range, gold_range) for _, _, unit_range in candidates for gold_range in ranges)
    )


def _overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] <= right[1] and right[0] <= left[1]


def _path(value: str) -> str:
    return value.replace("\\", "/").removeprefix("./")


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=("specgap", "silentswap"), required=True)
    parser.add_argument("--old-graph-root", type=Path, required=True)
    parser.add_argument("--new-graph-root", type=Path, required=True)
    parser.add_argument("--gold-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(
        args.benchmark,
        args.old_graph_root,
        args.new_graph_root,
        args.gold_root,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8", newline="")
    print(text, end="")
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
