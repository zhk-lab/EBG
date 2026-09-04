"""Validate every generated Repo graph/directory through the runtime backend."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agentloop.config import DEFAULT_CONFIG
from agentloop.context import FastTokenCounter
from agentloop.evidence import render_tool_result
from agentloop.graph_backend import GraphBackend
from agentloop.raw_backend import RawBackend
from beg.evidence_intake import load_visible_bundle
from scripts.graph_io import load_json


def audit(
    graph_root: Path,
    directory_root: Path,
    visible_bundle_root: Path,
    *,
    arm: str,
) -> dict[str, int | float]:
    counter = FastTokenCounter(DEFAULT_CONFIG.token_estimator)
    samples = 0
    listed_units = 0
    oversize_units = 0
    read_token_counts: list[int] = []
    for graph_dir in sorted(path for path in graph_root.iterdir() if path.is_dir()):
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-root", type=Path, required=True)
    parser.add_argument("--directory-root", type=Path, required=True)
    parser.add_argument("--visible-bundle-root", type=Path, required=True)
    parser.add_argument("--arm", choices=("graph", "raw"), default="graph")
    args = parser.parse_args()
    print(
        json.dumps(
            audit(
                args.graph_root,
                args.directory_root,
                args.visible_bundle_root,
                arm=args.arm,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
