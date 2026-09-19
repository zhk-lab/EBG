"""AgentLoop adapter for the RepoGraph symbol/ego-graph baseline."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from ebg.core.model import VisibleBundle
from evaluation_core.contracts import EvidenceSpan
from repograph.construction import RepoGraphError, validate_repograph

from .backend import TokenCounter, content_identity, pack_read_result
from .errors import BackendError
from .evidence import EvidenceUnit, SearchHit, SourceRegion, ToolResult


class RepoGraphBackend:
    """Expose definitions as Read IDs and return flattened one-hop ego graphs."""

    kind = "repograph"

    def __init__(
        self,
        bundle: VisibleBundle,
        graph: dict[str, Any],
        *,
        index_budget: int,
        count_tokens: TokenCounter,
    ) -> None:
        try:
            validate_repograph(bundle, graph)
        except RepoGraphError as error:
            raise BackendError(str(error)) from error
        self.input_id = bundle.input_id
        self.benchmark = bundle.benchmark
        self._graph = graph
        self._nodes = {str(node["node_id"]): node for node in graph["nodes"]}
        self._node_ids_by_name: dict[str, set[str]] = defaultdict(set)
        for node_id, node in self._nodes.items():
            self._node_ids_by_name[str(node["name"])].add(node_id)
        self._symbols = {
            str(symbol["read_id"]): symbol for symbol in graph["symbols"]
        }
        self._adjacency: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
        for edge in graph["edges"]:
            source = str(edge["source"])
            target = str(edge["target"])
            relation = str(edge["relation"])
            self._adjacency[source].append((target, relation, "out"))
            self._adjacency[target].append((source, relation, "in"))
        for node_id in self._adjacency:
            self._adjacency[node_id].sort(key=lambda item: (item[1], item[2], item[0]))

        self.initial_index, selected = _build_symbol_index(
            tuple(self._symbols.values()),
            index_budget=index_budget,
            count_tokens=count_tokens,
        )
        self.initial_index_tokens = count_tokens(self.initial_index)
        self._initial_ids = frozenset(selected)
        self._units: dict[str, EvidenceUnit] = {}
        self._resume_identity = content_identity(
            [
                json.dumps(
                    graph,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                self.initial_index,
            ]
        )

    @property
    def initial_ids(self) -> frozenset[str]:
        return self._initial_ids

    @property
    def priority_groups(self) -> dict[str, tuple[str, ...]]:
        return {"RepoGraph symbols": tuple(sorted(self._initial_ids))}

    @property
    def resume_identity(self) -> dict[str, object]:
        return dict(self._resume_identity)

    def has_id(self, unit_id: str) -> bool:
        return unit_id in self._symbols

    def search(
        self,
        text: str,
        *,
        limit: int,
        exact_path_limit: int | None = None,
    ) -> ToolResult:
        query = text.strip().casefold()
        keywords = tuple(part for part in query.split() if part)
        ranked: list[tuple[int, str, SearchHit]] = []
        for read_id, symbol in self._symbols.items():
            name = str(symbol["name"])
            qualified = str(symbol["qualified_name"])
            path = str(symbol["path"])
            fields = {
                "read_id": read_id.casefold(),
                "symbol": name.casefold(),
                "qualified_symbol": qualified.casefold(),
                "path": path.casefold(),
            }
            if query == fields["read_id"]:
                rank, matched = 0, "read_id"
            elif query == fields["symbol"]:
                rank, matched = 1, "symbol"
            elif query == fields["qualified_symbol"]:
                rank, matched = 2, "qualified_symbol"
            elif query == fields["path"]:
                rank, matched = 3, "path"
            elif keywords and all(
                any(keyword in value for value in fields.values())
                for keyword in keywords
            ):
                rank, matched = 4, "symbol/path"
            else:
                continue
            ranked.append(
                (
                    rank,
                    read_id,
                    SearchHit(
                        unit_id=read_id,
                        name=f"{qualified} ({symbol['category']})",
                        path=path,
                        symbol=qualified,
                        matched_field=matched,
                    ),
                )
            )
        ranked.sort(key=lambda item: (item[0], item[1]))
        exact_path = any(
            query == str(symbol["path"]).casefold() for symbol in self._symbols.values()
        )
        result_limit = exact_path_limit if exact_path and exact_path_limit else limit
        return ToolResult(
            action="search",
            status="ok",
            search_text=text,
            total_matches=len(ranked),
            hits=tuple(item[2] for item in ranked[:result_limit]),
        )

    def read(
        self,
        ids: Sequence[str],
        *,
        token_budget: int,
        max_atomic_unit_tokens: int,
        count_tokens: TokenCounter,
    ) -> ToolResult:
        units: dict[str, EvidenceUnit] = {}
        for read_id in ids:
            if read_id not in self._symbols:
                raise BackendError(f"unknown read ID: {read_id}")
            if read_id not in self._units:
                self._units[read_id] = self._build_unit(read_id)
            units[read_id] = self._units[read_id]
        return pack_read_result(
            ids,
            units,
            token_budget=token_budget,
            max_atomic_unit_tokens=max_atomic_unit_tokens,
            count_tokens=count_tokens,
        )

    def _build_unit(self, read_id: str) -> EvidenceUnit:
        symbol = self._symbols[read_id]
        root = self._nodes[str(symbol["node_id"])]
        search_name = str(symbol["name"])
        matching_roots = {
            root["node_id"],
            *(
                node_id
                for node_id in self._node_ids_by_name[search_name]
                if self._nodes[node_id]["kind"] == "ref"
            ),
        }
        included = set(matching_roots)
        relation_by_node: dict[str, set[str]] = defaultdict(set)
        for node_id in sorted(matching_roots):
            for neighbor, relation, direction in self._adjacency.get(node_id, []):
                included.add(neighbor)
                relation_by_node[neighbor].add(f"{direction}going {relation}")

        ordered = sorted(
            included,
            key=lambda node_id: (
                0 if node_id == root["node_id"] else 1 if node_id in matching_roots else 2,
                self._nodes[node_id]["path"],
                self._nodes[node_id]["line_start"],
                node_id,
            ),
        )
        lines = [
            "[[REPOGRAPH EGO GRAPH]]",
            f"Search symbol: {search_name}",
            "Hop depth: 1",
            "Relations: contain and invoke",
            "Only numbered source below is implementation evidence.",
        ]
        regions: list[SourceRegion] = []
        seen_regions: set[tuple[str, int, int, str]] = set()
        for node_id in ordered:
            node = self._nodes[node_id]
            if node_id == root["node_id"]:
                label = "ROOT DEFINITION"
                relation = "selected symbol"
            elif node_id in matching_roots:
                label = f"ROOT {str(node['kind']).upper()}"
                relation = "same search symbol"
            else:
                label = "ONE-HOP NEIGHBOR"
                relation = ", ".join(sorted(relation_by_node[node_id]))
            lines.extend(
                [
                    "",
                    f"[{label}]",
                    f"Relation: {relation}",
                    (
                        f"{node['path']}::{node['qualified_name']}@"
                        f"{node['line_start']}-{node['line_end']} "
                        f"({node['kind']} {node['category']})"
                    ),
                    str(node["source"]),
                ]
            )
            key = (
                str(node["path"]),
                int(node["line_start"]),
                int(node["line_end"]),
                str(node["source"]),
            )
            if key not in seen_regions:
                seen_regions.add(key)
                regions.append(
                    SourceRegion(
                        path=str(node["path"]),
                        symbols=(str(node["qualified_name"]),),
                        start=int(node["line_start"]),
                        end=int(node["line_end"]),
                        source=str(node["source"]),
                    )
                )
        lines.extend(["", "[[END REPOGRAPH EGO GRAPH]]"])
        return EvidenceUnit(
            unit_id=read_id,
            unit_kind="local_graph",
            name=f"RepoGraph symbol {symbol['qualified_name']}",
            span=EvidenceSpan(
                path=str(root["path"]),
                symbol=str(root["qualified_name"]),
                start=int(root["line_start"]),
                end=int(root["line_end"]),
            ),
            source="\n".join(lines),
            source_regions=tuple(regions),
        )


def _build_symbol_index(
    symbols: tuple[dict[str, Any], ...],
    *,
    index_budget: int,
    count_tokens: TokenCounter,
) -> tuple[str, tuple[str, ...]]:
    header = [
        "This is a neutral RepoGraph symbol directory; entries are reading options, not findings.",
        "Use search with a function/class name to locate omitted symbols.",
        "Read an S ID to retrieve the flattened one-hop RepoGraph around that symbol.",
        "",
        "[REPOGRAPH SYMBOLS]",
        "",
    ]
    empty = _render_symbol_index(header, [])
    if count_tokens(empty) > index_budget:
        raise BackendError("index budget cannot hold the RepoGraph directory header")
    ordered = _coverage_order(symbols)
    low, high = 0, len(ordered)
    while low < high:
        middle = (low + high + 1) // 2
        candidate = _render_symbol_index(header, ordered[:middle])
        if count_tokens(candidate) <= index_budget:
            low = middle
        else:
            high = middle - 1
    selected = ordered[:low]
    rendered = _render_symbol_index(header, selected)
    return rendered, tuple(str(item["read_id"]) for item in selected)


def _render_symbol_index(
    header: list[str], symbols: list[dict[str, Any]]
) -> str:
    lines = list(header)
    if not symbols:
        lines.append("No symbol row fits; use search to inspect the complete symbol directory.")
    else:
        for symbol in symbols:
            lines.append(
                f"{symbol['read_id']}  {symbol['path']}::{symbol['qualified_name']}  "
                f"{symbol['category']}  lines {symbol['line_start']}-{symbol['line_end']}"
            )
    return "\n".join(lines).rstrip() + "\n"


def _coverage_order(symbols: tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
    """Interleave repository regions so a bounded index is not prefix-biased."""

    lane_count = min(8, len(symbols))
    if lane_count == 0:
        return []
    lane_size = (len(symbols) + lane_count - 1) // lane_count
    lanes = [
        symbols[start : start + lane_size]
        for start in range(0, len(symbols), lane_size)
    ]
    ordered: list[dict[str, Any]] = []
    for offset in range(lane_size):
        ordered.extend(lane[offset] for lane in lanes if offset < len(lane))
    return ordered
