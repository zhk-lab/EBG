"""AgentLoop adapter for module-six linear Local Graph retrieval."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from typing import Any, Callable

from beg.behavior_directory import render_ranked_directory
from beg.core.model import VisibleBundle
from beg.local_graph_retrieval import LocalGraphRetriever, RetrievalError
from evaluation_core.contracts import EvidenceSpan

from .backend import TokenCounter, content_identity, pack_read_result
from .errors import BackendError
from .evidence import (
    EvidenceUnit,
    SearchHit,
    SourceRegion,
    ToolResult,
    render_tool_result,
)


class GraphBackend:
    """Expose Ranked Directory files as complete linear Local Graph units."""

    kind = "graph"

    def __init__(
        self,
        bundle: VisibleBundle,
        graph: dict[str, Any],
        directory: dict[str, Any],
        *,
        count_tokens: TokenCounter | None = None,
    ) -> None:
        try:
            retriever = LocalGraphRetriever(
                bundle,
                graph,
                directory,
                count_tokens=count_tokens,
            )
        except RetrievalError as error:
            raise BackendError(str(error)) from error

        self.input_id = bundle.input_id
        self.benchmark = bundle.benchmark
        self.initial_index = render_ranked_directory(directory)
        self.initial_index_tokens = int(directory["token_count"])
        self._retriever = retriever
        self._paths_by_id = {
            str(item["read_id"]): str(item["path"])
            for item in directory["query_index"]
        }
        self._paths_by_id.update(
            {
                read_id: endpoint[0]
                for read_id, endpoint in retriever.root_endpoint_by_read_id.items()
            }
        )
        self._units: dict[tuple[str, int], EvidenceUnit] = {}
        displayed = {
            str(item["read_id"])
            for item in directory["entries"]
            if str(item["read_id"]) in self._paths_by_id
        }
        self._initial_ids = frozenset(displayed)
        self._priority_groups = _priority_groups(directory, displayed)
        self._resume_identity = content_identity(
            [
                json.dumps(
                    graph,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                json.dumps(
                    directory,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            ]
        )

    @property
    def initial_ids(self) -> frozenset[str]:
        return self._initial_ids

    @property
    def priority_groups(self) -> dict[str, tuple[str, ...]]:
        return dict(self._priority_groups)

    @property
    def resume_identity(self) -> dict[str, object]:
        return dict(self._resume_identity)

    def has_id(self, unit_id: str) -> bool:
        return unit_id in self._paths_by_id

    def search(
        self,
        text: str,
        *,
        limit: int,
        exact_path_limit: int | None = None,
    ) -> ToolResult:
        folded = text.strip().casefold()
        is_exact_path = any(
            path.casefold() == folded for path in self._paths_by_id.values()
        )
        requested_limit = (
            exact_path_limit
            if is_exact_path and exact_path_limit is not None
            else limit
        )
        result_limit = min(12, requested_limit)
        try:
            matches = self._retriever.search(text, limit=result_limit)
        except RetrievalError as error:
            raise BackendError(str(error)) from error
        hits = tuple(
            SearchHit(
                unit_id=str(item["read_id"]),
                name=str(item["path"]),
                path=str(item["path"]),
                symbol="; ".join(item["matched_symbols"]) or "<file>",
                matched_field=str(item["match_reason"]),
            )
            for item in matches
        )
        return ToolResult(
            action="search",
            status="ok",
            search_text=text,
            total_matches=len(hits),
            hits=hits,
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
            if read_id not in self._paths_by_id:
                raise BackendError(f"unknown read ID: {read_id}")
            cache_key = (read_id, max_atomic_unit_tokens)
            if cache_key not in self._units:
                self._units[cache_key] = self._build_unit(
                    read_id,
                    self._paths_by_id[read_id],
                    token_budget=max_atomic_unit_tokens,
                    count_tokens=count_tokens,
                )
            units[read_id] = self._units[cache_key]
        return pack_read_result(
            ids,
            units,
            token_budget=token_budget,
            max_atomic_unit_tokens=max_atomic_unit_tokens,
            count_tokens=count_tokens,
        )

    def local_graph(
        self,
        read_id: str,
        *,
        root_symbols: Sequence[str] | None = None,
        token_budget: int = 65_536,
        measure_tokens: Callable[[dict[str, Any]], int] | None = None,
    ) -> dict[str, Any]:
        """Return a module-six graph, optionally rooted by an exact search hit."""

        try:
            return self._retriever.read(
                read_id,
                root_symbols=root_symbols,
                token_budget=token_budget,
                measure_tokens=measure_tokens,
            )
        except RetrievalError as error:
            raise BackendError(str(error)) from error

    def _build_unit(
        self,
        read_id: str,
        path: str,
        *,
        token_budget: int,
        count_tokens: TokenCounter,
    ) -> EvidenceUnit:
        def measure(local_graph: dict[str, Any]) -> int:
            unit = self._unit_from_local_graph(read_id, path, local_graph)
            result = ToolResult(
                action="read",
                status="ok",
                requested_ids=(read_id,),
                units=(unit,),
            )
            return count_tokens(render_tool_result(result))

        local_graph = self.local_graph(
            read_id,
            token_budget=token_budget,
            measure_tokens=measure,
        )
        return self._unit_from_local_graph(read_id, path, local_graph)

    def _unit_from_local_graph(
        self, read_id: str, path: str, local_graph: dict[str, Any]
    ) -> EvidenceUnit:
        regions = _source_regions(local_graph)
        if regions:
            span = EvidenceSpan(
                path=regions[0].path or path,
                symbol=", ".join(regions[0].symbols),
                start=regions[0].start,
                end=regions[0].end,
            )
        else:
            span = EvidenceSpan(
                path=path,
                symbol="<no observable behavior>",
                start=1,
                end=1,
            )
        return EvidenceUnit(
            unit_id=read_id,
            unit_kind="local_graph",
            name=f"{path} Local Graph",
            span=span,
            source=json.dumps(local_graph, ensure_ascii=False, indent=2),
            source_regions=regions,
            groundable=bool(regions),
            behavior_total=sum(
                len(item["root"].get("behaviors", []))
                + sum(
                    len(step["node"].get("behaviors", []))
                    for graph_path in item["paths"]
                    for step in graph_path["steps"]
                )
                for item in local_graph["graphs"]
            ),
        )


def _source_regions(local_graph: dict[str, Any]) -> tuple[SourceRegion, ...]:
    regions: list[SourceRegion] = []
    seen: set[tuple[str, str, int, int]] = set()
    for graph in local_graph["graphs"]:
        nodes = [] if "ref" in graph["root"] else [graph["root"]]
        nodes.extend(
            step["node"]
            for path in graph["paths"]
            for step in path["steps"]
            if "ref" not in step["node"]
        )
        for node in nodes:
            key = (
                str(node["path"]),
                str(node["symbol"]),
                int(node["lines"][0]),
                int(node["lines"][1]),
            )
            if key in seen:
                continue
            seen.add(key)
            regions.append(
                SourceRegion(
                    symbols=(key[1],),
                    start=key[2],
                    end=key[3],
                    source=str(node["source"]),
                    path=key[0],
                )
            )
    return tuple(regions)


def _priority_groups(
    directory: dict[str, Any], displayed: set[str]
) -> dict[str, tuple[str, ...]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for item in directory["entries"]:
        read_id = str(item["read_id"])
        if read_id not in displayed:
            continue
        sections = item["related_document_sections"] or ["Ranked Directory"]
        for section in sections:
            groups[str(section)].append(read_id)
    return {
        section: tuple(dict.fromkeys(read_ids))
        for section, read_ids in groups.items()
    }
