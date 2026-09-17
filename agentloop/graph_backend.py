"""AgentLoop adapter for module-six linear Local Graph retrieval."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Sequence
from typing import Any, Callable

from ebg.behavior_directory import document_section_bodies, render_ranked_directory
from ebg.core.model import VisibleBundle
from ebg.local_graph_retrieval import LocalGraphRetriever, RetrievalError
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
        behavior_level_neighbors: bool = True,
        compact_rendering: bool = True,
        minimal_roots: bool = True,
        expand_neighbors: bool = True,
        document_quotes: bool = False,
        expansion_hops: int = 1,
    ) -> None:
        try:
            retriever = LocalGraphRetriever(
                bundle,
                graph,
                directory,
                count_tokens=count_tokens,
                behavior_level_neighbors=behavior_level_neighbors,
                compact_edges=compact_rendering,
                minimal_roots=minimal_roots,
                expand_neighbors=expand_neighbors,
                expansion_hops=expansion_hops,
            )
        except RetrievalError as error:
            raise BackendError(str(error)) from error

        self.input_id = bundle.input_id
        self.benchmark = bundle.benchmark
        self.compact_rendering = compact_rendering
        self.document_bodies = (
            document_section_bodies(bundle.task_document.content)
            if document_quotes and bundle.task_document is not None
            else None
        )
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
                f"behavior_level_neighbors={behavior_level_neighbors}",
                f"compact_rendering={compact_rendering}",
                f"minimal_roots={minimal_roots}",
                f"expand_neighbors={expand_neighbors}",
                *(["document_quotes=True"] if document_quotes else []),
                *([f"expansion_hops={expansion_hops}"] if expansion_hops != 1 else []),
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
        rendered_source = (
            _render_compact_local_graph(local_graph, document_bodies=self.document_bodies)
            if self.compact_rendering
            else json.dumps(local_graph, ensure_ascii=False, indent=2)
        )
        return EvidenceUnit(
            unit_id=read_id,
            unit_kind="local_graph",
            name=f"{path} Local Graph",
            span=span,
            source=rendered_source,
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
            path = str(node["path"])
            symbol = str(node["symbol"])
            for start, end, source in _numbered_source_regions(str(node["source"])):
                key = (path, symbol, start, end)
                if key in seen:
                    continue
                seen.add(key)
                regions.append(
                    SourceRegion(
                        symbols=(symbol,),
                        start=start,
                        end=end,
                        source=source,
                        path=path,
                    )
                )
    return tuple(regions)


def _numbered_source_regions(source: str) -> tuple[tuple[int, int, str], ...]:
    regions: list[tuple[int, int, str]] = []
    current: list[tuple[int, str]] = []
    for line in source.splitlines():
        match = re.match(r"^(\d+) \|", line)
        if match is None:
            if current:
                regions.append(
                    (current[0][0], current[-1][0], "\n".join(item[1] for item in current))
                )
                current = []
            continue
        line_number = int(match.group(1))
        if current and line_number != current[-1][0] + 1:
            regions.append(
                (current[0][0], current[-1][0], "\n".join(item[1] for item in current))
            )
            current = []
        current.append((line_number, line))
    if current:
        regions.append(
            (current[0][0], current[-1][0], "\n".join(item[1] for item in current))
        )
    return tuple(regions)


def _render_compact_local_graph(
    local_graph: dict[str, Any], *, document_bodies: dict[str, str] | None = None
) -> str:
    lines: list[str] = []
    shown_source_lines: set[tuple[str, int]] = set()
    for graph in local_graph["graphs"]:
        root = graph["root"]
        if lines:
            lines.append("")
        lines.append("[DIRECT ROOT]")
        if "ref" in root:
            lines.append(f"Root already shown: {root['ref']}")
        else:
            document_section = root.get("document_section")
            if document_section:
                body = (document_bodies or {}).get(document_section)
                if body:
                    lines.append("Doc (original text):")
                    lines.extend(
                        f"> {line}"
                        for line in _document_quote(body, str(root["symbol"])).splitlines()
                    )
                else:
                    lines.append(f"Doc: {document_section}")
            ranges = _source_range_label(str(root["source"]))
            lines.append(f"{root['path']}::{root['symbol']}@{ranges}")
            source = _deduplicated_source(
                str(root["source"]), str(root["path"]), shown_source_lines
            )
            if source:
                lines.append(source)
        for graph_path in graph["paths"]:
            for step in graph_path["steps"]:
                relation = _compact_relation_name(str(step["edge"]))
                lines.extend(
                    ["", f"[CONTEXT via {relation}]", str(step["edge"])]
                )
                node = step["node"]
                if "ref" not in node:
                    source = _deduplicated_source(
                        str(node["source"]),
                        str(node["path"]),
                        shown_source_lines,
                    )
                    if source:
                        lines.append(source)
    return "\n".join(lines).rstrip()


def _document_quote(body: str, symbol: str) -> str:
    """Select verbatim paragraphs/list items naming the symbol, or the full body."""
    blocks = re.split(r"\n[ \t]*\n|\n(?=[ \t]*[-*+] )", body)
    names = [symbol, symbol.rsplit(".", 1)[-1]]
    pattern = re.compile(r"(?<![\w])(?:" + "|".join(re.escape(n) for n in names) + r")(?![\w])")
    matching = [block for block in blocks if pattern.search(block)]
    return "\n\n".join(matching) if matching else body


def _compact_relation_name(edge: str) -> str:
    if re.search(r"\sfeeds(?:\([^)]*\))?\s", edge):
        return "feeds"
    if re.search(r"\scalls\s", edge):
        return "calls"
    return "relation"


def _source_range_label(source: str) -> str:
    return ",".join(
        str(start) if start == end else f"{start}-{end}"
        for start, end, _ in _numbered_source_regions(source)
    ) or "?"


def _deduplicated_source(
    source: str,
    path: str,
    shown: set[tuple[str, int]],
) -> str:
    retained: list[tuple[int, str]] = []
    for line in source.splitlines():
        match = re.match(r"^(\d+) \|", line)
        if match is None:
            continue
        line_number = int(match.group(1))
        key = (path, line_number)
        if key in shown:
            continue
        shown.add(key)
        retained.append((line_number, line))
    rendered: list[str] = []
    previous: int | None = None
    for line_number, line in retained:
        if previous is not None and previous + 1 < line_number:
            rendered.append("...")
        rendered.append(line)
        previous = line_number
    return "\n".join(rendered)


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
