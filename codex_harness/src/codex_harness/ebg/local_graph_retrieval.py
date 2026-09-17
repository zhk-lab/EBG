"""Repo-only linear Local Graph retrieval for module six."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Callable, Iterable

import tiktoken

from .behavior_directory import select_minimal_repo_roots, validate_ranked_directory
from .core.errors import EBGError
from .core.model import VisibleBundle


class RetrievalError(EBGError):
    """A Local Graph query or read cannot be satisfied exactly."""


TokenCounter = Callable[[str], int]
GraphTokenCounter = Callable[[dict[str, Any]], int]
Endpoint = tuple[str, str]
NodeKey = tuple[str, str, int, int]
MAX_EXPANDED_EDGES = 4
MAX_LOCAL_GRAPH_TOKENS = 65_536
COMPACT_LOCAL_GRAPH_TOKENS = 32_768


class LocalGraphRetriever:
    """Search a complete Repo graph and render one-hop linear Local Graphs."""

    def __init__(
        self,
        bundle: VisibleBundle,
        graph: dict[str, Any],
        directory: dict[str, Any],
        *,
        count_tokens: TokenCounter | None = None,
        behavior_level_neighbors: bool = False,
        compact_edges: bool = False,
        minimal_roots: bool = False,
        expand_neighbors: bool = True,
    ) -> None:
        if bundle.benchmark not in {"specgap", "silentswap"}:
            raise RetrievalError("Local Graph retrieval supports only Repo benchmarks")
        try:
            validate_ranked_directory(
                bundle, graph, directory, count_tokens=count_tokens
            )
        except EBGError as error:
            raise RetrievalError("Local Graph requires a valid Ranked Directory") from error
        self.bundle = bundle
        self.graph = graph
        self.directory = directory
        self.count_tokens = count_tokens or _default_token_count
        self.behavior_level_neighbors = behavior_level_neighbors
        self.compact_edges = compact_edges
        self.minimal_roots = minimal_roots
        self.expand_neighbors = expand_neighbors
        self.evidence_by_id = {
            str(item["evidence_id"]): item for item in graph["evidence"]
        }
        self.evidence_by_path: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in graph["evidence"]:
            self.evidence_by_path[str(item["locator"]["path"])].append(item)
        self.behaviors_by_endpoint: dict[Endpoint, list[dict[str, Any]]] = defaultdict(list)
        for behavior in graph["behaviors"]:
            self.behaviors_by_endpoint[
                (str(behavior["path"]), str(behavior["symbol"]))
            ].append(behavior)
        self.contexts_by_endpoint: dict[Endpoint, list[dict[str, Any]]] = defaultdict(list)
        for context in graph["source_contexts"]:
            self.contexts_by_endpoint[
                (str(context["path"]), str(context["symbol"]))
            ].append(context)
        for values in self.contexts_by_endpoint.values():
            values.sort(key=lambda item: (int(item["lines"][0]), int(item["lines"][1])))
        self.complete_behavior_endpoints = self._complete_behavior_endpoints()
        self.index_by_id = {
            str(item["read_id"]): item for item in directory["query_index"]
        }
        self.read_id_by_path = {
            str(item["path"]): str(item["read_id"])
            for item in directory["query_index"]
        }
        self.root_symbols_by_path = {
            str(item["path"]): tuple(str(value) for value in item["root_symbols"])
            for item in directory["query_index"]
        }
        self.module_root_evidence_ids: dict[Endpoint, tuple[str, ...]] = {}
        self.document_section_by_endpoint: dict[Endpoint, str] = {}
        if minimal_roots:
            (
                self.root_symbols_by_path,
                self.module_root_evidence_ids,
                self.document_section_by_endpoint,
            ) = select_minimal_repo_roots(
                bundle,
                graph,
                count_tokens=self.count_tokens,
            )
        self.edge_behavior_ids = self._map_edge_behavior_ids()
        direct_ids = {
            str(item["read_id"])
            for item in directory["entries"]
            if item["related_document_sections"]
        }
        self.direct_endpoints = {
            (str(item["path"]), str(symbol))
            for item in directory["query_index"]
            if str(item["read_id"]) in direct_ids
            for symbol in self.root_symbols_by_path.get(str(item["path"]), ())
        }
        self.endpoints_by_path: dict[str, list[Endpoint]] = defaultdict(list)
        for endpoint in self.behaviors_by_endpoint:
            self.endpoints_by_path[endpoint[0]].append(endpoint)
        for values in self.endpoints_by_path.values():
            values.sort(key=self._endpoint_order)
        self.root_endpoint_by_read_id: dict[str, Endpoint] = {}
        self.root_read_id_by_endpoint: dict[Endpoint, str] = {}
        for path, endpoints in self.endpoints_by_path.items():
            base_read_id = self.read_id_by_path[path]
            for index, endpoint in enumerate(endpoints, start=1):
                root_read_id = f"{base_read_id}.S{index:04d}"
                self.root_endpoint_by_read_id[root_read_id] = endpoint
                self.root_read_id_by_endpoint[endpoint] = root_read_id

    @property
    def read_ids(self) -> frozenset[str]:
        return frozenset(
            [*self.index_by_id, *self.root_endpoint_by_read_id]
        )

    def search(self, text: str, *, limit: int = 12) -> list[dict[str, Any]]:
        if not isinstance(text, str) or not text.strip():
            raise RetrievalError("search text must be non-empty")
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 12:
            raise RetrievalError("search limit must be between 1 and 12")
        query = text.strip()
        folded = query.casefold()
        keywords = tuple(part for part in folded.split() if part)
        exact_root = next(
            (
                (read_id, endpoint)
                for read_id, endpoint in self.root_endpoint_by_read_id.items()
                if read_id.casefold() == folded
            ),
            None,
        )
        if exact_root is not None:
            read_id, endpoint = exact_root
            return [
                {
                    "read_id": read_id,
                    "path": endpoint[0],
                    "matched_symbols": [endpoint[1]],
                    "match_reason": "exact read ID",
                }
            ]
        ranked: list[tuple[Any, ...]] = []
        for read_id, index_item in self.index_by_id.items():
            path = str(index_item["path"])
            endpoints = self.endpoints_by_path.get(path, [])
            symbols = [endpoint[1] for endpoint in endpoints]
            behaviors = [
                behavior
                for endpoint in endpoints
                for behavior in self.behaviors_by_endpoint[endpoint]
            ]
            evidence = self.evidence_by_path.get(path, [])
            reason: str | None = None
            matched_symbols: list[str] = []
            rank = 9
            if folded == read_id.casefold():
                rank, reason = 0, "exact read ID"
            elif folded == path.casefold():
                rank, reason = 1, "exact path"
            else:
                exact_symbols = [item for item in symbols if item.casefold() == folded]
                if exact_symbols:
                    rank, reason = 2, "exact symbol"
                    matched_symbols = exact_symbols
                elif any(str(item["behavior_id"]).casefold() == folded for item in behaviors):
                    rank, reason = 3, "exact behavior ID"
                    matched_symbols = [
                        str(item["symbol"])
                        for item in behaviors
                        if str(item["behavior_id"]).casefold() == folded
                    ]
                else:
                    searchable = "\n".join(
                        [
                            read_id,
                            path,
                            *symbols,
                            *(str(item["behavior_id"]) for item in behaviors),
                            *(str(item["result_type"]) for item in behaviors),
                            *(str(item["content"]) for item in evidence),
                        ]
                    ).casefold()
                    if keywords and all(keyword in searchable for keyword in keywords):
                        rank, reason = 4, "keyword AND match"
                        matched_symbols = [
                            symbol
                            for symbol in symbols
                            if any(keyword in symbol.casefold() for keyword in keywords)
                        ]
            if reason is None:
                continue
            roots = matched_symbols or list(
                self.root_symbols_by_path.get(path, ())
            )
            result_read_id = read_id
            if reason != "exact read ID" and len(roots) == 1:
                result_read_id = self.root_read_id_by_endpoint.get(
                    (path, roots[0]), read_id
                )
            ranked.append(
                (
                    rank,
                    path.casefold(),
                    path,
                    result_read_id,
                    {
                        "read_id": result_read_id,
                        "path": path,
                        "matched_symbols": list(dict.fromkeys(roots)),
                        "match_reason": reason,
                    },
                )
            )
        ranked.sort(key=lambda item: item[:-1])
        return [item[-1] for item in ranked[:limit]]

    def read(
        self,
        read_id: str,
        *,
        root_symbols: Iterable[str] | None = None,
        token_budget: int = MAX_LOCAL_GRAPH_TOKENS,
        measure_tokens: GraphTokenCounter | None = None,
    ) -> dict[str, Any]:
        if (
            not isinstance(token_budget, int)
            or isinstance(token_budget, bool)
            or token_budget < 1
        ):
            raise RetrievalError("read token budget must be a positive integer")
        root_endpoint = self.root_endpoint_by_read_id.get(read_id)
        base_read_id = (
            self.read_id_by_path[root_endpoint[0]]
            if root_endpoint is not None
            else read_id
        )
        item = self.index_by_id.get(base_read_id)
        if item is None:
            raise RetrievalError(f"unknown read ID: {read_id}")
        path = str(item["path"])
        symbols = (
            list(dict.fromkeys(str(value) for value in root_symbols))
            if root_symbols is not None
            else (
                [root_endpoint[1]]
                if root_endpoint is not None
                else list(self.root_symbols_by_path.get(path, ()))
            )
        )
        available = {endpoint[1] for endpoint in self.endpoints_by_path.get(path, [])}
        unknown = [symbol for symbol in symbols if symbol not in available]
        if unknown:
            raise RetrievalError(f"root Symbol is not in {path}: {unknown[0]}")
        counter = measure_tokens or self._measure_local_graph
        if self.minimal_roots:
            symbols = self._budgeted_minimal_roots(
                read_id,
                path,
                symbols,
                token_budget=token_budget,
                measure_tokens=counter,
            )
        seen_full_nodes: set[NodeKey] = set()
        seen_behavior_ids: dict[NodeKey, set[str]] = defaultdict(set)
        graphs: list[dict[str, Any]] = []
        for symbol in symbols:
            root_keys = sorted(
                (
                    self._context_key(context)
                    for context in self.contexts_by_endpoint.get((path, symbol), [])
                ),
                key=lambda key: (key[2], key[3]),
            )
            trial_full = set(seen_full_nodes)
            trial_behaviors = {
                key: set(behavior_ids)
                for key, behavior_ids in seen_behavior_ids.items()
            }
            trial_graphs = [
                self._build_root_graph(key, trial_full, trial_behaviors)
                for key in root_keys
            ]
            if not self.minimal_roots:
                candidate = {
                    "read_id": read_id,
                    "graphs": [*graphs, *trial_graphs],
                }
                if graphs and counter(candidate) > token_budget:
                    break
            graphs.extend(trial_graphs)
            seen_full_nodes = trial_full
            seen_behavior_ids = trial_behaviors
        return {"read_id": read_id, "graphs": graphs}

    def _budgeted_minimal_roots(
        self,
        read_id: str,
        path: str,
        symbols: list[str],
        *,
        token_budget: int,
        measure_tokens: GraphTokenCounter,
    ) -> list[str]:
        """Choose one ranked Root prefix shared by full and Behavior arms."""

        saved_behavior_level = self.behavior_level_neighbors
        saved_compact_edges = self.compact_edges
        saved_expand_neighbors = self.expand_neighbors
        self.behavior_level_neighbors = False
        self.compact_edges = saved_compact_edges
        self.expand_neighbors = True
        selection_budget = (
            min(token_budget, COMPACT_LOCAL_GRAPH_TOKENS)
            if saved_compact_edges
            else token_budget
        )
        try:
            admitted: list[str] = []
            controls: dict[
                bool,
                tuple[
                    list[dict[str, Any]],
                    set[NodeKey],
                    dict[NodeKey, set[str]],
                ],
            ] = {
                False: ([], set(), defaultdict(set)),
                True: ([], set(), defaultdict(set)),
            }
            for symbol in symbols:
                root_keys = sorted(
                    (
                        self._context_key(context)
                        for context in self.contexts_by_endpoint.get(
                            (path, symbol), []
                        )
                    ),
                    key=lambda key: (key[2], key[3]),
                )
                trials = {}
                for behavior_level, state in controls.items():
                    self.behavior_level_neighbors = behavior_level
                    graphs, seen_full_nodes, seen_behavior_ids = state
                    trial_full = set(seen_full_nodes)
                    trial_behaviors = {
                        key: set(behavior_ids)
                        for key, behavior_ids in seen_behavior_ids.items()
                    }
                    trial_graphs = [
                        self._build_root_graph(
                            key, trial_full, trial_behaviors
                        )
                        for key in root_keys
                    ]
                    candidate_graphs = [*graphs, *trial_graphs]
                    trials[behavior_level] = (
                        candidate_graphs,
                        trial_full,
                        trial_behaviors,
                    )
                candidate_tokens = max(
                    measure_tokens(
                        {"read_id": read_id, "graphs": trial[0]}
                    )
                    for trial in trials.values()
                )
                if candidate_tokens > selection_budget:
                    if not admitted and candidate_tokens <= token_budget:
                        admitted.append(symbol)
                        controls = trials
                    break
                admitted.append(symbol)
                controls = trials
            return admitted
        finally:
            self.behavior_level_neighbors = saved_behavior_level
            self.compact_edges = saved_compact_edges
            self.expand_neighbors = saved_expand_neighbors

    def _measure_local_graph(self, value: dict[str, Any]) -> int:
        return self.count_tokens(
            json.dumps(value, ensure_ascii=False, indent=2)
        )

    def _build_root_graph(
        self,
        root_key: NodeKey,
        seen_full_nodes: set[NodeKey],
        seen_behavior_ids: dict[NodeKey, set[str]],
    ) -> dict[str, Any]:
        root_endpoint = root_key[:2]
        root_behaviors = self._root_behaviors(root_key)
        root_evidence = {
            evidence_id
            for behavior in root_behaviors
            for evidence_id in self._behavior_evidence_ids(behavior)
        }
        selected = []
        if self.expand_neighbors:
            selected = [
                edge
                for edge in self.graph["edges"]
                if self._incident(edge, root_endpoint)
                and root_evidence.intersection(edge["evidence_ids"])
                and (
                    not self.compact_edges
                    or self._is_useful_neighbor_edge(edge, root_endpoint)
                )
            ]
        selected.sort(key=lambda edge: self._edge_order(edge, root_endpoint))
        if self.compact_edges:
            distinct: list[dict[str, Any]] = []
            seen_neighbors: set[NodeKey] = set()
            for edge in selected:
                neighbor_endpoint = self._other_endpoint(edge, root_endpoint)
                neighbor_key = self._neighbor_key(neighbor_endpoint, edge)
                if neighbor_key is None or neighbor_key in seen_neighbors:
                    continue
                seen_neighbors.add(neighbor_key)
                distinct.append(edge)
            selected = distinct
        selected = selected[:MAX_EXPANDED_EDGES]

        if root_key in seen_full_nodes:
            root: dict[str, Any] = {"ref": self._node_ref(root_key)}
        else:
            root = self._node(
                root_key,
                behaviors=(
                    root_behaviors
                    if self._is_sliced_module_root(root_key)
                    else None
                ),
                source_evidence_ids=self.module_root_evidence_ids.get(
                    root_endpoint, ()
                ),
            )
            document_section = self.document_section_by_endpoint.get(root_endpoint)
            if document_section:
                root["document_section"] = document_section
            seen_full_nodes.add(root_key)
            seen_behavior_ids.pop(root_key, None)
        paths: list[dict[str, Any]] = []
        for edge in selected:
            neighbor_endpoint = self._other_endpoint(edge, root_endpoint)
            neighbor_key = self._neighbor_key(neighbor_endpoint, edge)
            if neighbor_key is None:
                continue
            matched_behaviors = (
                self._matching_neighbor_behaviors(
                    edge, root_endpoint, neighbor_key
                )
                if self.behavior_level_neighbors
                else None
            )
            rendered_behaviors = matched_behaviors
            if neighbor_key in seen_full_nodes:
                node: dict[str, Any] = {"ref": self._node_ref(neighbor_key)}
            elif matched_behaviors is None:
                node = self._node(neighbor_key)
                seen_full_nodes.add(neighbor_key)
                seen_behavior_ids.pop(neighbor_key, None)
            else:
                already_seen = seen_behavior_ids.setdefault(neighbor_key, set())
                unseen = [
                    behavior
                    for behavior in matched_behaviors
                    if str(behavior["behavior_id"]) not in already_seen
                ]
                if not unseen:
                    node = {"ref": self._node_ref(neighbor_key)}
                else:
                    node = self._node(neighbor_key, behaviors=unseen)
                    already_seen.update(
                        str(behavior["behavior_id"]) for behavior in unseen
                    )
                    rendered_behaviors = unseen
            edge_text = (
                self._compact_edge_text(
                    edge,
                    root_key,
                    neighbor_key,
                    rendered_behaviors,
                )
                if self.compact_edges
                else self._edge_text(edge)
            )
            paths.append(
                {
                    "path_id": f"P{len(paths) + 1}",
                    "steps": [
                        {
                            "edge": edge_text,
                            "node": node,
                        }
                    ],
                }
            )

        return {
            "root": root,
            "paths": paths,
        }

    def _node(
        self,
        key: NodeKey,
        *,
        behaviors: list[dict[str, Any]] | None = None,
        source_evidence_ids: Iterable[str] = (),
    ) -> dict[str, Any]:
        context = self._context_for_key(key)
        selected = self._behaviors_for_key(key) if behaviors is None else behaviors
        source = self._numbered_source(str(context["source"]), key[2])
        selected_evidence_ids = tuple(source_evidence_ids)
        if selected_evidence_ids:
            ranges = self._evidence_id_line_ranges(selected_evidence_ids, key)
            if ranges:
                source = self._numbered_source_ranges(
                    str(context["source"]), key[2], ranges
                )
        elif behaviors is not None:
            ranges = self._behavior_line_ranges(selected, key)
            if ranges:
                source = self._numbered_source_ranges(
                    str(context["source"]), key[2], ranges
                )
        return {
            "path": key[0],
            "symbol": key[1],
            "lines": [key[2], key[3]],
            "behaviors": [
                self._compact_behavior(item) for item in selected
            ],
            "source": source,
        }

    def _is_sliced_module_root(self, key: NodeKey) -> bool:
        return bool(
            self.minimal_roots
            and key[1] == "<module>"
            and self.module_root_evidence_ids.get(key[:2])
        )

    def _root_behaviors(self, key: NodeKey) -> list[dict[str, Any]]:
        behaviors = self._behaviors_for_key(key)
        evidence_ids = set(self.module_root_evidence_ids.get(key[:2], ()))
        if not self._is_sliced_module_root(key):
            return behaviors
        return [
            behavior
            for behavior in behaviors
            if evidence_ids.intersection(self._behavior_evidence_ids(behavior))
        ]

    def _matching_neighbor_behaviors(
        self,
        edge: dict[str, Any],
        root_endpoint: Endpoint,
        neighbor_key: NodeKey,
    ) -> list[dict[str, Any]] | None:
        """Return directly supported neighbor Behaviors, or None for full-Symbol fallback."""

        del root_endpoint
        if neighbor_key[:2] not in self.complete_behavior_endpoints:
            return None
        behavior_ids = set(
            self.edge_behavior_ids.get(
                (str(edge["edge_id"]), neighbor_key[:2]), ()
            )
        )
        if not behavior_ids:
            return None
        matched = [
            behavior
            for behavior in self._behaviors_for_key(neighbor_key)
            if str(behavior["behavior_id"]) in behavior_ids
        ]
        if matched and any(
            behavior["trigger_evidence_ids"] for behavior in matched
        ):
            selected_ids = {
                str(behavior["behavior_id"]) for behavior in matched
            }
            matched.extend(
                behavior
                for behavior in self._behaviors_for_key(neighbor_key)
                if str(behavior["behavior_id"]) not in selected_ids
                and behavior["result_type"] in {"return", "raise", "yield"}
                and not behavior["trigger_evidence_ids"]
                and not behavior["operation_evidence_ids"]
            )
        return matched or None

    def _complete_behavior_endpoints(self) -> set[Endpoint]:
        """Find Symbols whose meaningful code Evidence appears in a Behavior."""

        result: set[Endpoint] = set()
        for endpoint, behaviors in self.behaviors_by_endpoint.items():
            used = {
                evidence_id
                for behavior in behaviors
                for evidence_id in self._behavior_evidence_ids(behavior)
            }
            context_starts = {
                int(context["lines"][0])
                for context in self.contexts_by_endpoint.get(endpoint, [])
            }
            required: set[str] = set()
            for evidence in self.evidence_by_path.get(endpoint[0], []):
                locator = evidence["locator"]
                if str(locator["symbol"]) != endpoint[1]:
                    continue
                content = str(evidence["content"]).lstrip()
                start = int(locator["line_start"])
                is_signature = start in context_starts and content.startswith(
                    ("def ", "async def ", "class ")
                )
                is_docstring = content.startswith(('"""', "'''"))
                if not is_signature and not is_docstring:
                    required.add(str(evidence["evidence_id"]))
            if required.issubset(used):
                result.add(endpoint)
        return result

    def _map_edge_behavior_ids(
        self,
    ) -> dict[tuple[str, Endpoint], tuple[str, ...]]:
        """Resolve every edge endpoint to all Behaviors supported by its Evidence."""

        result: dict[tuple[str, Endpoint], tuple[str, ...]] = {}
        for edge in self.graph["edges"]:
            source = self._edge_endpoint(edge["from"])
            target = self._edge_endpoint(edge["to"])
            for endpoint in (source, target):
                support = self._endpoint_evidence_ids(edge, endpoint)
                behavior_ids = tuple(
                    str(behavior["behavior_id"])
                    for behavior in self.behaviors_by_endpoint.get(endpoint, [])
                    if support.intersection(
                        self._behavior_evidence_ids(behavior)
                    )
                )
                if behavior_ids:
                    result[(str(edge["edge_id"]), endpoint)] = behavior_ids
        return result

    def _endpoint_evidence_ids(
        self, edge: dict[str, Any], endpoint: Endpoint
    ) -> set[str]:
        return {
            str(evidence_id)
            for evidence_id in edge["evidence_ids"]
            if self._evidence_endpoint(str(evidence_id)) == endpoint
        }

    def _evidence_endpoint(self, evidence_id: str) -> Endpoint:
        locator = self.evidence_by_id[evidence_id]["locator"]
        return str(locator["path"]), str(locator["symbol"])

    def _behavior_line_ranges(
        self, behaviors: Iterable[dict[str, Any]], key: NodeKey
    ) -> tuple[tuple[int, int], ...]:
        ranges: list[tuple[int, int]] = []
        if key[1] != "<module>":
            for evidence in self.evidence_by_path.get(key[0], []):
                locator = evidence["locator"]
                if str(locator["symbol"]) != key[1]:
                    continue
                content = str(evidence["content"]).lstrip()
                if content.startswith(("def ", "async def ", "class ")):
                    ranges.append(
                        (
                            max(key[2], int(locator["line_start"])),
                            min(key[3], int(locator["line_end"])),
                        )
                    )
        for behavior in behaviors:
            for evidence_id in self._behavior_evidence_ids(behavior):
                evidence = self.evidence_by_id[str(evidence_id)]
                locator = evidence["locator"]
                if (
                    str(locator["path"]),
                    str(locator["symbol"]),
                ) != key[:2]:
                    continue
                start = max(key[2], int(locator["line_start"]))
                end = min(key[3], int(locator["line_end"]))
                if start <= end:
                    ranges.append((start, end))
        return self._merge_ranges(ranges)

    def _evidence_id_line_ranges(
        self, evidence_ids: Iterable[str], key: NodeKey
    ) -> tuple[tuple[int, int], ...]:
        ranges: list[tuple[int, int]] = []
        for evidence_id in evidence_ids:
            evidence = self.evidence_by_id[str(evidence_id)]
            locator = evidence["locator"]
            if (
                str(locator["path"]),
                str(locator["symbol"]),
            ) != key[:2]:
                continue
            start = max(key[2], int(locator["line_start"]))
            end = min(key[3], int(locator["line_end"]))
            if start <= end:
                ranges.append((start, end))
        return self._merge_ranges(ranges)

    def _compact_edge_text(
        self,
        edge: dict[str, Any],
        root_key: NodeKey,
        neighbor_key: NodeKey,
        neighbor_behaviors: list[dict[str, Any]] | None,
    ) -> str:
        root_endpoint = root_key[:2]
        neighbor_endpoint = neighbor_key[:2]
        source_endpoint = self._edge_endpoint(edge["from"])
        target_endpoint = self._edge_endpoint(edge["to"])
        neighbor_ranges = (
            self._behavior_line_ranges(neighbor_behaviors, neighbor_key)
            if neighbor_behaviors is not None
            else ((neighbor_key[2], neighbor_key[3]),)
        )

        def ranges_for(endpoint: Endpoint) -> tuple[tuple[int, int], ...]:
            support_ranges = self._evidence_line_ranges(edge, endpoint)
            if support_ranges:
                return support_ranges
            if endpoint == neighbor_endpoint:
                return neighbor_ranges
            if endpoint == root_endpoint:
                return ((root_key[2], root_key[3]),)
            return ()

        relation = str(edge["type"])
        if edge.get("via"):
            relation += f"({edge['via']})"
        return (
            f"{self._compact_endpoint_ref(source_endpoint, root_endpoint)}"
            f"@{self._format_ranges(ranges_for(source_endpoint))} "
            f"{relation} "
            f"{self._compact_endpoint_ref(target_endpoint, root_endpoint)}"
            f"@{self._format_ranges(ranges_for(target_endpoint))}"
        )

    def _evidence_line_ranges(
        self, edge: dict[str, Any], endpoint: Endpoint
    ) -> tuple[tuple[int, int], ...]:
        ranges = []
        for evidence_id in self._endpoint_evidence_ids(edge, endpoint):
            locator = self.evidence_by_id[evidence_id]["locator"]
            ranges.append(
                (int(locator["line_start"]), int(locator["line_end"]))
            )
        return self._merge_ranges(ranges)

    @staticmethod
    def _merge_ranges(
        ranges: Iterable[tuple[int, int]],
    ) -> tuple[tuple[int, int], ...]:
        merged: list[list[int]] = []
        for start, end in sorted(set(ranges)):
            if merged and start <= merged[-1][1] + 1:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        return tuple((start, end) for start, end in merged)

    @staticmethod
    def _format_ranges(ranges: Iterable[tuple[int, int]]) -> str:
        rendered = [
            str(start) if start == end else f"{start}-{end}"
            for start, end in ranges
        ]
        return ",".join(rendered) or "?"

    @staticmethod
    def _compact_endpoint_ref(endpoint: Endpoint, root: Endpoint) -> str:
        return endpoint[1] if endpoint[0] == root[0] else f"{endpoint[0]}::{endpoint[1]}"

    def _compact_behavior(self, behavior: dict[str, Any]) -> dict[str, Any]:
        return {
            "behavior_id": str(behavior["behavior_id"]),
            "result_type": str(behavior["result_type"]),
            "trigger": self._evidence_refs(behavior["trigger_evidence_ids"]),
            "operation": self._evidence_refs(behavior["operation_evidence_ids"]),
            "result": self._evidence_refs(behavior["result_evidence_ids"]),
        }

    def _evidence_refs(self, evidence_ids: Iterable[str]) -> list[str]:
        return [self._evidence_ref(str(item)) for item in evidence_ids]

    def _evidence_ref(self, evidence_id: str) -> str:
        locator = self.evidence_by_id[evidence_id]["locator"]
        start = int(locator["line_start"])
        end = int(locator["line_end"])
        suffix = str(start) if start == end else f"{start}-{end}"
        return f"{evidence_id}@{suffix}"

    def _edge_text(self, edge: dict[str, Any]) -> str:
        evidence = ",".join(
            self._evidence_ref(str(item)) for item in edge["evidence_ids"]
        )
        via = f" via {edge['via']}" if edge["via"] else ""
        return (
            f"{self._endpoint_ref(self._edge_endpoint(edge['from']))} "
            f"--{edge['type']}[{evidence}]{via}--> "
            f"{self._endpoint_ref(self._edge_endpoint(edge['to']))}"
        )

    def _neighbor_key(
        self, endpoint: Endpoint, edge: dict[str, Any]
    ) -> NodeKey | None:
        contexts = self.contexts_by_endpoint.get(endpoint, [])
        if len(contexts) == 1:
            return self._context_key(contexts[0])
        support = set(str(item) for item in edge["evidence_ids"])
        matched = [
            context
            for context in contexts
            if any(
                support.intersection(self._behavior_evidence_ids(behavior))
                for behavior in self._behaviors_for_context(context)
            )
        ]
        return self._context_key(matched[0]) if len(matched) == 1 else None

    def _behaviors_for_key(self, key: NodeKey) -> list[dict[str, Any]]:
        return [
            item
            for item in self.behaviors_by_endpoint.get(key[:2], [])
            if [key[2], key[3]] == item["symbol_lines"]
        ]

    def _behaviors_for_context(self, context: dict[str, Any]) -> list[dict[str, Any]]:
        return self._behaviors_for_key(self._context_key(context))

    @staticmethod
    def _behavior_evidence_ids(behavior: dict[str, Any]) -> list[str]:
        return [
            *behavior["trigger_evidence_ids"],
            *behavior["operation_evidence_ids"],
            *behavior["result_evidence_ids"],
        ]

    def _endpoint_order(self, endpoint: Endpoint) -> tuple[Any, ...]:
        contexts = self.contexts_by_endpoint.get(endpoint, [])
        start = int(contexts[0]["lines"][0]) if contexts else 0
        return start, endpoint[1].casefold(), endpoint[1]

    def _edge_order(
        self, edge: dict[str, Any], root: Endpoint
    ) -> tuple[Any, ...]:
        neighbor = self._other_endpoint(edge, root)
        if self.compact_edges:
            return (
                self._neighbor_edge_class(edge, root),
                0 if neighbor in self.direct_endpoints else 1,
                self._edge_relation_priority(edge, root),
                self._root_edge_line(edge, root),
                neighbor[0].casefold(),
                neighbor[0],
                neighbor[1].casefold(),
                neighbor[1],
                str(edge["edge_id"]),
            )
        return (
            0 if neighbor in self.direct_endpoints else 1,
            0 if edge["type"] == "feeds" else 1,
            neighbor[0].casefold(),
            neighbor[0],
            neighbor[1].casefold(),
            neighbor[1],
            str(edge["edge_id"]),
        )

    def _is_useful_neighbor_edge(
        self, edge: dict[str, Any], root: Endpoint
    ) -> bool:
        """Keep direct matches and one-hop causal dependencies/results."""

        source = self._edge_endpoint(edge["from"])
        target = self._edge_endpoint(edge["to"])
        neighbor = self._other_endpoint(edge, root)
        if neighbor in self.direct_endpoints:
            return True
        if edge["type"] == "calls":
            return source == root
        if edge["type"] == "feeds":
            return target == root or (
                source == root
                and (
                    "result" in str(edge.get("via") or "").split(" | ")
                    or self._same_symbol_owner(source, target)
                )
            )
        return False

    def _neighbor_edge_class(
        self, edge: dict[str, Any], root: Endpoint
    ) -> int:
        source = self._edge_endpoint(edge["from"])
        target = self._edge_endpoint(edge["to"])
        if edge["type"] == "calls" and source == root:
            return 0
        if edge["type"] == "feeds" and target == root:
            return 0
        if (
            edge["type"] == "feeds"
            and source == root
            and (
                "result" in str(edge.get("via") or "").split(" | ")
                or self._same_symbol_owner(source, target)
            )
        ):
            return 1
        return 2

    @staticmethod
    def _same_symbol_owner(source: Endpoint, target: Endpoint) -> bool:
        if (
            source[0] != target[0]
            or "." not in source[1]
            or "." not in target[1]
        ):
            return False
        return source[1].rsplit(".", 1)[0] == target[1].rsplit(".", 1)[0]

    def _edge_relation_priority(
        self, edge: dict[str, Any], root: Endpoint
    ) -> int:
        source = self._edge_endpoint(edge["from"])
        if edge["type"] == "calls" and source == root:
            return 0
        if edge["type"] == "feeds":
            return 1
        return 2

    def _root_edge_line(self, edge: dict[str, Any], root: Endpoint) -> int:
        lines = [
            int(self.evidence_by_id[evidence_id]["locator"]["line_start"])
            for evidence_id in edge["evidence_ids"]
            if self._evidence_endpoint(str(evidence_id)) == root
        ]
        return min(lines, default=MAX_LOCAL_GRAPH_TOKENS)

    def _context_for_key(self, key: NodeKey) -> dict[str, Any]:
        for context in self.contexts_by_endpoint[key[:2]]:
            if [key[2], key[3]] == context["lines"]:
                return context
        raise RetrievalError(f"missing source context: {self._node_ref(key)}")

    @staticmethod
    def _context_key(context: dict[str, Any]) -> NodeKey:
        return (
            str(context["path"]),
            str(context["symbol"]),
            int(context["lines"][0]),
            int(context["lines"][1]),
        )

    @staticmethod
    def _edge_endpoint(value: dict[str, Any]) -> Endpoint:
        return str(value["path"]), str(value["symbol"])

    def _other_endpoint(self, edge: dict[str, Any], endpoint: Endpoint) -> Endpoint:
        source = self._edge_endpoint(edge["from"])
        target = self._edge_endpoint(edge["to"])
        if source == endpoint:
            return target
        if target == endpoint:
            return source
        raise RetrievalError("edge is not incident to the requested root")

    def _incident(self, edge: dict[str, Any], endpoint: Endpoint) -> bool:
        return endpoint in {
            self._edge_endpoint(edge["from"]), self._edge_endpoint(edge["to"])
        }

    @staticmethod
    def _endpoint_ref(endpoint: Endpoint) -> str:
        return f"{endpoint[0]}::{endpoint[1]}"

    def _node_ref(self, key: NodeKey) -> str:
        return f"{self._endpoint_ref(key[:2])}@{key[2]}-{key[3]}"

    @staticmethod
    def _numbered_source(source: str, start: int) -> str:
        lines = source.splitlines()
        return "\n".join(
            f"{line_number} | {line}"
            for line_number, line in enumerate(lines, start=start)
        )

    @staticmethod
    def _numbered_source_ranges(
        source: str,
        source_start: int,
        ranges: Iterable[tuple[int, int]],
    ) -> str:
        lines = source.splitlines()
        rendered: list[str] = []
        previous_end: int | None = None
        for start, end in ranges:
            if previous_end is not None and previous_end + 1 < start:
                rendered.append("...")
            for line_number in range(start, end + 1):
                offset = line_number - source_start
                if 0 <= offset < len(lines):
                    rendered.append(f"{line_number} | {lines[offset]}")
            previous_end = end
        return "\n".join(rendered)


def _default_token_count(text: str) -> int:
    return len(tiktoken.get_encoding("o200k_base").encode(text))
