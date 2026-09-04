"""Repo-only linear Local Graph retrieval for module six."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Callable, Iterable

import tiktoken

from .behavior_directory import validate_ranked_directory
from .core.errors import BEGError
from .core.model import VisibleBundle


class RetrievalError(BEGError):
    """A Local Graph query or read cannot be satisfied exactly."""


TokenCounter = Callable[[str], int]
GraphTokenCounter = Callable[[dict[str, Any]], int]
Endpoint = tuple[str, str]
NodeKey = tuple[str, str, int, int]
MAX_EXPANDED_EDGES = 4
MAX_LOCAL_GRAPH_TOKENS = 65_536


class LocalGraphRetriever:
    """Search a complete Repo graph and render one-hop linear Local Graphs."""

    def __init__(
        self,
        bundle: VisibleBundle,
        graph: dict[str, Any],
        directory: dict[str, Any],
        *,
        count_tokens: TokenCounter | None = None,
    ) -> None:
        if bundle.benchmark not in {"specgap", "silentswap"}:
            raise RetrievalError("Local Graph retrieval supports only Repo benchmarks")
        try:
            validate_ranked_directory(
                bundle, graph, directory, count_tokens=count_tokens
            )
        except BEGError as error:
            raise RetrievalError("Local Graph requires a valid Ranked Directory") from error
        self.bundle = bundle
        self.graph = graph
        self.directory = directory
        self.count_tokens = count_tokens or _default_token_count
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
        self.index_by_id = {
            str(item["read_id"]): item for item in directory["query_index"]
        }
        self.read_id_by_path = {
            str(item["path"]): str(item["read_id"])
            for item in directory["query_index"]
        }
        direct_ids = {
            str(item["read_id"])
            for item in directory["entries"]
            if item["related_document_sections"]
        }
        self.direct_endpoints = {
            (str(item["path"]), str(symbol))
            for item in directory["query_index"]
            if str(item["read_id"]) in direct_ids
            for symbol in item["root_symbols"]
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
            roots = matched_symbols or list(index_item["root_symbols"])
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
                else list(item["root_symbols"])
            )
        )
        available = {endpoint[1] for endpoint in self.endpoints_by_path.get(path, [])}
        unknown = [symbol for symbol in symbols if symbol not in available]
        if unknown:
            raise RetrievalError(f"root Symbol is not in {path}: {unknown[0]}")
        counter = measure_tokens or self._measure_local_graph
        seen_nodes: set[NodeKey] = set()
        graphs: list[dict[str, Any]] = []
        for symbol in symbols:
            root_keys = sorted(
                (
                    self._context_key(context)
                    for context in self.contexts_by_endpoint.get((path, symbol), [])
                ),
                key=lambda key: (key[2], key[3]),
            )
            trial_seen = set(seen_nodes)
            trial_graphs = [
                self._build_root_graph(key, trial_seen) for key in root_keys
            ]
            candidate = {
                "read_id": read_id,
                "graphs": [*graphs, *trial_graphs],
            }
            if graphs and counter(candidate) > token_budget:
                break
            graphs.extend(trial_graphs)
            seen_nodes = trial_seen
        return {"read_id": read_id, "graphs": graphs}

    def _measure_local_graph(self, value: dict[str, Any]) -> int:
        return self.count_tokens(
            json.dumps(value, ensure_ascii=False, indent=2)
        )

    def _build_root_graph(
        self, root_key: NodeKey, seen_nodes: set[NodeKey]
    ) -> dict[str, Any]:
        root_endpoint = root_key[:2]
        root_behaviors = self._behaviors_for_key(root_key)
        root_evidence = {
            evidence_id
            for behavior in root_behaviors
            for evidence_id in self._behavior_evidence_ids(behavior)
        }
        selected = [
            edge
            for edge in self.graph["edges"]
            if self._incident(edge, root_endpoint)
            and root_evidence.intersection(edge["evidence_ids"])
        ]
        selected.sort(key=lambda edge: self._edge_order(edge, root_endpoint))
        selected = selected[:MAX_EXPANDED_EDGES]

        if root_key in seen_nodes:
            root: dict[str, Any] = {"ref": self._node_ref(root_key)}
        else:
            root = self._node(root_key)
            seen_nodes.add(root_key)
        paths: list[dict[str, Any]] = []
        for edge in selected:
            neighbor_endpoint = self._other_endpoint(edge, root_endpoint)
            neighbor_key = self._neighbor_key(neighbor_endpoint, edge)
            if neighbor_key is None:
                continue
            node: dict[str, Any]
            if neighbor_key in seen_nodes:
                node = {"ref": self._node_ref(neighbor_key)}
            else:
                node = self._node(neighbor_key)
                seen_nodes.add(neighbor_key)
            paths.append(
                {
                    "path_id": f"P{len(paths) + 1}",
                    "steps": [
                        {
                            "edge": self._edge_text(edge),
                            "node": node,
                        }
                    ],
                }
            )

        return {
            "root": root,
            "paths": paths,
        }

    def _node(self, key: NodeKey) -> dict[str, Any]:
        context = self._context_for_key(key)
        return {
            "path": key[0],
            "symbol": key[1],
            "lines": [key[2], key[3]],
            "behaviors": [
                self._compact_behavior(item) for item in self._behaviors_for_key(key)
            ],
            "source": self._numbered_source(str(context["source"]), key[2]),
        }

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
        return (
            0 if neighbor in self.direct_endpoints else 1,
            0 if edge["type"] == "feeds" else 1,
            neighbor[0].casefold(),
            neighbor[0],
            neighbor[1].casefold(),
            neighbor[1],
            str(edge["edge_id"]),
        )

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


def _default_token_count(text: str) -> int:
    return len(tiktoken.get_encoding("o200k_base").encode(text))
