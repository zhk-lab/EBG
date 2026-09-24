"""One-hop evidence components and deterministic display roots."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

from ..ebg.behavior_directory import _explicit_code_name_pattern
from .matching import _literal_pattern, _qualified_symbol_aliases

Endpoint = tuple[str, str]


def endpoint(value: dict[str, str]) -> Endpoint:
    return value['path'], value['symbol']


def node_id(node: Endpoint) -> str:
    return f'{node[0]}::{node[1]}'


def document_position(node: Endpoint, document: str) -> int:
    path, symbol = node
    aliases = [symbol, *_qualified_symbol_aliases(path, symbol)]
    matches = [match.start() for alias in aliases
               if (match := _literal_pattern(alias).search(document))]
    if not matches:
        matches = [match.start() for match in _explicit_code_name_pattern(symbol.rsplit('.', 1)[-1]).finditer(document)]
    return min(matches, default=len(document))


def components(seeds: set[Endpoint], edges: list[dict[str, Any]], document: str) -> list[dict[str, Any]]:
    """Select every incident edge once, then traverse only that frozen selection."""
    adjacency: dict[Endpoint, set[Endpoint]] = {seed: set() for seed in seeds}
    selected = []
    for edge in edges:
        left, right = endpoint(edge['from']), endpoint(edge['to'])
        if left not in seeds and right not in seeds:
            continue
        selected.append(edge)
        adjacency.setdefault(left, set())
        adjacency.setdefault(right, set())
        if left != right:
            adjacency[left].add(right)
            adjacency[right].add(left)

    positions = {node: document_position(node, document) for node in adjacency}

    def order(node: Endpoint):
        return positions[node], *node

    remaining = set(adjacency)
    result = []
    while remaining:
        start = min(remaining, key=order)
        members = {start}
        pending = [start]
        while pending:
            for neighbor in adjacency[pending.pop()] - members:
                members.add(neighbor)
                pending.append(neighbor)
        remaining -= members
        root = min(members & seeds, key=lambda node: (-len(adjacency[node]), *order(node)))
        visited = {root}
        queue = deque([root])
        traversal = []
        while queue:
            node = queue.popleft()
            traversal.append(node)
            for neighbor in sorted(adjacency[node] - visited, key=order):
                visited.add(neighbor)
                queue.append(neighbor)
        result.append({'root': root, 'nodes': traversal,
                       'edges': [edge for edge in selected if endpoint(edge['from']) in members]})
    return sorted(result, key=lambda component: order(component['root']))


def relations_by_node(edges: list[dict[str, Any]], evidence: dict[str, Any]) -> dict[Endpoint, list[dict]]:
    result = defaultdict(list)
    for edge in edges:
        left, right = endpoint(edge['from']), endpoint(edge['to'])
        relation = {'from': node_id(left), 'to': node_id(right), 'type': edge['type'],
                    'positions': [evidence[key]['locator'] for key in edge['evidence_ids']]}
        for node in {left, right}:
            if relation not in result[node]:
                result[node].append(relation)
    return result
