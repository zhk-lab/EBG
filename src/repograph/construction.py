"""Build the Python symbol graph used by the RepoGraph baseline.

The representation follows the public RepoGraph implementation and paper: function
and class definitions and call references are graph nodes, while ``contain`` and
``invoke`` are the only edge types.  The upstream implementation uses Tree-sitter for
tag extraction; this benchmark already owns immutable Python source strings, so the
standard-library AST provides the same tags without importing code from the sample.
"""

from __future__ import annotations

import ast
import builtins
import re
from pathlib import PurePosixPath
from typing import Any

from ebg.core.model import VisibleBundle
from ebg.core.syntax import parse_python


SCHEMA_VERSION = 1


class RepoGraphError(ValueError):
    """Raised when a RepoGraph artifact is invalid."""


def build_repograph(bundle: VisibleBundle) -> dict[str, Any]:
    """Construct a deterministic line-level RepoGraph from visible Python source."""

    if bundle.benchmark not in {"specgap", "silentswap"}:
        raise RepoGraphError("RepoGraph supports only repository benchmarks")

    definitions: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []
    contain_pairs: list[tuple[str, str]] = []
    parse_errors: list[dict[str, str]] = []
    for artifact in sorted(bundle.repo_artifacts, key=lambda item: item.path):
        if PurePosixPath(artifact.path).suffix != ".py":
            continue
        try:
            tree = _parse_repository_python(artifact.content, artifact.path)
        except (SyntaxError, ValueError) as error:
            parse_errors.append(
                {"path": artifact.path, "error": f"{type(error).__name__}: {error}"}
            )
            continue
        collector = _TagCollector(artifact.path, artifact.content)
        collector.visit(tree)
        definitions.extend(collector.definitions)
        references.extend(collector.references)
        contain_pairs.extend(collector.contain_pairs)

    project_names = {node["name"] for node in definitions}
    excluded_names = _excluded_callable_names()
    references = [
        node
        for node in references
        if node["name"] in project_names and node["name"] not in excluded_names
    ]

    nodes = [*definitions, *references]
    nodes.sort(key=_node_key)
    id_by_temporary = {
        node.pop("temporary_id"): f"N{index:05d}"
        for index, node in enumerate(nodes, start=1)
    }
    for index, node in enumerate(nodes, start=1):
        node["node_id"] = f"N{index:05d}"

    edges: set[tuple[str, str, str]] = set()
    for parent, child in contain_pairs:
        if parent in id_by_temporary and child in id_by_temporary:
            edges.add((id_by_temporary[parent], id_by_temporary[child], "contain"))
    for node in nodes:
        owner = node.pop("owner_temporary_id", None)
        if node["kind"] == "ref" and owner in id_by_temporary:
            edges.add((id_by_temporary[owner], node["node_id"], "invoke"))

    graph_edges = [
        {"source": source, "target": target, "relation": relation}
        for source, target, relation in sorted(edges, key=lambda item: (item[2], item[0], item[1]))
    ]
    definition_nodes = [node for node in nodes if node["kind"] == "def"]
    symbols = [
        {
            "read_id": f"S{index:04d}",
            "node_id": node["node_id"],
            "name": node["name"],
            "qualified_name": node["qualified_name"],
            "category": node["category"],
            "path": node["path"],
            "line_start": node["line_start"],
            "line_end": node["line_end"],
        }
        for index, node in enumerate(definition_nodes, start=1)
    ]
    result = {
        "schema_version": SCHEMA_VERSION,
        "method": "RepoGraph",
        "input_id": bundle.input_id,
        "benchmark": bundle.benchmark,
        "language": "python",
        "retrieval": {"strategy": "symbol_ego_graph", "hop_depth": 1},
        "nodes": nodes,
        "edges": graph_edges,
        "symbols": symbols,
        "parse_errors": parse_errors,
    }
    validate_repograph(bundle, result)
    return result


def validate_repograph(bundle: VisibleBundle, graph: dict[str, Any]) -> None:
    """Validate graph structure and that all source spans still match the bundle."""

    required = {
        "schema_version",
        "method",
        "input_id",
        "benchmark",
        "language",
        "retrieval",
        "nodes",
        "edges",
        "symbols",
        "parse_errors",
    }
    if set(graph) != required:
        raise RepoGraphError("RepoGraph artifact fields are invalid")
    if graph["schema_version"] != SCHEMA_VERSION or graph["method"] != "RepoGraph":
        raise RepoGraphError("RepoGraph artifact version or method is invalid")
    if graph["input_id"] != bundle.input_id or graph["benchmark"] != bundle.benchmark:
        raise RepoGraphError("RepoGraph artifact belongs to another input")
    if graph["language"] != "python" or graph["retrieval"] != {
        "strategy": "symbol_ego_graph",
        "hop_depth": 1,
    }:
        raise RepoGraphError("RepoGraph retrieval configuration is invalid")
    if not isinstance(graph["nodes"], list) or not isinstance(graph["edges"], list):
        raise RepoGraphError("RepoGraph nodes and edges must be arrays")
    if not isinstance(graph["symbols"], list) or not graph["symbols"]:
        raise RepoGraphError("RepoGraph contains no readable definitions")

    sources = {artifact.path: artifact.content.splitlines() for artifact in bundle.repo_artifacts}
    node_ids: set[str] = set()
    node_by_id: dict[str, dict[str, Any]] = {}
    for node in graph["nodes"]:
        if not isinstance(node, dict):
            raise RepoGraphError("RepoGraph node must be an object")
        node_id = node.get("node_id")
        if not isinstance(node_id, str) or node_id in node_ids:
            raise RepoGraphError("RepoGraph node IDs must be unique strings")
        node_ids.add(node_id)
        node_by_id[node_id] = node
        if node.get("kind") not in {"def", "ref"}:
            raise RepoGraphError("RepoGraph node kind is invalid")
        if node.get("category") not in {"class", "function"}:
            raise RepoGraphError("RepoGraph node category is invalid")
        path = node.get("path")
        start = node.get("line_start")
        end = node.get("line_end")
        if path not in sources or type(start) is not int or type(end) is not int:
            raise RepoGraphError("RepoGraph node source span is invalid")
        if not (1 <= start <= end <= len(sources[path])):
            raise RepoGraphError("RepoGraph node source span is out of bounds")
        expected = "\n".join(
            f"{line} | {sources[path][line - 1]}" for line in range(start, end + 1)
        )
        if node.get("source") != expected:
            raise RepoGraphError("RepoGraph node source does not match visible input")

    edge_keys: list[tuple[str, str, str]] = []
    for edge in graph["edges"]:
        if not isinstance(edge, dict) or set(edge) != {"source", "target", "relation"}:
            raise RepoGraphError("RepoGraph edge fields are invalid")
        if edge["source"] not in node_ids or edge["target"] not in node_ids:
            raise RepoGraphError("RepoGraph edge endpoint is missing")
        if edge["relation"] not in {"contain", "invoke"}:
            raise RepoGraphError("RepoGraph edge relation is invalid")
        edge_keys.append((edge["source"], edge["target"], edge["relation"]))
    if edge_keys != sorted(set(edge_keys), key=lambda item: (item[2], item[0], item[1])):
        raise RepoGraphError("RepoGraph edges are not canonical")

    read_ids: set[str] = set()
    for symbol in graph["symbols"]:
        read_id = symbol.get("read_id") if isinstance(symbol, dict) else None
        node = node_by_id.get(symbol.get("node_id")) if isinstance(symbol, dict) else None
        if not isinstance(read_id, str) or read_id in read_ids or node is None:
            raise RepoGraphError("RepoGraph symbol directory is invalid")
        read_ids.add(read_id)
        if node["kind"] != "def" or any(
            symbol.get(key) != node[key]
            for key in (
                "name",
                "qualified_name",
                "category",
                "path",
                "line_start",
                "line_end",
            )
        ):
            raise RepoGraphError("RepoGraph symbol does not match its definition node")


def render_symbol_directory(graph: dict[str, Any]) -> str:
    """Render the complete deterministic function/class directory."""

    lines = [
        "[REPOGRAPH SYMBOL DIRECTORY]",
        "Each S ID selects one function/class definition for one-hop graph retrieval.",
        "",
    ]
    current_path: str | None = None
    for symbol in graph["symbols"]:
        path = str(symbol["path"])
        if path != current_path:
            lines.append(f"File: {path}")
            current_path = path
        lines.append(
            f"  {symbol['read_id']}  {symbol['qualified_name']}  "
            f"{symbol['category']}  lines {symbol['line_start']}-{symbol['line_end']}"
        )
    return "\n".join(lines).rstrip() + "\n"


class _TagCollector(ast.NodeVisitor):
    def __init__(self, path: str, source: str) -> None:
        self.path = path
        self.lines = source.splitlines()
        self.definitions: list[dict[str, Any]] = []
        self.references: list[dict[str, Any]] = []
        self.contain_pairs: list[tuple[str, str]] = []
        self.scope: list[tuple[str, str]] = []
        self._reference_number = 0

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._visit_definition(node, "class")

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_definition(node, "function")

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_definition(node, "function")

    def visit_Call(self, node: ast.Call) -> None:
        name = _call_name(node.func)
        if name:
            self._reference_number += 1
            start = int(node.lineno)
            end = int(getattr(node, "end_lineno", start) or start)
            self.references.append(
                {
                    "temporary_id": (
                        f"ref:{self.path}:{start}:{end}:{name}:{self._reference_number}"
                    ),
                    "owner_temporary_id": self.scope[-1][0] if self.scope else None,
                    "name": name,
                    "qualified_name": name,
                    "kind": "ref",
                    "category": "function",
                    "path": self.path,
                    "line_start": start,
                    "line_end": end,
                    "source": self._numbered_source(start, end),
                    "owner": self.scope[-1][1] if self.scope else "<module>",
                }
            )
        self.generic_visit(node)

    def _visit_definition(
        self,
        node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
        category: str,
    ) -> None:
        owners = [item[1].split(".")[-1] for item in self.scope]
        qualified_name = ".".join([*owners, node.name])
        start = _decorated_start(node)
        end = int(getattr(node, "end_lineno", node.lineno) or node.lineno)
        temporary_id = f"def:{self.path}:{start}:{end}:{qualified_name}"
        self.definitions.append(
            {
                "temporary_id": temporary_id,
                "name": node.name,
                "qualified_name": qualified_name,
                "kind": "def",
                "category": category,
                "path": self.path,
                "line_start": start,
                "line_end": end,
                "source": self._numbered_source(start, end),
                "owner": self.scope[-1][1] if self.scope else "<module>",
            }
        )
        if self.scope:
            self.contain_pairs.append((self.scope[-1][0], temporary_id))
        self.scope.append((temporary_id, qualified_name))
        self.generic_visit(node)
        self.scope.pop()

    def _numbered_source(self, start: int, end: int) -> str:
        return "\n".join(
            f"{line} | {self.lines[line - 1]}" for line in range(start, end + 1)
        )


def _decorated_start(
    node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
) -> int:
    decorator_lines = [int(item.lineno) for item in node.decorator_list]
    return min([int(node.lineno), *decorator_lines])


def _call_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _node_key(node: dict[str, Any]) -> tuple[Any, ...]:
    return (
        node["path"],
        node["line_start"],
        0 if node["kind"] == "def" else 1,
        node["line_end"],
        node["qualified_name"],
        node["temporary_id"],
    )


def _excluded_callable_names() -> set[str]:
    names = set(dir(builtins))
    for value in (list, dict, set, str, tuple):
        names.update(dir(value))
    return names


def _parse_repository_python(source: str, path: str) -> ast.Module:
    """Parse current Python, then apply the upstream line-preserving Py2 fallbacks."""

    try:
        return parse_python(source, path)
    except SyntaxError as original:
        compatible = re.sub(
            r"(?m)^(\s*)print\s+(?!\()",
            r"\1yield ",
            source,
        )
        compatible = re.sub(
            r"(?m)^(\s*)except\s+([^:\n,()]+)\s*,\s*([^:\n]+):",
            r"\1except \2 as \3:",
            compatible,
        )
        if compatible == source:
            raise original
        return parse_python(compatible, path)
