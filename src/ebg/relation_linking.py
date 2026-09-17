"""Relation Linking: build deterministic, source-proven direct edges."""

from __future__ import annotations

import ast
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Iterable
from urllib.parse import urlsplit, urlunsplit

from .behavior_atomization import validate_behaviors
from .core.errors import RelationError
from .core.model import RepoArtifact, VisibleBundle
from .core.syntax import parse_python
from .evidence_intake import validate_evidence


REPO_RELATIONS = {"calls", "feeds"}
STATE_MUTATORS = {
    "add", "append", "clear", "discard", "extend", "insert", "pop",
    "remove", "setdefault", "sort", "update",
}
OBJECT_KEYS = {
    "file", "file_path", "filepath", "path", "uri", "url", "task_id",
    "taskid", "execution_id", "executionid", "job_id", "jobid",
    "sample_id", "sampleid",
}
WINDOWS_PATH = re.compile(r"^[A-Za-z]:[\\/]")

Endpoint = tuple[str, str]
Binding = tuple[str, str | None]


@dataclass(frozen=True, slots=True)
class _Scope:
    path: str
    symbol: str
    node: ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef


class _EvidenceIndex:
    def __init__(self, evidence: list[dict[str, Any]]) -> None:
        self.by_id = {str(item["evidence_id"]): item for item in evidence}
        self.order = {
            str(item["evidence_id"]): index for index, item in enumerate(evidence)
        }
        self.by_scope: dict[Endpoint, list[tuple[int, int, str]]] = defaultdict(list)
        for item in evidence:
            if item.get("source_type") != "code":
                continue
            locator = item["locator"]
            key = (str(locator["path"]), str(locator["symbol"]))
            self.by_scope[key].append(
                (
                    int(locator["line_start"]),
                    int(locator["line_end"]),
                    str(item["evidence_id"]),
                )
            )

    def for_node(self, endpoint: Endpoint, node: ast.AST) -> list[str]:
        start = int(getattr(node, "lineno", 1))
        end = int(getattr(node, "end_lineno", start))
        found = {
            evidence_id
            for line_start, line_end, evidence_id in self.by_scope.get(endpoint, [])
            if line_start <= end and start <= line_end
        }
        return self.sorted(found)

    def sorted(self, evidence_ids: Iterable[str]) -> list[str]:
        return sorted(set(evidence_ids), key=self.order.__getitem__)


class _Resolver:
    def __init__(
        self,
        trees: dict[str, ast.Module],
        definitions: set[Endpoint],
    ) -> None:
        self.definitions = definitions
        self.by_path: dict[str, set[str]] = defaultdict(set)
        for path, symbol in definitions:
            self.by_path[path].add(symbol)
        self.module_paths: dict[str, set[str]] = defaultdict(set)
        for path in trees:
            module = PurePosixPath(path).with_suffix("").as_posix().replace("/", ".")
            candidates = {module}
            parts = module.split(".")
            if parts[0] in {"src", "lib"} and len(parts) > 1:
                candidates.add(".".join(parts[1:]))
            if parts[-1] == "__init__":
                candidates.update(
                    candidate.rsplit(".__init__", 1)[0]
                    for candidate in tuple(candidates)
                )
            for candidate in candidates:
                self.module_paths[candidate].add(path)

    def resolve_call(
        self,
        source: Endpoint,
        function: ast.AST,
        bindings: dict[str, set[Binding]],
    ) -> Endpoint | None:
        label = _call_label(function)
        if not label:
            return None
        parts = label.split(".")
        candidates: set[Endpoint] = set()
        if len(parts) == 1:
            candidates.update(self._same_path(source, parts[0]))
            candidates.update(self._binding_targets(bindings.get(parts[0], set()), ""))
        elif parts[0] in {"self", "cls"}:
            class_symbol = source[1].rsplit(".", 1)[0] if "." in source[1] else ""
            if class_symbol:
                candidate = (source[0], f"{class_symbol}.{'.'.join(parts[1:])}")
                if candidate in self.definitions:
                    candidates.add(candidate)
        elif parts[0] in bindings:
            candidates.update(
                self._binding_targets(bindings[parts[0]], ".".join(parts[1:]))
            )
        else:
            candidate = (source[0], label)
            if candidate in self.definitions:
                candidates.add(candidate)
        return next(iter(candidates)) if len(candidates) == 1 else None

    def imported_state(
        self,
        binding: Binding,
        suffix: str,
    ) -> tuple[str, str] | None:
        module, imported = binding
        paths = self._paths_for_module(module)
        if len(paths) != 1:
            return None
        path = next(iter(paths))
        name = ".".join(part for part in (imported, suffix) if part)
        return f"{path}::<module>.{name}", name

    def _same_path(self, source: Endpoint, name: str) -> set[Endpoint]:
        path, source_symbol = source
        candidates: set[Endpoint] = set()
        parents = source_symbol.split(".")[:-1]
        for length in range(len(parents), 0, -1):
            symbol = ".".join([*parents[:length], name])
            if (path, symbol) in self.definitions:
                candidates.add((path, symbol))
        for symbol in (name, f"{name}.__init__"):
            if (path, symbol) in self.definitions:
                candidates.add((path, symbol))
        return candidates

    def _binding_targets(
        self,
        bindings: set[Binding],
        suffix: str,
    ) -> set[Endpoint]:
        candidates: set[Endpoint] = set()
        for module, imported in bindings:
            symbol = ".".join(part for part in (imported, suffix) if part)
            for path in self._paths_for_module(module):
                for target_symbol in (symbol, f"{symbol}.__init__"):
                    if (path, target_symbol) in self.definitions:
                        candidates.add((path, target_symbol))
            if imported and suffix:
                nested_module = f"{module}.{imported}"
                for path in self._paths_for_module(nested_module):
                    if (path, suffix) in self.definitions:
                        candidates.add((path, suffix))
        return candidates

    def _paths_for_module(self, module: str) -> set[str]:
        paths = set(self.module_paths.get(module, set()))
        if paths:
            return paths
        matches = [
            values
            for name, values in self.module_paths.items()
            if name.endswith(f".{module}")
        ]
        return set(matches[0]) if len(matches) == 1 else set()


def build_edges(
    bundle: VisibleBundle,
    evidence: list[dict[str, Any]],
    behaviors: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    index = _EvidenceIndex(evidence)
    if bundle.benchmark == "feedbacktrace":
        edges = _build_trace_edges(index, behaviors)
    else:
        edges = _deduplicate_edges(
            _build_repo_edges(bundle, index, behaviors), index
        )
    validate_edges(bundle, evidence, behaviors, edges)
    return edges


def validate_edges(
    bundle: VisibleBundle,
    evidence: list[dict[str, Any]],
    behaviors: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> None:
    validate_evidence(bundle, evidence)
    validate_behaviors(bundle, evidence, behaviors)
    evidence_by_id = {str(item["evidence_id"]): item for item in evidence}
    expected_ids = [f"R{index:04d}" for index in range(1, len(edges) + 1)]
    if [edge.get("edge_id") for edge in edges] != expected_ids:
        raise RelationError("Relation IDs must be contiguous and deterministic")
    if bundle.benchmark == "feedbacktrace":
        _validate_trace_edges(evidence_by_id, behaviors, edges)
        return
    repo_endpoints = (
        {(str(item["path"]), str(item["symbol"])) for item in behaviors}
    )
    identities: list[tuple[Any, ...]] = []
    for edge in edges:
        if set(edge) != {
            "edge_id", "from", "type", "to", "via", "evidence_ids",
        }:
            raise RelationError("Relation has an unexpected field")
        relation = edge["type"]
        if relation not in REPO_RELATIONS:
            raise RelationError("Relation type is invalid for this benchmark")
        source = _validate_repo_endpoint(edge["from"], repo_endpoints)
        target = _validate_repo_endpoint(edge["to"], repo_endpoints)
        via = edge["via"]
        if via is not None and (not isinstance(via, str) or not via):
            raise RelationError("Relation via must be null or a non-empty string")
        if relation == "calls" and via is not None:
            raise RelationError(f"{relation} relation must use a null via")
        evidence_ids = edge["evidence_ids"]
        if (
            not isinstance(evidence_ids, list)
            or not evidence_ids
            or len(evidence_ids) != len(set(evidence_ids))
            or any(item not in evidence_by_id for item in evidence_ids)
        ):
            raise RelationError("Relation requires unique, known supporting Evidence")
        if any(evidence_by_id[item]["source_type"] != "code" for item in evidence_ids):
            raise RelationError("Relation Evidence has the wrong source type")
        identities.append((_endpoint_key(source), relation, _endpoint_key(target)))
    if len(identities) != len(set(identities)):
        raise RelationError("Same endpoints and relation type must be deduplicated")
    if identities != sorted(identities):
        raise RelationError("Relations are not deterministically ordered")


def _validate_repo_endpoint(
    value: Any,
    repo_endpoints: set[Endpoint],
) -> Endpoint:
    if not isinstance(value, dict) or set(value) != {"path", "symbol"}:
        raise RelationError("Repo relation endpoint must contain path and symbol")
    endpoint = (value.get("path"), value.get("symbol"))
    if endpoint not in repo_endpoints:
        raise RelationError("Repo relation endpoint has no Behavior")
    return str(endpoint[0]), str(endpoint[1])


def _build_repo_edges(
    bundle: VisibleBundle,
    index: _EvidenceIndex,
    behaviors: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    behavior_by_endpoint: dict[Endpoint, list[dict[str, Any]]] = defaultdict(list)
    for behavior in behaviors:
        behavior_by_endpoint[(str(behavior["path"]), str(behavior["symbol"]))].append(
            behavior
        )
    definitions = set(behavior_by_endpoint)
    artifacts = {artifact.path: artifact for artifact in bundle.repo_artifacts}
    trees: dict[str, ast.Module] = {}
    scopes: dict[Endpoint, list[_Scope]] = defaultdict(list)
    for artifact in bundle.repo_artifacts:
        if not _is_python(artifact):
            continue
        try:
            tree = parse_python(artifact.content, artifact.path)
        except SyntaxError:
            continue
        trees[artifact.path] = tree
        for scope in _python_scopes(artifact.path, tree):
            scopes[(scope.path, scope.symbol)].append(scope)
    unique_scopes = {
        endpoint: values[0]
        for endpoint, values in scopes.items()
        if len(values) == 1 and endpoint in definitions
    }
    resolver = _Resolver(trees, definitions)
    module_bindings = {
        path: _import_bindings(path, _owned_nodes(_Scope(path, "<module>", tree)))
        for path, tree in trees.items()
    }
    drafts: list[dict[str, Any]] = []
    writes: dict[str, list[tuple[Endpoint, list[str], str]]] = defaultdict(list)
    reads: dict[str, list[tuple[Endpoint, list[str], str]]] = defaultdict(list)
    template_endpoints = _template_endpoints(artifacts, definitions)

    for endpoint in sorted(unique_scopes):
        scope = unique_scopes[endpoint]
        nodes = list(_owned_nodes(scope))
        parents = _parent_map(nodes)
        bindings = _merge_bindings(
            module_bindings.get(scope.path, {}),
            _import_bindings(scope.path, nodes),
        )
        for node in nodes:
            if not isinstance(node, ast.Call):
                continue
            call_evidence = index.for_node(endpoint, node)
            if not call_evidence:
                continue
            target = resolver.resolve_call(endpoint, node.func, bindings)
            if target is not None:
                drafts.append(_repo_draft(endpoint, "calls", target, None, call_evidence))
                if not isinstance(parents.get(node), ast.Expr):
                    result_evidence = [
                        evidence_id
                        for behavior in behavior_by_endpoint[target]
                        if behavior["result_type"] in {"return", "yield"}
                        for evidence_id in behavior["result_evidence_ids"]
                    ]
                    if result_evidence:
                        drafts.append(
                            _repo_draft(
                                target,
                                "feeds",
                                endpoint,
                                "result",
                                [*result_evidence, *call_evidence],
                            )
                        )
            template_name = _literal_template_name(node)
            if template_name is not None:
                for template_endpoint in _matching_template_endpoints(
                    template_name, template_endpoints
                ):
                    result_evidence = [
                        evidence_id
                        for behavior in behavior_by_endpoint[template_endpoint]
                        for evidence_id in behavior["result_evidence_ids"]
                    ]
                    drafts.append(
                        _repo_draft(
                            template_endpoint,
                            "feeds",
                            endpoint,
                            "result",
                            [*result_evidence, *call_evidence],
                        )
                    )

        state_writers = any(
            behavior["result_type"] == "state_write"
            for behavior in behavior_by_endpoint[endpoint]
        )
        global_names = {
            name
            for node in nodes
            if isinstance(node, ast.Global)
            for name in node.names
        }
        local_names = _local_names(scope, nodes)
        for node in nodes:
            if state_writers and isinstance(
                node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Delete)
            ):
                evidence_ids = index.for_node(endpoint, node)
                for target in _assignment_targets(node):
                    state = _state_reference(
                        endpoint, target, bindings, resolver, global_names, local_names
                    )
                    if state is not None and evidence_ids:
                        writes[state[0]].append((endpoint, evidence_ids, state[1]))
            if state_writers and isinstance(node, ast.Call):
                if (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr.casefold() in STATE_MUTATORS
                ):
                    evidence_ids = index.for_node(endpoint, node)
                    state = _state_reference(
                        endpoint,
                        node.func.value,
                        bindings,
                        resolver,
                        global_names,
                        local_names,
                    )
                    if state is not None and evidence_ids:
                        writes[state[0]].append((endpoint, evidence_ids, state[1]))
            if isinstance(node, (ast.Name, ast.Attribute)) and isinstance(
                getattr(node, "ctx", None), ast.Load
            ):
                evidence_ids = index.for_node(endpoint, node)
                state = _state_reference(
                    endpoint, node, bindings, resolver, global_names, local_names
                )
                if state is not None and evidence_ids:
                    reads[state[0]].append((endpoint, evidence_ids, state[1]))

    for reference in sorted(set(writes) & set(reads)):
        for writer, write_evidence, field in writes[reference]:
            for reader, read_evidence, _ in reads[reference]:
                if writer != reader:
                    drafts.append(
                        _repo_draft(
                            writer,
                            "feeds",
                            reader,
                            field,
                            [*write_evidence, *read_evidence],
                        )
                    )
    return drafts


def _trace_keys(content: str) -> set[str]:
    """Extract explicit typed IDs and object URLs, not commands or shared paths."""
    pattern = re.compile(
        r"\b(task|job|run|execution|trial|shard)"
        r"(?:[_ ]+id[\s\"':=#`]*|[\"']\s*:\s*[\"']?|\s*[:=#-]\s*|\s+(?=\d))"
        r"([A-Za-z0-9][A-Za-z0-9_.-]*)\b", re.IGNORECASE,
    )
    keys = {
        f"{match[1].lower()}:{match[2]}"
        for match in pattern.finditer(content)
        if not match[2].isdigit() and any(character.isdigit() for character in match[2])
    }
    for match in re.finditer(r"\bcommit[\s\"':=`*]+([0-9a-f]{7,40})\b", content, re.I):
        keys.add(f"commit:{match[1].lower()}")
    # Object-specific URLs carry identity; generic repository URLs do not.
    for match in re.finditer(r"https?://[^\s<>\"']+", content):
        url = match[0].rstrip(".,;:!?)]}")
        if re.search(r"#(?:issuecomment|discussion_r|pullrequestreview)-\d+$", url) or re.search(
            r"/(?:commit/[0-9a-f]{7,40}|actions/runs/\d+)(?:[/?#]|$)", url
        ):
            keys.add(f"url:{url}")
    return keys


def _trace_output(item: dict[str, Any]) -> str:
    if item["locator"]["event_type"] == "assistant_response":
        return str(item["content"])
    if item["locator"]["event_type"] == "tool_exchange":
        _, separator, result = str(item["content"]).partition("Tool result:")
        return result if separator else ""
    return ""


def _trace_reference_text(item: dict[str, Any]) -> str:
    content = str(item["content"])
    if item["locator"]["event_type"] == "tool_exchange":
        # Repeated identifiers in results/logs do not demonstrate a reference.
        return content.partition("Tool result:")[0]
    return content


def _normalized_reference(text: str) -> str:
    return " ".join(text.casefold().split())


def _structural_support(
    request: str, prior: list[tuple[str, str, str]],
    recent: list[tuple[str, str, str]],
) -> list[tuple[str, str]]:
    """Resolve explicit quotations and item labels without topic/verb matching."""
    matches: list[tuple[str, str]] = []
    quotes = [next(value for value in groups if value) for groups in
              re.findall(r'"([^"\n]{20,})"|“([^”\n]{20,})”', request)]
    quotes.extend(re.findall(r"(?m)^\s*>\s?(.{20,})$", request))
    for quote in quotes:
        quote = _normalized_reference(quote)
        # Single code/path tokens are object mentions, not quoted propositions.
        if not re.search(r"\s|[\u3400-\u9fff]", quote):
            continue
        candidates = [(task, eid) for task, eid, text in prior
                      if quote in _normalized_reference(text)]
        if len({task for task, _ in candidates}) == 1:
            matches.append(candidates[-1])

    labels = set()
    for match in re.finditer(
        r"\b(?:item|option|point)\s+#?(\d+|[A-Z])\b"
        r"|第\s*(\d+)\s*[项条点]"
        r"|(?<!\w)#(\d+)\b", request, re.I,
    ):
        # Issue/PR numbers have repository scope, not local list scope.
        if re.search(r"\b(?:issue|pr|pull request)\s*$", request[:match.start()], re.I):
            continue
        labels.add(next(value for value in match.groups() if value).upper())
    bare_label = re.fullmatch(r"\s*([0-9]+|[A-Z])[.)]?\s*", request, re.I)
    if bare_label:
        labels.add(bare_label[1].upper())
    for label in sorted(labels):
        item_pattern = re.compile(
            rf"(?:^|\n)\s*(?:[-*]\s*)?(?:\*\*)?{re.escape(label)}[.)、]"
            rf"|(?:^|\n)\s*\|\s*(?:\*\*)?{re.escape(label)}(?:\*\*)?\s*\|"
            rf"|\boption\s+{re.escape(label)}\b", re.I,
        )
        candidates = [(task, eid) for task, eid, text in recent
                      for _ in item_pattern.finditer(text)]
        # Repeated label occurrences can refer to distinct lists, even in one task.
        if len(candidates) == 1:
            matches.append(candidates[0])
    return matches


def _build_trace_edges(
    index: _EvidenceIndex, behaviors: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    task_evidence: dict[str, set[str]] = defaultdict(set)
    contents: dict[tuple[str, str], list[str]] = defaultdict(list)
    for behavior in behaviors:
        task = str(behavior["task_id"])
        ids = task_evidence[task]
        ids.update(behavior["action_evidence_ids"])
        for evidence_id in behavior["action_evidence_ids"]:
            contents[task, evidence_id].append(str(index.by_id[evidence_id]["content"]))
        for field in ("demand_refs", "response_refs"):
            for ref in behavior[field]:
                evidence_id = ref["evidence_id"]
                ids.add(evidence_id)
                content = str(index.by_id[evidence_id]["content"])
                start, end = ref.get("char_range", (0, len(content)))
                contents[task, evidence_id].append(content[start:end])

    def position(evidence_id: str) -> tuple[int, int]:
        locator = index.by_id[evidence_id]["locator"]
        return int(locator["turn"]), int(locator["event_index"])

    first = {task: min(ids, key=position) for task, ids in task_evidence.items()}
    groups: dict[tuple[int, int], list[str]] = defaultdict(list)
    for task, evidence_id in first.items():
        groups[position(evidence_id)].append(task)
    drafts: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    ordered = sorted(groups)
    for earlier, later in zip(ordered, ordered[1:]):
        # A simultaneous start has no uniquely determined adjacent task.
        if len(groups[earlier]) == len(groups[later]) == 1:
            source, target = groups[earlier][0], groups[later][0]
            drafts[source, "precedes", target].update((first[source], first[target]))
    records = []
    for task, ids in task_evidence.items():
        for evidence_id in index.sorted(ids):
            item = {**index.by_id[evidence_id], "content": "\n".join(contents[task, evidence_id])}
            records.append((task, evidence_id, item))
    records.sort(key=lambda record: (position(record[1]), record[0]))
    output_index: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for owner, eid, other in records:
        for key in _trace_keys(_trace_output(other)):
            output_index[key].append((owner, eid))
    for task, evidence_id, item in records:
        request = _trace_reference_text(item)
        # Assistant repetition alone is not an explicit reference operation.
        keys = _trace_keys(request) if item["locator"]["event_type"] != "assistant_response" else set()
        prior = [(owner, eid, other) for owner, eid, other in records
                 if position(eid) < position(evidence_id)]
        for key in keys:
            candidates = [(owner, eid) for owner, eid in output_index.get(key, [])
                          if position(eid) < position(evidence_id)]
            # Without an explicit recency pointer, multiple source tasks are ambiguous.
            if len({owner for owner, _ in candidates}) == 1:
                owner, source_id = candidates[-1]
                if owner != task:
                    drafts[task, "references", owner].update((source_id, evidence_id))
        if item["locator"]["event_type"] == "user_prompt":
            # Numbered replies refer to the preceding response, not an
            # unrelated old list. Explicit quotations can reach farther back.
            prior_responses = [(owner, eid, str(other["content"]))
                               for owner, eid, other in prior
                               if other["locator"]["event_type"] == "assistant_response"]
            last_user = max((position(eid) for _, eid, other in prior
                             if other["locator"]["event_type"] == "user_prompt"),
                            default=(-1, -1))
            recent = [record for record in prior_responses if position(record[1]) > last_user]
            for owner, source_id in _structural_support(request, prior_responses, recent):
                source_tasks = {other_task for other_task, eid, _ in prior_responses if eid == source_id}
                if owner != task and source_tasks == {owner}:
                    drafts[task, "references", owner].update((source_id, evidence_id))
    return [
        {
            "edge_id": f"R{number:04d}",
            "source_task_id": source,
            "target_task_id": target,
            "type": relation,
            "evidence_ids": index.sorted(drafts[source, relation, target]),
        }
        for number, (source, relation, target) in enumerate(sorted(drafts), 1)
    ]


def _validate_trace_edges(
    evidence_by_id: dict[str, dict[str, Any]],
    behaviors: list[dict[str, Any]], edges: list[dict[str, Any]],
) -> None:
    expected = _build_trace_edges(_EvidenceIndex(list(evidence_by_id.values())), behaviors)
    if edges != expected:
        raise RelationError("Trace edges do not match explicit references and task order")


def _deduplicate_edges(
    drafts: Iterable[dict[str, Any]],
    index: _EvidenceIndex,
) -> list[dict[str, Any]]:
    merged: dict[tuple[Any, ...], dict[str, Any]] = {}
    vias: dict[tuple[Any, ...], set[str]] = defaultdict(set)
    for draft in drafts:
        identity = (
            _endpoint_key(draft["from"]),
            str(draft["type"]),
            _endpoint_key(draft["to"]),
        )
        if identity not in merged:
            merged[identity] = {
                "from": draft["from"],
                "type": draft["type"],
                "to": draft["to"],
                "evidence_ids": [],
            }
        merged[identity]["evidence_ids"].extend(draft["evidence_ids"])
        if isinstance(draft.get("via"), str) and draft["via"]:
            vias[identity].add(draft["via"])
    result: list[dict[str, Any]] = []
    for number, identity in enumerate(sorted(merged), start=1):
        item = merged[identity]
        relation = str(item["type"])
        via_values = sorted(vias[identity])
        via = (
            None
            if relation == "calls" or not via_values
            else " | ".join(via_values)
        )
        result.append(
            {
                "edge_id": f"R{number:04d}",
                "from": item["from"],
                "type": relation,
                "to": item["to"],
                "via": via,
                "evidence_ids": index.sorted(item["evidence_ids"]),
            }
        )
    return result


def _repo_draft(
    source: Endpoint,
    relation: str,
    target: Endpoint,
    via: str | None,
    evidence_ids: list[str],
) -> dict[str, Any]:
    return {
        "from": _endpoint_object(source),
        "type": relation,
        "to": _endpoint_object(target),
        "via": via,
        "evidence_ids": evidence_ids,
    }


def _endpoint_object(endpoint: Endpoint) -> dict[str, str]:
    return {"path": endpoint[0], "symbol": endpoint[1]}


def _endpoint_key(endpoint: dict[str, Any] | str | Endpoint) -> tuple[str, str]:
    if isinstance(endpoint, dict):
        return str(endpoint["path"]), str(endpoint["symbol"])
    if isinstance(endpoint, tuple):
        return endpoint
    return "", endpoint


def _python_scopes(path: str, tree: ast.Module) -> list[_Scope]:
    result = [_Scope(path, "<module>", tree)]

    def visit(body: list[ast.stmt], parent: str | None) -> None:
        for statement in body:
            if not isinstance(
                statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                for child_body in _statement_child_bodies(statement):
                    visit(child_body, parent)
                continue
            if isinstance(
                statement, (ast.FunctionDef, ast.AsyncFunctionDef)
            ) and _is_overload(statement):
                continue
            symbol = f"{parent}.{statement.name}" if parent else statement.name
            result.append(_Scope(path, symbol, statement))
            visit(statement.body, symbol)

    visit(tree.body, None)
    return result


def _owned_nodes(scope: _Scope) -> Iterable[ast.AST]:
    stack: list[ast.AST] = list(reversed(scope.node.body))
    while stack:
        node = stack.pop()
        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda),
        ):
            continue
        yield node
        stack.extend(reversed(list(ast.iter_child_nodes(node))))


def _parent_map(nodes: list[ast.AST]) -> dict[ast.AST, ast.AST]:
    owned = set(nodes)
    return {
        child: parent
        for parent in nodes
        for child in ast.iter_child_nodes(parent)
        if child in owned
    }


def _statement_child_bodies(statement: ast.stmt) -> list[list[ast.stmt]]:
    bodies: list[list[ast.stmt]] = []
    for _, value in ast.iter_fields(statement):
        if isinstance(value, list) and value and all(
            isinstance(item, ast.stmt) for item in value
        ):
            bodies.append(value)
        elif isinstance(value, ast.ExceptHandler):
            bodies.append(value.body)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, ast.ExceptHandler):
                    bodies.append(item.body)
                elif isinstance(item, ast.match_case):
                    bodies.append(item.body)
    return bodies


def _import_bindings(
    path: str,
    nodes: Iterable[ast.AST],
) -> dict[str, set[Binding]]:
    result: dict[str, set[Binding]] = defaultdict(set)
    for node in nodes:
        if isinstance(node, ast.Import):
            for alias in node.names:
                result[alias.asname or alias.name.split(".")[0]].add(
                    (alias.name, None)
                )
        elif isinstance(node, ast.ImportFrom):
            module = _import_from_module(path, node)
            if module is None:
                continue
            for alias in node.names:
                if alias.name != "*":
                    result[alias.asname or alias.name].add((module, alias.name))
    return result


def _merge_bindings(
    first: dict[str, set[Binding]],
    second: dict[str, set[Binding]],
) -> dict[str, set[Binding]]:
    result = {name: set(values) for name, values in first.items()}
    for name, values in second.items():
        result.setdefault(name, set()).update(values)
    return result


def _import_from_module(path: str, statement: ast.ImportFrom) -> str | None:
    if statement.level == 0:
        return statement.module
    package = list(PurePosixPath(path).with_suffix("").parts[:-1])
    remove = statement.level - 1
    if remove > len(package):
        return None
    if remove:
        package = package[:-remove]
    if statement.module:
        package.extend(statement.module.split("."))
    return ".".join(package) or None


def _local_names(scope: _Scope, nodes: Iterable[ast.AST]) -> set[str]:
    names: set[str] = set()
    if isinstance(scope.node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        arguments = scope.node.args
        names.update(argument.arg for argument in arguments.posonlyargs)
        names.update(argument.arg for argument in arguments.args)
        names.update(argument.arg for argument in arguments.kwonlyargs)
        if arguments.vararg:
            names.add(arguments.vararg.arg)
        if arguments.kwarg:
            names.add(arguments.kwarg.arg)
    for node in nodes:
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
    return names


def _state_reference(
    endpoint: Endpoint,
    node: ast.AST,
    bindings: dict[str, set[Binding]],
    resolver: _Resolver,
    global_names: set[str],
    local_names: set[str],
) -> tuple[str, str] | None:
    path, symbol = endpoint
    if isinstance(node, ast.Attribute):
        label = _call_label(node)
        if not label:
            return None
        parts = label.split(".")
        if parts[0] in {"self", "cls"} and "." in symbol:
            class_symbol = symbol.rsplit(".", 1)[0]
            field = ".".join(parts[1:])
            return f"{path}::{class_symbol}.{field}", field
        if parts[0] in bindings and len(bindings[parts[0]]) == 1:
            binding = next(iter(bindings[parts[0]]))
            return resolver.imported_state(binding, ".".join(parts[1:]))
        return None
    if not isinstance(node, ast.Name):
        return None
    if node.id in bindings and len(bindings[node.id]) == 1:
        binding = next(iter(bindings[node.id]))
        if binding[1] is not None:
            return resolver.imported_state(binding, "")
    if symbol == "<module>" or node.id in global_names or node.id not in local_names:
        return f"{path}::<module>.{node.id}", node.id
    return None


def _assignment_targets(node: ast.AST) -> list[ast.AST]:
    if isinstance(node, ast.Assign):
        return list(node.targets)
    if isinstance(node, ast.AnnAssign):
        return [node.target]
    if isinstance(node, ast.AugAssign):
        return [node.target]
    if isinstance(node, ast.Delete):
        return list(node.targets)
    return []


def _template_endpoints(
    artifacts: dict[str, RepoArtifact],
    definitions: set[Endpoint],
) -> dict[str, set[Endpoint]]:
    result: dict[str, set[Endpoint]] = defaultdict(set)
    for endpoint in definitions:
        artifact = artifacts.get(endpoint[0])
        if artifact is not None and artifact.kind == "runtime_template":
            result[endpoint[0]].add(endpoint)
    return result


def _matching_template_endpoints(
    template_name: str,
    endpoints: dict[str, set[Endpoint]],
) -> set[Endpoint]:
    suffix = f"/{template_name}"
    paths = [
        path for path in endpoints if path == template_name or path.endswith(suffix)
    ]
    return set(endpoints[paths[0]]) if len(paths) == 1 else set()


def _literal_template_name(call: ast.Call) -> str | None:
    label = _call_label(call.func)
    if not label or label.rsplit(".", 1)[-1] not in {
        "get_template", "render_template",
    }:
        return None
    value = call.args[0] if call.args else None
    if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
        return None
    raw = value.value.replace("\\", "/")
    pure = PurePosixPath(raw)
    if not raw or pure.is_absolute() or ".." in pure.parts:
        return None
    return pure.as_posix()


def _call_label(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_label(node.value)
        return f"{prefix}.{node.attr}" if prefix else ""
    return ""


def extract_trace_objects(content: str) -> set[str]:
    """Return exact file, URL, and task identifiers from a tool exchange."""
    marker = "Tool invocation:"
    start = content.find(marker)
    if start < 0:
        return set()
    payload = content[start + len(marker) :]
    result_marker = payload.find("\n\nTool result:")
    if result_marker >= 0:
        payload = payload[:result_marker]
    try:
        value = json.loads(payload.strip())
    except json.JSONDecodeError:
        return set()
    objects: set[str] = set()

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                normalized = str(key).casefold().replace("-", "_")
                if normalized in OBJECT_KEYS and isinstance(child, str):
                    canonical = _canonical_object(normalized, child)
                    if canonical is not None:
                        objects.add(canonical)
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return objects


def _canonical_object(key: str, value: str) -> str | None:
    text = value.strip().strip("\"'")
    if not text:
        return None
    if key in {"url", "uri"} or text.casefold().startswith(("http://", "https://")):
        parsed = urlsplit(text)
        if parsed.scheme and parsed.netloc:
            path = re.sub(r"/{2,}", "/", parsed.path).rstrip("/") or "/"
            return urlunsplit(
                (parsed.scheme.casefold(), parsed.netloc.casefold(), path, parsed.query, "")
            )
    if key in {"file", "file_path", "filepath", "path"} or WINDOWS_PATH.match(text):
        normalized = re.sub(r"/{2,}", "/", text.replace("\\", "/"))
        if WINDOWS_PATH.match(text):
            normalized = normalized[0].casefold() + normalized[1:]
        return normalized.rstrip("/") or "/"
    if key in {
        "task_id", "taskid", "execution_id", "executionid", "job_id",
        "jobid", "sample_id", "sampleid",
    }:
        return f"{key}:{text}"
    return None


def _is_overload(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return any(
        _call_label(decorator.func if isinstance(decorator, ast.Call) else decorator)
        .rsplit(".", 1)[-1]
        == "overload"
        for decorator in node.decorator_list
    )


def _is_python(artifact: RepoArtifact) -> bool:
    suffix = PurePosixPath(artifact.path).suffix.casefold()
    first = artifact.content.splitlines()[0] if artifact.content.splitlines() else ""
    return suffix in {".py", ".pyw"} or (
        first.startswith("#!") and "python" in first.casefold()
    )
