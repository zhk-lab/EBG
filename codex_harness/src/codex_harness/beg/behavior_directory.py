"""Ranked Directory: deterministic retrieval entry points for Repo and Trace."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Callable, Iterable

import tiktoken

from .core.errors import BEGError, DirectoryError
from .core.model import RepoArtifact, VisibleBundle
from .graph_assembly import validate_graph


DIRECTORY_TOKEN_BUDGET = 3072
DIRECTORY_ENCODING = "o200k_base"
MAX_HINT_SYMBOLS = 5
MAX_SECTION_HINTS = 4

TokenCounter = Callable[[str], int]
_HEADING = re.compile(r"(?m)^(?P<marks>#{1,6})[ \t]+(?P<title>.+?)\s*$")
_CALL_NAME = re.compile(r"(?<![\w.])([A-Za-z_]\w*)\s*\(")
_IDENTIFIER = re.compile(r"(?<![\w.])([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)")
_BACKTICK_CODE = re.compile(r"`([^`\r\n]+)`")
_PYTHON_KEYWORDS = {
    "and", "as", "assert", "async", "await", "break", "case", "class",
    "continue", "def", "del", "elif", "else", "except", "False", "finally",
    "for", "from", "global", "if", "import", "in", "is", "lambda", "match",
    "None", "nonlocal", "not", "or", "pass", "raise", "return", "True",
    "try", "while", "with", "yield",
}
_PYTHON_KEYWORDS_FOLDED = frozenset(
    item.casefold() for item in _PYTHON_KEYWORDS
)
_ARTIFACT_ORDER = {
    "source": 0,
    "executable": 1,
    "runtime_template": 2,
    "configuration": 3,
}


@dataclass(frozen=True, slots=True)
class _Section:
    title: str
    start: int
    end: int
    order: int
    level: int


@dataclass(frozen=True, slots=True)
class _RepoFile:
    read_id: str
    path: str
    artifact_kind: str
    behavior_count: int
    symbols: tuple[str, ...]
    sections: tuple[str, ...]
    section_orders: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _RootCandidate:
    symbol: str
    priority: int
    section_orders: frozenset[int]
    evidence_ids: frozenset[str]
    source_tokens: int
    legacy_order: int


def build_ranked_directory(
    bundle: VisibleBundle,
    graph: dict[str, Any],
    *,
    count_tokens: TokenCounter | None = None,
) -> dict[str, Any]:
    """Build the module-five Repo directory or chronological Trace index."""

    _validate_graph_input(bundle, graph)
    if bundle.benchmark == "feedbacktrace":
        raise DirectoryError("FeedbackTrace bypasses Ranked Directory")
    if bundle.task_document is None:
        raise DirectoryError("Repo directory requires the complete task document")
    counter = count_tokens or _default_token_count
    if not callable(counter):
        raise DirectoryError("count_tokens must be callable")
    result = _build_repo_directory(bundle, graph, counter)
    validate_ranked_directory(bundle, graph, result, count_tokens=count_tokens)
    return result


def validate_ranked_directory(
    bundle: VisibleBundle,
    graph: dict[str, Any],
    directory: dict[str, Any],
    *,
    count_tokens: TokenCounter | None = None,
) -> None:
    """Reject a directory that is stale, reordered, enlarged, or hand-edited."""

    _validate_graph_input(bundle, graph)
    if not isinstance(directory, dict):
        raise DirectoryError("Ranked Directory must be an object")
    if bundle.benchmark == "feedbacktrace":
        raise DirectoryError("FeedbackTrace bypasses Ranked Directory")
    if bundle.task_document is None:
        raise DirectoryError("Repo directory requires the complete task document")
    counter = count_tokens or _default_token_count
    if not callable(counter):
        raise DirectoryError("count_tokens must be callable")
    expected = _build_repo_directory(bundle, graph, counter)
    if directory != expected:
        raise DirectoryError("Ranked Directory does not match its graph and visible input")


def render_ranked_directory(directory: dict[str, Any]) -> str:
    """Render only the model-facing navigation fields."""

    if directory.get("directory_type") == "repo":
        return _render_repo_entries(directory.get("entries", []))
    raise DirectoryError("Unknown Ranked Directory type")


def _validate_graph_input(bundle: VisibleBundle, graph: dict[str, Any]) -> None:
    if not isinstance(graph, dict):
        raise DirectoryError("Graph must be an object")
    evidence = graph.get("evidence")
    behaviors = graph.get("behaviors")
    edges = graph.get("edges")
    if not isinstance(evidence, list) or not isinstance(behaviors, list) or not isinstance(edges, list):
        raise DirectoryError("Graph is missing its Evidence, Behavior, or Edge list")
    try:
        validate_graph(bundle, evidence, behaviors, edges, graph)
    except BEGError as error:
        raise DirectoryError("Ranked Directory requires a valid module-four graph") from error


def _build_repo_directory(
    bundle: VisibleBundle,
    graph: dict[str, Any],
    count_tokens: TokenCounter,
) -> dict[str, Any]:
    document = bundle.task_document.content if bundle.task_document else ""
    sections = _document_sections(document)
    by_path: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for behavior in graph["behaviors"]:
        by_path[str(behavior["path"])].append(behavior)
    evidence_by_path: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in graph["evidence"]:
        evidence_by_path[str(item["locator"]["path"])].append(item)
    call_name_paths: dict[str, set[str]] = defaultdict(set)
    symbol_paths: dict[str, set[str]] = defaultdict(set)
    for path, items in evidence_by_path.items():
        for item in items:
            symbol = str(item["locator"]["symbol"])
            if symbol != "<module>":
                symbol_paths[symbol].add(path)
            for match in _CALL_NAME.finditer(str(item["content"])):
                name = match.group(1)
                if name not in _PYTHON_KEYWORDS:
                    call_name_paths[name].add(path)

    artifacts = sorted(bundle.repo_artifacts, key=lambda item: _stable_path_key(item.path))
    read_ids = {artifact.path: f"F{index:04d}" for index, artifact in enumerate(artifacts, 1)}
    files: list[_RepoFile] = []
    for artifact in artifacts:
        path_behaviors = by_path.get(artifact.path, [])
        symbols = _ordered_symbols(path_behaviors)
        matched_sections = _match_document_sections(
            document,
            sections,
            artifact,
            {
                symbol
                for symbol, paths in symbol_paths.items()
                if paths == {artifact.path}
            }
            | {
                alias
                for symbol in symbols
                for alias in _qualified_symbol_aliases(artifact.path, symbol)
            },
            {
                name
                for name, paths in call_name_paths.items()
                if paths == {artifact.path}
            },
        )
        section_titles = tuple(
            dict.fromkeys(section.title for section in matched_sections)
        )
        files.append(
            _RepoFile(
                read_id=read_ids[artifact.path],
                path=artifact.path,
                artifact_kind=artifact.kind,
                behavior_count=len(path_behaviors),
                symbols=symbols,
                sections=section_titles,
                section_orders=tuple(section.order for section in matched_sections),
            )
        )

    direct = [item for item in files if item.sections]
    direct_order = _round_robin_sections(direct)
    direct_paths = {item.path for item in direct}
    neighbor_paths = _neighbor_paths(graph["edges"], direct_paths)
    neighbors = sorted(
        [item for item in files if item.path in neighbor_paths and item.path not in direct_paths],
        key=_file_rank,
    )
    remaining = sorted(
        [
            item
            for item in files
            if item.path not in direct_paths and item.path not in neighbor_paths
        ],
        key=_file_rank,
    )
    candidates = [*direct_order, *neighbors, *remaining]
    root_symbols = _select_repo_root_symbols(graph, document)
    selected: list[dict[str, Any]] = []
    for item in candidates:
        entry = _directory_entry(item)
        candidate_text = _render_repo_entries([*selected, entry])
        if count_tokens(candidate_text) > DIRECTORY_TOKEN_BUDGET:
            break
        selected.append(entry)
    rendered = _render_repo_entries(selected)
    token_count = count_tokens(rendered)
    if token_count > DIRECTORY_TOKEN_BUDGET:
        raise DirectoryError("Repo directory exceeds the fixed token budget")
    return {
        "input_id": bundle.input_id,
        "benchmark": bundle.benchmark,
        "directory_type": "repo",
        "token_budget": DIRECTORY_TOKEN_BUDGET,
        "token_count": token_count,
        "entries": selected,
        "query_index": [
            {
                "read_id": read_ids[item.path],
                "path": item.path,
                "root_symbols": list(root_symbols.get(item.path, ())),
            }
            for item in artifacts
        ],
    }


def _select_repo_root_symbols(
    graph: dict[str, Any], document: str
) -> dict[str, tuple[str, ...]]:
    behaviors_by_endpoint: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for behavior in graph["behaviors"]:
        behaviors_by_endpoint[
            (str(behavior["path"]), str(behavior["symbol"]))
        ].append(behavior)

    symbol_alias_endpoints: dict[str, set[tuple[str, str]]] = defaultdict(set)
    scope_alias_endpoints: dict[str, set[tuple[str, str]]] = defaultdict(set)
    leaf_endpoints: dict[str, set[tuple[str, str]]] = defaultdict(set)
    call_endpoints: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for endpoint in behaviors_by_endpoint:
        if endpoint[1] != "<module>":
            symbol_alias_endpoints[endpoint[1]].add(endpoint)
            for alias in _qualified_symbol_aliases(*endpoint):
                symbol_alias_endpoints[alias].add(endpoint)
            symbol_parts = endpoint[1].split(".")
            for end in range(1, len(symbol_parts)):
                scope = ".".join(symbol_parts[:end])
                scope_alias_endpoints[scope].add(endpoint)
                for alias in _qualified_symbol_aliases(endpoint[0], scope):
                    scope_alias_endpoints[alias].add(endpoint)
            leaf_endpoints[endpoint[1].rsplit(".", 1)[-1]].add(endpoint)
    evidence_term_endpoints: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for item in graph["evidence"]:
        locator = item["locator"]
        endpoint = (str(locator["path"]), str(locator["symbol"]))
        if endpoint not in behaviors_by_endpoint:
            continue
        content = str(item["content"])
        for match in _CALL_NAME.finditer(content):
            name = match.group(1)
            if name not in _PYTHON_KEYWORDS:
                call_endpoints[name].add(endpoint)
        for match in _IDENTIFIER.finditer(content):
            term = match.group(1)
            if term not in _PYTHON_KEYWORDS:
                evidence_term_endpoints[term.casefold()].add(endpoint)

    matched: set[tuple[str, str]] = set()
    for alias, endpoints in symbol_alias_endpoints.items():
        if _literal_pattern(alias).search(document):
            matched.update(_unambiguous_endpoints(endpoints))
    for alias, endpoints in scope_alias_endpoints.items():
        if _literal_pattern(alias).search(document):
            matched.add(_representative_endpoint(endpoints, behaviors_by_endpoint))
    for leaf, endpoints in leaf_endpoints.items():
        if _explicit_code_name_pattern(leaf).search(document):
            matched.update(_unambiguous_endpoints(endpoints))
    for name, endpoints in call_endpoints.items():
        escaped = re.escape(name)
        patterns = (
            re.compile(rf"(?<![\w.]){escaped}\s*\(", re.IGNORECASE),
            re.compile(
                rf"`(?:[\w.]+\.)?{escaped}(?:\s*\([^`\n]*\))?`",
                re.IGNORECASE,
            ),
        )
        if any(pattern.search(document) for pattern in patterns):
            matched.update(_unambiguous_endpoints(endpoints))
    for term in _document_code_terms(document):
        endpoints = evidence_term_endpoints.get(term.casefold(), set())
        if len(endpoints) == 1:
            matched.update(endpoints)

    by_path: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    for (path, symbol), behaviors in behaviors_by_endpoint.items():
        start = min(int(item["symbol_lines"][0]) for item in behaviors)
        by_path[path].append((symbol, len(behaviors), start))
    result: dict[str, tuple[str, ...]] = {}
    for path, values in by_path.items():
        direct = [item for item in values if (path, item[0]) in matched]
        selected = direct or [
            min(
                values,
                key=lambda item: (-item[1], item[2], item[0].casefold(), item[0]),
            )
        ]
        result[path] = tuple(
            item[0]
            for item in sorted(
                selected,
                key=lambda item: (
                    -item[1],
                    item[2],
                    item[0].casefold(),
                    item[0],
                ),
            )
        )
    return result


def select_minimal_repo_roots(
    bundle: VisibleBundle,
    graph: dict[str, Any],
    *,
    count_tokens: TokenCounter | None = None,
) -> tuple[
    dict[str, tuple[str, ...]],
    dict[tuple[str, str], tuple[str, ...]],
    dict[tuple[str, str], str],
]:
    """Choose a small, document-anchored root set independently of read budgets."""

    if bundle.task_document is None:
        raise DirectoryError("Repo root selection requires the complete task document")
    document = bundle.task_document.content
    counter = count_tokens or _default_token_count
    sections = _document_sections(document)
    legacy = _select_repo_root_symbols(graph, document)
    behaviors_by_endpoint: dict[
        tuple[str, str], list[dict[str, Any]]
    ] = defaultdict(list)
    for behavior in graph["behaviors"]:
        behaviors_by_endpoint[
            (str(behavior["path"]), str(behavior["symbol"]))
        ].append(behavior)
    evidence_by_id = {
        str(item["evidence_id"]): item for item in graph["evidence"]
    }
    contexts_by_endpoint: dict[
        tuple[str, str], list[dict[str, Any]]
    ] = defaultdict(list)
    for context in graph["source_contexts"]:
        contexts_by_endpoint[
            (str(context["path"]), str(context["symbol"]))
        ].append(context)

    code_term_sections = _document_code_term_sections(document, sections)
    evidence_term_endpoints: dict[str, set[tuple[str, str]]] = defaultdict(set)
    call_name_endpoints: dict[str, set[tuple[str, str]]] = defaultdict(set)
    endpoint_term_evidence: dict[
        tuple[str, str], dict[str, set[str]]
    ] = defaultdict(lambda: defaultdict(set))
    endpoint_call_evidence: dict[
        tuple[str, str], dict[str, set[str]]
    ] = defaultdict(lambda: defaultdict(set))
    for evidence_id, evidence in evidence_by_id.items():
        locator = evidence["locator"]
        endpoint = (str(locator["path"]), str(locator["symbol"]))
        if endpoint not in behaviors_by_endpoint:
            continue
        content = str(evidence["content"])
        for match in _CALL_NAME.finditer(content):
            name = match.group(1)
            if name.casefold() in _PYTHON_KEYWORDS_FOLDED:
                continue
            call_name_endpoints[name.casefold()].add(endpoint)
            endpoint_call_evidence[endpoint][name.casefold()].add(evidence_id)
        for match in _IDENTIFIER.finditer(content):
            term = match.group(1).casefold()
            if term in _PYTHON_KEYWORDS_FOLDED:
                continue
            for candidate in {term, term.rsplit(".", 1)[-1]}:
                evidence_term_endpoints[candidate].add(endpoint)
                endpoint_term_evidence[endpoint][candidate].add(evidence_id)

    selected_by_path: dict[str, tuple[str, ...]] = {}
    module_evidence: dict[tuple[str, str], tuple[str, ...]] = {}
    document_section_by_endpoint: dict[tuple[str, str], str] = {}
    for path, legacy_symbols in legacy.items():
        candidates: list[_RootCandidate] = []
        for order, symbol in enumerate(legacy_symbols):
            endpoint = (path, symbol)
            strong_sections = _strong_symbol_section_matches(
                document, sections, path, symbol
            )
            scope_sections = _scope_section_matches(
                document, sections, path, symbol
            )
            evidence_sections: set[int] = set()
            matched_evidence: set[str] = set()
            for name, endpoints in call_name_endpoints.items():
                if endpoints != {endpoint}:
                    continue
                escaped = re.escape(name)
                call_sections = _matched_section_orders(
                    document,
                    sections,
                    (
                        re.compile(
                            rf"(?<![\w.]){escaped}\s*\(",
                            re.IGNORECASE,
                        ),
                        re.compile(
                            rf"`(?:[\w.]+\.)?{escaped}"
                            rf"(?:\s*\([^`\n]*\))?`",
                            re.IGNORECASE,
                        ),
                    ),
                )
                if call_sections:
                    evidence_sections.update(call_sections)
                    matched_evidence.update(
                        endpoint_call_evidence[endpoint][name]
                    )
            for term, section_orders in code_term_sections.items():
                endpoints = evidence_term_endpoints.get(term, set())
                if endpoints != {endpoint}:
                    continue
                evidence_sections.update(section_orders)
                matched_evidence.update(endpoint_term_evidence[endpoint][term])
            if symbol == "<module>" and not matched_evidence:
                continue
            # Keep only the strongest available anchor. Mixing a parent-scope
            # mention into an exact Symbol match can attach an unrelated, deeper
            # subsection to the Root.
            matched_sections = (
                strong_sections or evidence_sections or scope_sections
            )
            priority = (
                0
                if strong_sections
                else 1
                if evidence_sections
                else 2
                if scope_sections
                else 3
            )
            source_cost = sum(
                counter(str(context["source"]))
                for context in contexts_by_endpoint.get(endpoint, [])
            )
            if symbol == "<module>" and matched_evidence:
                source_cost = sum(
                    counter(str(evidence_by_id[evidence_id]["content"]))
                    for evidence_id in matched_evidence
                )
            candidates.append(
                _RootCandidate(
                    symbol=symbol,
                    priority=priority,
                    section_orders=frozenset(matched_sections),
                    evidence_ids=frozenset(matched_evidence),
                    source_tokens=max(1, source_cost),
                    legacy_order=order,
                )
            )

        directly_matched = [item for item in candidates if item.priority < 3]
        if directly_matched:
            # A section can name several independent Symbols.  Treating the
            # whole section as one coverage item drops those explicit anchors.
            # The legacy matcher has already collapsed an ambiguous parent
            # scope to one representative; retain every remaining direct root.
            chosen = sorted(
                directly_matched,
                key=lambda item: (
                    item.priority,
                    -len(item.section_orders),
                    item.symbol == "<module>",
                    item.source_tokens,
                    item.legacy_order,
                ),
            )
        else:
            ordinary = [
                item for item in candidates if item.symbol != "<module>"
            ]
            chosen = (ordinary or candidates)[:1]

        selected_by_path[path] = tuple(item.symbol for item in chosen)
        for item in chosen:
            precise_sections = _exact_symbol_section_matches(
                document, sections, path, item.symbol
            )
            section = _most_specific_section(
                sections, precise_sections, symbol=item.symbol
            )
            # `Doc:` is a precision signal. Evidence terms, enclosing scopes,
            # and an unqualified method leaf such as ``__init__`` may rank a
            # useful Root but cannot claim a direct section-level match.
            if section is not None:
                document_section_by_endpoint[(path, item.symbol)] = section.title
            if item.symbol == "<module>" and item.evidence_ids:
                module_evidence[(path, item.symbol)] = tuple(
                    sorted(item.evidence_ids)
                )
    return selected_by_path, module_evidence, document_section_by_endpoint


def _exact_symbol_section_matches(
    document: str,
    sections: list[_Section],
    path: str,
    symbol: str,
) -> set[int]:
    """Find sections that name this Symbol, without ambiguous leaf aliases."""

    if symbol == "<module>":
        return set()
    patterns = [_literal_pattern(symbol)]
    pure = PurePosixPath(path)
    module_parts = [*pure.parts[:-1], pure.stem]
    if module_parts and module_parts[-1] == "__init__":
        module_parts.pop()
    if module_parts and all(part.isidentifier() for part in module_parts):
        variants = [module_parts]
        if module_parts[0] == "src" and len(module_parts) > 1:
            variants.append(module_parts[1:])
        patterns.extend(
            _literal_pattern(f"{'.'.join(parts)}.{symbol}")
            for parts in variants
        )
    if "." not in symbol:
        patterns.append(_explicit_code_name_pattern(symbol))
    return _matched_section_orders(document, sections, patterns)


def _most_specific_section(
    sections: list[_Section],
    section_orders: Iterable[int],
    *,
    symbol: str,
) -> _Section | None:
    """Return the narrowest deepest section that actually matched a Root."""

    orders = set(section_orders)
    matched = [item for item in sections if item.order in orders]
    if not matched:
        return None
    symbol_name = symbol.casefold()
    short_name = symbol.rsplit(".", 1)[-1].casefold()
    return min(
        matched,
        key=lambda item: (
            0
            if symbol_name in item.title.casefold()
            or re.search(
                rf"(?<!\w){re.escape(short_name)}(?!\w)",
                item.title,
                re.IGNORECASE,
            )
            else 1,
            -item.level,
            item.end - item.start,
            item.order,
        ),
    )


def _strong_symbol_section_matches(
    document: str,
    sections: list[_Section],
    path: str,
    symbol: str,
) -> set[int]:
    if symbol == "<module>":
        return set()
    patterns = [_literal_pattern(symbol)]
    patterns.extend(
        _literal_pattern(alias)
        for alias in sorted(_qualified_symbol_aliases(path, symbol))
    )
    patterns.append(_explicit_code_name_pattern(symbol.rsplit(".", 1)[-1]))
    return _matched_section_orders(document, sections, patterns)


def _scope_section_matches(
    document: str,
    sections: list[_Section],
    path: str,
    symbol: str,
) -> set[int]:
    if symbol == "<module>":
        return set()
    patterns: list[re.Pattern[str]] = []
    symbol_parts = symbol.split(".")
    for end in range(1, len(symbol_parts)):
        scope = ".".join(symbol_parts[:end])
        patterns.append(_literal_pattern(scope))
        patterns.extend(
            _literal_pattern(alias)
            for alias in sorted(_qualified_symbol_aliases(path, scope))
        )
    return _matched_section_orders(document, sections, patterns)


def _document_code_term_sections(
    document: str, sections: list[_Section]
) -> dict[str, set[int]]:
    result: dict[str, set[int]] = defaultdict(set)
    for span in _BACKTICK_CODE.finditer(document):
        section = next(
            (item for item in sections if item.start <= span.start() < item.end),
            sections[-1],
        )
        for match in _IDENTIFIER.finditer(span.group(1)):
            term = match.group(1)
            if term in _PYTHON_KEYWORDS:
                continue
            result[term.casefold()].add(section.order)
            result[term.rsplit(".", 1)[-1].casefold()].add(section.order)
    return result


def _matched_section_orders(
    document: str,
    sections: list[_Section],
    patterns: Iterable[re.Pattern[str]],
) -> set[int]:
    result: set[int] = set()
    for pattern in patterns:
        for match in pattern.finditer(document):
            section = next(
                (item for item in sections if item.start <= match.start() < item.end),
                sections[-1],
            )
            result.add(section.order)
    return result


def _document_sections(document: str) -> list[_Section]:
    headings = list(_HEADING.finditer(document))
    if not headings:
        return [_Section("Document", 0, len(document), 0, 0)]
    sections: list[_Section] = []
    if headings[0].start() > 0:
        sections.append(_Section("Document", 0, headings[0].start(), 0, 0))
    for index, heading in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(document)
        sections.append(
            _Section(
                heading.group("title").strip(),
                heading.start(),
                end,
                len(sections),
                len(heading.group("marks")),
            )
        )
    return sections


def _match_document_sections(
    document: str,
    sections: list[_Section],
    artifact: RepoArtifact,
    match_symbols: set[str],
    call_names: set[str],
) -> list[_Section]:
    patterns = [_literal_pattern(artifact.path, path=True)]
    for symbol in sorted(match_symbols):
        patterns.append(_literal_pattern(symbol))
    for name in sorted(call_names):
        escaped = re.escape(name)
        patterns.append(re.compile(rf"(?<![\w.]){escaped}\s*\(", re.IGNORECASE))
        patterns.append(re.compile(rf"`(?:[\w.]+\.)?{escaped}(?:\s*\([^`\n]*\))?`", re.IGNORECASE))
    matched_orders: set[int] = set()
    for pattern in patterns:
        for match in pattern.finditer(document):
            section = next(
                (item for item in sections if item.start <= match.start() < item.end),
                sections[-1],
            )
            matched_orders.add(section.order)
    return [item for item in sections if item.order in matched_orders]


def _literal_pattern(value: str, *, path: bool = False) -> re.Pattern[str]:
    boundary = r"[\w/-]" if path else r"[\w.]"
    return re.compile(
        rf"(?<!{boundary}){re.escape(value)}(?!{boundary})",
        re.IGNORECASE,
    )


def _qualified_symbol_aliases(path: str, symbol: str) -> set[str]:
    pure = PurePosixPath(path)
    module_parts = [*pure.parts[:-1], pure.stem]
    if module_parts and module_parts[-1] == "__init__":
        module_parts.pop()
    if not module_parts or any(not part.isidentifier() for part in module_parts):
        return set()
    variants = [module_parts]
    if module_parts[0] == "src" and len(module_parts) > 1:
        variants.append(module_parts[1:])
    leaf = symbol.rsplit(".", 1)[-1]
    aliases = {
        f"{'.'.join(parts)}.{symbol}"
        for parts in variants
        if parts
    }
    for parts in variants:
        aliases.update(f"{part}.{leaf}" for part in parts)
    return aliases


def _explicit_code_name_pattern(name: str) -> re.Pattern[str]:
    escaped = re.escape(name)
    return re.compile(
        rf"(?:`[^`\n]*?(?<![\w.]){escaped}(?:\s*\(|(?![\w.]))|"
        rf"(?<![\w.]){escaped}\s*\()",
        re.IGNORECASE,
    )


def _unambiguous_endpoints(
    endpoints: set[tuple[str, str]],
) -> set[tuple[str, str]]:
    if len(endpoints) == 1:
        return set(endpoints)
    return set()


def _representative_endpoint(
    endpoints: set[tuple[str, str]],
    behaviors_by_endpoint: dict[tuple[str, str], list[dict[str, Any]]],
) -> tuple[str, str]:
    """Choose one observable member for an explicitly named parent scope."""

    return min(
        endpoints,
        key=lambda endpoint: (
            -len(behaviors_by_endpoint[endpoint]),
            min(
                int(item["symbol_lines"][0])
                for item in behaviors_by_endpoint[endpoint]
            ),
            endpoint[1].casefold(),
            endpoint[1],
        ),
    )


def _document_code_terms(document: str) -> set[str]:
    terms: set[str] = set()
    for span in _BACKTICK_CODE.findall(document):
        for match in _IDENTIFIER.finditer(span):
            term = match.group(1)
            if term not in _PYTHON_KEYWORDS:
                terms.add(term)
                terms.add(term.rsplit(".", 1)[-1])
    return terms


def _ordered_symbols(behaviors: Iterable[dict[str, Any]]) -> tuple[str, ...]:
    ordered = sorted(
        behaviors,
        key=lambda item: (
            int(item["symbol_lines"][0]),
            int(item["symbol_lines"][1]),
            str(item["symbol"]).casefold(),
            str(item["symbol"]),
        ),
    )
    return tuple(dict.fromkeys(str(item["symbol"]) for item in ordered))


def _round_robin_sections(files: list[_RepoFile]) -> list[_RepoFile]:
    ranked = sorted(files, key=_file_rank)
    section_orders = sorted({order for item in ranked for order in item.section_orders})
    buckets = {
        order: [item for item in ranked if order in item.section_orders]
        for order in section_orders
    }
    selected: list[_RepoFile] = []
    seen: set[str] = set()
    while any(buckets.values()):
        for order in section_orders:
            bucket = buckets[order]
            while bucket and bucket[0].path in seen:
                bucket.pop(0)
            if bucket:
                item = bucket.pop(0)
                selected.append(item)
                seen.add(item.path)
    selected.extend(item for item in ranked if item.path not in seen)
    return selected


def _neighbor_paths(edges: list[dict[str, Any]], direct_paths: set[str]) -> set[str]:
    result: set[str] = set()
    for edge in edges:
        source = str(edge["from"]["path"])
        target = str(edge["to"]["path"])
        if source in direct_paths and target not in direct_paths:
            result.add(target)
        if target in direct_paths and source not in direct_paths:
            result.add(source)
    return result


def _file_rank(item: _RepoFile) -> tuple[Any, ...]:
    return (
        _ARTIFACT_ORDER[item.artifact_kind],
        -item.behavior_count,
        *_stable_path_key(item.path),
    )


def _stable_path_key(path: str) -> tuple[str, str]:
    return path.casefold(), path


def _directory_entry(item: _RepoFile) -> dict[str, Any]:
    shown = item.symbols[:MAX_HINT_SYMBOLS]
    if shown:
        suffix = "; ".join(shown)
        if len(item.symbols) > len(shown):
            suffix += "; ..."
        hint = f"{len(item.symbols)} linked symbols: {suffix}"
    else:
        hint = "no observable symbols"
    return {
        "read_id": item.read_id,
        "path": item.path,
        "related_document_sections": list(item.sections[:MAX_SECTION_HINTS]),
        "code_hints": hint,
    }


def _render_repo_entries(entries: list[dict[str, Any]]) -> str:
    lines = ["Read ID | File | Related document sections | Code hints"]
    for entry in entries:
        sections = "; ".join(entry["related_document_sections"]) or "-"
        lines.append(
            " | ".join(
                (
                    _table_text(entry["read_id"]),
                    _table_text(entry["path"]),
                    _table_text(sections),
                    _table_text(entry["code_hints"]),
                )
            )
        )
    return "\n".join(lines) + "\n"


def _table_text(value: Any) -> str:
    return str(value).replace("|", "¦").replace("\r", " ").replace("\n", " ")


def _default_token_count(text: str) -> int:
    encoding = tiktoken.get_encoding(DIRECTORY_ENCODING)
    return len(encoding.encode(text, disallowed_special=()))
