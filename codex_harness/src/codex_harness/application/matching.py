"""Harness-only adaptation of BEG's matching rules; no BEG module mutation.

BEG's private matchers are reused here behind one compatibility boundary.
Codex payload handling and exclusion of navigation fallbacks stay local.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any

from ..beg.behavior_atomization import _trace_demand_spans, _trace_signal, _trace_tool_operation
from ..beg.behavior_directory import (
    _CALL_NAME, _IDENTIFIER, _PYTHON_KEYWORDS, _document_code_terms,
    _literal_pattern, _qualified_symbol_aliases,
    _select_repo_root_symbols,
)


_CHINESE_TEST_RUN = re.compile(r"(?:运行|执行)\s+[^。\n]{0,160}(?:tests?[/\\]|test_[\w.-]+\.py)", re.IGNORECASE)


def demand_spans(content: str) -> list[tuple[int, int]]:
    return _trace_demand_spans(content)


def trace_signal(content: str, *, tool_name: str | None = None) -> dict[str, list[str]]:
    # Preserve the original in storage; only BEG's input representation is adapted.
    name = "Edit" if tool_name and tool_name.rsplit("__", 1)[-1] == "apply_patch" else tool_name
    locator = {"event_type": "tool_exchange" if name else "user_prompt", "tool_name": name}
    invocation = content.split("\n\nTool result:", 1)[0]
    if invocation.startswith("Tool invocation:\n"):
        invocation = invocation.split("\n", 1)[1]
    adapted = f"Tool invocation:\n{invocation}" if name else content
    signal = _trace_signal({"content": adapted, "locator": locator})
    operations = set(signal.operations)
    if name:
        try:
            payload = json.loads(invocation)
        except (json.JSONDecodeError, TypeError):
            payload = None
        if isinstance(payload, dict):
            command = payload.get("command", payload.get("cmd"))
            if isinstance(command, str):
                operations = {_trace_tool_operation({"content": command, "locator": locator})}
    elif _CHINESE_TEST_RUN.search(content):
        operations.add("verify")
    return {"operations": sorted(operations), "objects": sorted(signal.objects)}


def literal_match(text: str, value: str, *, path: bool = False) -> bool:
    return bool(value and _literal_pattern(value, path=path).search(text.replace("\\", "/")))


def explicit_symbol_match(text: str, symbol: str) -> bool:
    """Plain words need code markup or call syntax; qualified names are explicit."""
    pattern = rf'(?<![\w.]){re.escape(symbol)}(?!\w|\.\w)'
    if ('.' in symbol or '_' in symbol or symbol != symbol.lower()) and re.search(pattern, text):
        return True
    if re.search(rf"(?<![\w.]){re.escape(symbol)}\s*\(", text):
        return True
    # Match paired spans, never restart at a closing backtick and consume prose.
    return any(literal_match(span, symbol) for span in re.findall(r'`([^`\n]+)`', text))


def source_line_paths(requirement: dict, sources: list[dict], paths: list[str]) -> dict:
    """Retain file anchors on the quoted source lines, without expanding the demand."""
    known = {s['id']: s for s in sources}
    anchors = {}
    for ref in requirement['refs']:
        source = known[ref['source_id']]
        text = source['content']
        start = text.rfind('\n', 0, ref['start']) + 1
        end = text.find('\n', max(ref['start'], ref['end'] - 1))
        end = len(text) if end < 0 else end
        context = text[start:end]
        if source['kind'] == 'repo' and source.get('path') in paths:
            anchors.setdefault(source['path'], {'source': ref['source'], 'content': ref['content']})
        for path in paths:
            if literal_match(context, path, path=True):
                anchors.setdefault(path, {'source': f"{source['label']}@{text.count(chr(10), 0, start) + 1}",
                                          'content': context})
    return anchors


def section_navigation(requirement: dict[str, Any], sources: list[dict[str, Any]],
                       paths: list[str]) -> list[dict[str, str]]:
    """Use enclosing Markdown headings only as labeled navigation, not evidence."""
    known = {source['id']: source for source in sources}
    headings = []
    for ref in requirement['refs']:
        source = known[ref['source_id']]
        if source['kind'] != 'plan':
            continue
        stack = []
        fence = None
        position = 0
        for line in source['content'].splitlines(keepends=True):
            if position > ref['start']:
                break
            marker = re.match(r'^\s{0,3}(`{3,}|~{3,})', line)
            if marker:
                token = marker.group(1)
                if fence is None:
                    fence = token
                elif token[0] == fence[0] and len(token) >= len(fence):
                    fence = None
            elif fence is None:
                heading = re.match(r'^(#{1,6})\s+(.+?)\s*#*\s*$', line)
                if heading:
                    level, title = len(heading.group(1)), heading.group(2)
                    while stack and stack[-1][0] >= level:
                        stack.pop()
                    stack.append((level, title, source['content'].count('\n', 0, position) + 1))
            position += len(line)
        headings.extend((title, f"{source['path']}@{line}") for _, title, line in stack)

    def words(text: str) -> set[str]:
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        return {word.casefold() for word in re.findall(r'[A-Za-z][a-zA-Z0-9]*', text) if len(word) > 1}

    candidates = []
    for path in sorted(paths):
        stem = path.replace('\\', '/').rsplit('/', 1)[-1].rsplit('.', 1)[0]
        parts = words(stem)
        for title, source in headings:
            if len(parts) >= 2 and parts <= words(title):
                candidates.append({'path': path, 'basis': f'{source}: {title} → file name'})
                break
    return candidates


def direct_repo_roots(graph: dict[str, Any], document: str) -> dict[str, tuple[str, ...]]:
    """Filter BEG navigation candidates to explicit source-supported anchors."""
    candidates = _select_repo_root_symbols(graph, document)
    endpoints = {(b["path"], b["symbol"]) for b in graph["behaviors"]}
    aliases: dict[str, set] = defaultdict(set)
    leaves: dict[str, set] = defaultdict(set)
    scopes: dict[str, set] = defaultdict(set)
    for path, symbol in endpoints:
        if symbol == "<module>":
            continue
        for alias in [symbol, *_qualified_symbol_aliases(path, symbol)]:
            aliases[alias].add((path, symbol))
        leaves[symbol.rsplit(".", 1)[-1]].add((path, symbol))
        parts = symbol.split(".")
        for end in range(1, len(parts)):
            scope = ".".join(parts[:end])
            for alias in [scope, *_qualified_symbol_aliases(path, scope)]:
                scopes[alias].add((path, symbol))
    terms: dict[str, set] = defaultdict(set)
    calls: dict[str, set] = defaultdict(set)
    for item in graph["evidence"]:
        endpoint = item["locator"]["path"], item["locator"]["symbol"]
        if endpoint not in endpoints:
            continue
        for match in _IDENTIFIER.finditer(item["content"]):
            term = match.group(1)
            if term not in _PYTHON_KEYWORDS:
                terms[term.casefold()].add(endpoint)
        for match in _CALL_NAME.finditer(item["content"]):
            if match.group(1) not in _PYTHON_KEYWORDS:
                calls[match.group(1)].add(endpoint)
    explicit = set()
    for alias, matches in aliases.items():
        if len(matches) == 1 and explicit_symbol_match(document, alias):
            explicit.update(matches)
    for leaf, matches in leaves.items():
        if len(matches) == 1 and explicit_symbol_match(document, leaf):
            explicit.update(matches)
    for scope, matches in scopes.items():
        if len({path for path, symbol in matches}) == 1 and explicit_symbol_match(document, scope):
            explicit.update(matches)
    for term in _document_code_terms(document):
        matches = terms.get(term.casefold(), set())
        if len(matches) == 1:
            explicit.update(matches)
    for name, matches in calls.items():
        if len(matches) == 1 and explicit_symbol_match(document, name):
            explicit.update(matches)
    for path, symbol in sorted(explicit):
        if symbol not in candidates.get(path, ()):
            candidates[path] = (*candidates.get(path, ()), symbol)
    result = {}
    for path, symbols in candidates.items():
        selected = tuple(symbol for symbol in symbols if (path, symbol) in explicit)
        if selected:
            result[path] = selected
    return result
