"""Small standard-library language adapters used by the EBG repository graph.

Python keeps its existing AST and reaching-definition implementation.  These
adapters give Go, TypeScript and JavaScript the same basic symbols and local
call edges without making the harness depend on an external parser package.
They are deliberately conservative: an uncertain declaration or call is
omitted instead of being presented as a verified relation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath

from .core.model import BehaviorCandidate, RepoArtifact, SymbolSpan


_EXTENSIONS = {
    ".go": "go",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
}
_CALL_KEYWORDS = {
    "if", "for", "switch", "catch", "while", "return", "typeof", "sizeof",
    "function", "class", "interface", "new", "make", "go", "defer",
}


@dataclass(frozen=True, slots=True)
class GenericDeclaration:
    symbol: str
    line_start: int
    line_end: int
    kind: str


def language_for_path(path: str) -> str | None:
    return _EXTENSIONS.get(PurePosixPath(path).suffix.casefold())


def is_supported_source(artifact: RepoArtifact) -> bool:
    return language_for_path(artifact.path) is not None


def declarations(path: str, content: str) -> list[GenericDeclaration]:
    language = language_for_path(path)
    if language is None:
        return []
    lines = content.splitlines()
    if not lines:
        return []
    result: list[GenericDeclaration] = []
    for index, line in enumerate(lines, start=1):
        match = _declaration_match(language, line)
        if match is None:
            continue
        symbol, kind = match
        end = _declaration_end(content, lines, index)
        result.append(GenericDeclaration(symbol, index, end, kind))
    # A method may be discovered after its enclosing class.  Keep the source
    # order and discard exact duplicates from overloaded declaration patterns.
    unique: dict[tuple[str, int, int], GenericDeclaration] = {}
    for item in result:
        unique.setdefault((item.symbol, item.line_start, item.line_end), item)
    ordered = sorted(unique.values(), key=lambda item: (item.line_start, item.line_end, item.symbol))
    types = [item for item in ordered if item.kind == "type"]
    qualified: list[GenericDeclaration] = []
    for item in ordered:
        if language in {"typescript", "javascript"} and item.kind == "method" and "." not in item.symbol:
            parents = [parent for parent in types if parent.line_start <= item.line_start <= parent.line_end]
            if parents:
                parent = min(parents, key=lambda value: value.line_end - value.line_start)
                item = GenericDeclaration(
                    f"{parent.symbol}.{item.symbol}", item.line_start, item.line_end, item.kind
                )
        qualified.append(item)
    return qualified


def generic_symbol_spans(path: str, content: str) -> list[SymbolSpan]:
    return [
        SymbolSpan(path, item.symbol, item.line_start, item.line_end)
        for item in declarations(path, content)
    ]


def generic_source_drafts(artifact: RepoArtifact) -> list[tuple[str, int, int, str]]:
    """Return module and symbol excerpts for evidence intake."""
    lines = artifact.content.splitlines(keepends=True)
    if not lines:
        return []
    drafts = [("<module>", 1, len(lines), artifact.content)]
    for item in declarations(artifact.path, artifact.content):
        text = "".join(lines[item.line_start - 1 : item.line_end])
        if text:
            drafts.append((item.symbol, item.line_start, item.line_end, text))
    return drafts


def generic_behavior_candidates(
    artifact: RepoArtifact,
    evidence_index: object,
) -> list[BehaviorCandidate]:
    """Create conservative function/class behavior anchors for non-Python code."""
    candidates: list[BehaviorCandidate] = []
    lines = artifact.content.splitlines()
    for item in declarations(artifact.path, artifact.content):
        if item.kind not in {"function", "method", "arrow"}:
            continue
        evidence_ids = evidence_index.ids_for_ranges(
            artifact.path, item.symbol, [(item.line_start, item.line_end)]
        )
        if not evidence_ids:
            continue
        result_line, result_kind = _result_anchor(lines, item)
        candidates.append(
            BehaviorCandidate(
                path=artifact.path,
                symbol=item.symbol,
                result_kind=result_kind,
                result_line=result_line,
                evidence_ids=evidence_ids,
            )
        )
    return candidates


def generic_call_sites(path: str, content: str) -> list[tuple[str, str, int]]:
    """Return conservative ``(source_symbol, target_name, line)`` call sites."""
    lines = content.splitlines()
    items = [item for item in declarations(path, content) if item.kind in {"function", "method", "arrow"}]
    definitions = {item.symbol for item in items}
    by_name: dict[str, list[str]] = {}
    for symbol in definitions:
        by_name.setdefault(symbol.rsplit(".", 1)[-1], []).append(symbol)
    calls: list[tuple[str, str, int]] = []
    for item in items:
        for line_number in range(item.line_start, min(item.line_end, len(lines)) + 1):
            line = _strip_strings_and_comments(lines[line_number - 1])
            for match in re.finditer(r"\b([A-Za-z_$][\w$]*)\s*\(", line):
                name = match.group(1)
                if name in _CALL_KEYWORDS or name == item.symbol.rsplit(".", 1)[-1]:
                    continue
                targets = by_name.get(name, [])
                if len(targets) == 1 and targets[0] != item.symbol:
                    calls.append((item.symbol, targets[0], line_number))
    return sorted(set(calls), key=lambda item: (item[0], item[2], item[1]))


def _declaration_match(language: str, line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith(("//", "/*", "*", "#")):
        return None
    if language == "go":
        match = re.match(r"func\s*(?:\([^)]*\*?([A-Za-z_]\w*)\)\s*)?([A-Za-z_]\w*)\s*\(", stripped)
        if match:
            receiver, name = match.groups()
            return (f"{receiver}.{name}" if receiver else name, "method" if receiver else "function")
        match = re.match(r"type\s+([A-Za-z_]\w*)\s+(struct|interface)\b", stripped)
        if match:
            return match.group(1), "type"
        return None
    match = re.match(r"(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)\s*\(", stripped)
    if match:
        return match.group(1), "function"
    match = re.match(r"(?:export\s+)?(?:abstract\s+)?class\s+([A-Za-z_$][\w$]*)\b", stripped)
    if match:
        return match.group(1), "type"
    match = re.match(r"(?:export\s+)?(?:interface|type|enum|namespace)\s+([A-Za-z_$][\w$]*)\b", stripped)
    if match:
        return match.group(1), "type"
    match = re.match(r"(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*=>", stripped)
    if match:
        return match.group(1), "arrow"
    # Methods are only accepted when they look like a declaration, avoiding
    # ordinary calls and control statements.
    match = re.match(r"(?:public|private|protected|static|async|get|set|readonly\s+)*\s*([A-Za-z_$][\w$]*)\s*\([^;{}]*\)\s*(?::[^{}]+)?\s*\{", stripped)
    if match and match.group(1) not in _CALL_KEYWORDS:
        return match.group(1), "method"
    return None


def _declaration_end(content: str, lines: list[str], start_line: int) -> int:
    raw_lines = content.splitlines(keepends=True)
    start_offset = sum(len(line) for line in raw_lines[: start_line - 1])
    opening = _find_open_brace(content, start_offset)
    if opening is None:
        return start_line
    depth = 0
    line = start_line
    state = "normal"
    quote = ""
    index = opening
    while index < len(content):
        char = content[index]
        next_char = content[index + 1] if index + 1 < len(content) else ""
        if state == "line_comment":
            if char == "\n":
                state = "normal"
                line += 1
            index += 1
            continue
        if state == "block_comment":
            if char == "*" and next_char == "/":
                state = "normal"
                index += 2
                continue
            if char == "\n":
                line += 1
            index += 1
            continue
        if state == "string":
            if char == "\\":
                index += 2
                continue
            if char == quote:
                state = "normal"
            if char == "\n":
                line += 1
            index += 1
            continue
        if char == "/" and next_char == "/":
            state = "line_comment"
            index += 2
            continue
        if char == "/" and next_char == "*":
            state = "block_comment"
            index += 2
            continue
        if char in "'\"`":
            state, quote = "string", char
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return line
        if char == "\n":
            line += 1
        index += 1
    return len(lines)


def _find_open_brace(content: str, offset: int) -> int | None:
    state = "normal"
    quote = ""
    index = offset
    while index < len(content):
        char = content[index]
        next_char = content[index + 1] if index + 1 < len(content) else ""
        if state == "line_comment":
            if char == "\n":
                state = "normal"
            index += 1
            continue
        if state == "block_comment":
            if char == "*" and next_char == "/":
                state = "normal"
                index += 2
                continue
            index += 1
            continue
        if state == "string":
            if char == "\\":
                index += 2
                continue
            if char == quote:
                state = "normal"
            index += 1
            continue
        if char == "/" and next_char == "/":
            state = "line_comment"
            index += 2
            continue
        if char == "/" and next_char == "*":
            state = "block_comment"
            index += 2
            continue
        if char in "'\"`":
            state, quote = "string", char
        elif char == "{":
            return index
        index += 1
    return None


def _result_anchor(lines: list[str], item: GenericDeclaration) -> tuple[int, str]:
    for line_number in range(item.line_start, min(item.line_end, len(lines)) + 1):
        line = lines[line_number - 1]
        if re.search(r"\b(?:return|panic)\b", line):
            return line_number, "return"
        if re.search(r"\b(?:throw|raise)\b", line):
            return line_number, "raise"
        if re.search(r"\byield\b", line):
            return line_number, "yield"
    return item.line_start, "output"


def _strip_strings_and_comments(line: str) -> str:
    line = re.sub(r"//.*$", "", line)
    line = re.sub(r"/\*.*?\*/", "", line)
    line = re.sub(r"(['\"`]).*?\1", "", line)
    return line
