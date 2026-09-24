"""Code locations and patch overlays shared by SilentSwap construction tools."""

from __future__ import annotations

import ast
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

class EvaluationError(RuntimeError):
    pass

def run(command: list[str], cwd: Path) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise EvaluationError(completed.stderr.strip() or "command failed")
    return completed.stdout

def load_swapped_overlay(sample: Path) -> dict[str, str]:
    patch = sample / "swap.patch"
    repository = sample / "repository"
    changed = []
    for line in run(
        ["git", "apply", "--numstat", str(patch.resolve())], sample
    ).splitlines():
        changed.append(line.split("\t", 2)[2])

    with tempfile.TemporaryDirectory() as directory:
        worktree = Path(directory)
        for relative in changed:
            source = repository / relative
            destination = worktree / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.exists():
                blob = subprocess.run(
                    ["git", "show", f"HEAD:{relative}"],
                    cwd=repository,
                    capture_output=True,
                    check=True,
                ).stdout
                destination.write_bytes(blob)
        run(["git", "apply", str(patch.resolve())], worktree)
        return {
            relative: (worktree / relative).read_text(encoding="utf-8")
            for relative in changed
            if (worktree / relative).is_file()
        }

def changed_lines_from_patch(patch: str) -> dict[str, set[int]]:
    changed: dict[str, set[int]] = {}
    current_path: str | None = None
    new_line: int | None = None
    hunk_deletion_anchor: int | None = None
    hunk_has_addition = False
    hunk_has_deletion = False
    hunk_pattern = re.compile(r"^@@ -\d+(?:,\d+)? \+(?P<start>\d+)(?:,\d+)? @@")

    def finish_hunk() -> None:
        if (
            current_path is not None
            and hunk_deletion_anchor is not None
            and hunk_has_deletion
            and not hunk_has_addition
        ):
            changed[current_path].add(max(1, hunk_deletion_anchor))

    for line in patch.splitlines():
        if line.startswith("+++ b/"):
            finish_hunk()
            current_path = line[len("+++ b/") :]
            changed.setdefault(current_path, set())
            new_line = None
            hunk_deletion_anchor = None
            hunk_has_addition = False
            hunk_has_deletion = False
            continue
        if line.startswith("@@ "):
            finish_hunk()
            match = hunk_pattern.match(line)
            if match is None or current_path is None:
                raise EvaluationError("invalid unified diff hunk")
            new_line = int(match.group("start"))
            hunk_deletion_anchor = None
            hunk_has_addition = False
            hunk_has_deletion = False
            continue
        if new_line is None:
            continue
        if line.startswith("\\ No newline at end of file"):
            continue
        if line.startswith("+"):
            if line[1:].strip():
                changed[current_path].add(new_line)
                hunk_has_addition = True
            new_line += 1
        elif line.startswith("-"):
            hunk_has_deletion = True
            if hunk_deletion_anchor is None:
                hunk_deletion_anchor = new_line
            continue
        else:
            new_line += 1

    finish_hunk()

    if not changed or any(not lines for lines in changed.values()):
        raise EvaluationError(
            "swap patch must add or replace at least one line per file"
        )
    return changed

def ranges_from_lines(lines: set[int]) -> list[dict[str, int]]:
    ordered = sorted(lines)
    ranges: list[dict[str, int]] = []
    start = end = ordered[0]
    for line in ordered[1:]:
        if line == end + 1:
            end = line
            continue
        ranges.append({"start": start, "end": end})
        start = end = line
    ranges.append({"start": start, "end": end})
    return ranges

def _ast_parents(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    return {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }

def _ancestor_nodes(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> list[ast.AST]:
    ancestors = []
    parent = parents.get(node)
    while parent is not None:
        ancestors.append(parent)
        parent = parents.get(parent)
    return ancestors

def _node_contains_line(node: ast.AST, line: int) -> bool:
    return hasattr(node, "lineno") and node.lineno <= line <= getattr(
        node, "end_lineno", node.lineno
    )

def _qualified_name(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> list[str]:
    named_nodes = [
        ancestor
        for ancestor in reversed(_ancestor_nodes(node, parents))
        if isinstance(ancestor, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    return [ancestor.name for ancestor in named_nodes] + [node.name]

def _assignment_name(node: ast.Assign | ast.AnnAssign) -> str | None:
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    names = [target.id for target in targets if isinstance(target, ast.Name)]
    return names[0] if len(names) == 1 else None

def symbol_for_line(
    tree: ast.AST, parents: dict[ast.AST, ast.AST], line: int
) -> tuple[dict[str, Any], int]:
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and _node_contains_line(node, line)
    ]
    if functions:
        node = min(functions, key=lambda item: item.end_lineno - item.lineno)
        kind = "method" if isinstance(parents.get(node), ast.ClassDef) else "function"
        return (
            {"kind": kind, "qualified_name": _qualified_name(node, parents)},
            node.end_lineno - node.lineno,
        )

    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and _node_contains_line(node, line)
        and _assignment_name(node) is not None
    ]
    if assignments:
        node = min(assignments, key=lambda item: item.end_lineno - item.lineno)
        class_names = [
            ancestor.name
            for ancestor in reversed(_ancestor_nodes(node, parents))
            if isinstance(ancestor, ast.ClassDef)
        ]
        return (
            {
                "kind": "field",
                "qualified_name": [*class_names, _assignment_name(node)],
            },
            node.end_lineno - node.lineno,
        )

    classes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and _node_contains_line(node, line)
    ]
    if classes:
        node = min(classes, key=lambda item: item.end_lineno - item.lineno)
        return (
            {"kind": "class", "qualified_name": _qualified_name(node, parents)},
            node.end_lineno - node.lineno,
        )

    return {"kind": "module", "qualified_name": []}, len(tree.body)

def common_symbol_for_lines(
    tree: ast.AST,
    parents: dict[ast.AST, ast.AST],
    lines: set[int],
) -> dict[str, Any]:
    nodes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and all(_node_contains_line(node, line) for line in lines)
    ]
    if not nodes:
        return {"kind": "module", "qualified_name": []}

    node = min(nodes, key=lambda item: item.end_lineno - item.lineno)
    if isinstance(node, ast.ClassDef):
        kind = "class"
    else:
        kind = "method" if isinstance(parents.get(node), ast.ClassDef) else "function"
    return {"kind": kind, "qualified_name": _qualified_name(node, parents)}
