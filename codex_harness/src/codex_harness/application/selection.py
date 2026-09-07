"""Source-backed value anchors and compact, structurally complete excerpts."""

from __future__ import annotations

import ast
import re
import textwrap
from typing import Any

from .repository import module_statement_span


def value_contexts(contexts: list[dict[str, Any]], text: str,
                   paths: set[str]) -> list[dict[str, Any]]:
    """Locate explicit values, without interpreting prose as a symbol name."""
    terms = set(re.findall(r'`([A-Za-z_][A-Za-z_0-9]*)`', text))
    terms.update(re.findall(r'\b[A-Z][A-Z_0-9]{2,}\b', text))
    terms = {term.casefold() for term in terms}
    # Definitions continue to use the unique-symbol resolver, including ambiguity.
    terms -= {c['symbol'].rsplit('.', 1)[-1].casefold() for c in contexts}
    if not terms:
        return []
    result = []
    for context in contexts:
        if not context['path'].endswith('.py') or (paths and context['path'] not in paths):
            continue
        code = context['source']
        if not terms.intersection(re.findall(r'[a-z_][a-z_0-9]*', code.casefold())):
            continue
        try:
            tree = ast.parse(textwrap.dedent(code))
        except (SyntaxError, ValueError):
            continue
        body = tree.body
        if context['symbol'] != '<module>':
            if len(body) != 1 or not isinstance(body[0], (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            body = body[0].body
        matches = []

        def visit(node: ast.AST) -> None:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                return  # Nested scopes have their own contexts.
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                return  # Documentation is not a use of a value.
            value = (node.id if isinstance(node, ast.Name) else
                     node.attr if isinstance(node, ast.Attribute) else
                     node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else '')
            hits = terms.intersection(re.findall(r'[a-z_][a-z_0-9]*', value.casefold()))
            if hits:
                matches.append((node.lineno, hits))
            for child in ast.iter_child_nodes(node):
                visit(child)

        for statement in body:
            visit(statement)
        if not matches:
            continue
        basis = 'explicit value: ' + ', '.join(sorted(set().union(*(hits for _, hits in matches))))
        if context['symbol'] == '<module>':
            lines = code.splitlines(keepends=True)
            for line, _ in matches:
                first, last = module_statement_span(context['path'], code, line, line)
                result.append({**context, 'lines': [first, last],
                               'source': ''.join(lines[first - 1:last]), 'match': basis})
        else:
            result.append({**context, 'match': basis})
    return result


def enclosing_contexts(selected: list[dict[str, Any]],
                       contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Include the nearest enclosing scope; this does not assert a runtime call."""
    result = []
    for child in selected:
        parents = [c for c in contexts if c['path'] == child['path']
                   and child['symbol'].startswith(c['symbol'] + '.')
                   and c['lines'][0] <= child['lines'][0] and c['lines'][1] >= child['lines'][1]]
        if parents:
            parent = min(parents, key=lambda c: c['lines'][1] - c['lines'][0])
            result.append({**parent, 'context': f"enclosing scope of {child['path']}::{child['symbol']}"})
    return result


def compact_contexts(contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep containing excerpts once, preserving all matching and edge reasons."""
    result = []
    for context in sorted(contexts, key=lambda c: (c['path'], c['lines'][0], -c['lines'][1])):
        parent = next((c for c in result if c['path'] == context['path']
                       and c['lines'][0] <= context['lines'][0] and c['lines'][1] >= context['lines'][1]), None)
        if parent is None:
            result.append(dict(context))
            continue
        for key in ('match', 'context'):
            values = dict.fromkeys((parent.get(key, '') + '\n' + context.get(key, '')).splitlines())
            if any(values):
                parent[key] = '\n'.join(filter(None, values))
    return result


def share_repo_excerpts(groups: dict[str, Any]) -> None:
    """Replace exact repeated excerpts with references to the frozen material."""
    seen = set()
    for group in groups.values():
        for entry in group.get('actual', {}).get('repo', []):
            key = entry['source'], entry['content']
            if key in seen:
                entry['content_ref'] = entry['read_ref']
                del entry['content']
            else:
                seen.add(key)
