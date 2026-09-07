"""Follow explicit file references into frozen configuration and data materials."""

from __future__ import annotations

from collections import deque
from pathlib import PurePosixPath
import posixpath
import re

from .table_evidence import table_evidence


FILE_REFERENCE = re.compile(r'[\w./\\-]+\.(?:json|csv|tsv|toml|yaml|yml|md|markdown|py)\b')


def linked_artifacts(store, view, seeds, *, limit=24, max_depth=3):
    """Return navigation evidence, not an assertion that a referenced file was used."""
    paths = set(view['files'])
    by_name = {}
    for path in sorted(paths):
        by_name.setdefault(PurePosixPath(path).name, []).append(path)
    queue = deque((path, text, 0) for path, text in seeds)
    visited = set()
    items = []
    ambiguous = []
    limited = False
    while queue:
        origin, text, depth = queue.popleft()
        for match in FILE_REFERENCE.finditer(text):
            reference = match.group().replace('\\', '/')
            relative = posixpath.normpath(posixpath.join(posixpath.dirname(origin), reference))
            if relative in paths:
                target, basis = relative, 'relative path'
            elif reference in paths:
                target, basis = reference, 'repository path'
            elif '/' not in reference and len(by_name.get(reference, [])) == 1:
                target, basis = by_name[reference][0], 'unique basename; location requires confirmation'
            else:
                candidates = by_name.get(reference, []) if '/' not in reference else []
                if len(candidates) > 1:
                    entry = {'from': origin, 'reference': reference, 'candidates': candidates}
                    if entry not in ambiguous:
                        ambiguous.append(entry)
                continue
            if target in visited:
                continue
            if len(visited) >= limit:
                limited = True
                continue
            visited.add(target)
            content = store.file(view['files'][target])['content']
            if depth < max_depth:
                queue.append((target, content, depth + 1))
            elif FILE_REFERENCE.search(content):
                limited = True
            if target.endswith('.py'):
                continue  # Code excerpts are supplied by BEG; follow their file references only.
            lines = content.splitlines(keepends=True)
            excerpt = content if len(content) <= 4000 else ''.join(lines[:8])[:1200]
            items.append({'path': target, 'from': origin, 'reference': reference, 'match': basis,
                          'source': f'{target}@1-{max(1, len(excerpt.splitlines()))}',
                          'content': excerpt, 'excerpt_only': len(excerpt) < len(content),
                          'read_ref': f"{view['view_id']}:repo:{target}"})
    if not items and not ambiguous:
        return None
    return {'note': '沿要求、引用和代码中的文件引用补取冻结材料；引用关系不证明实际使用，也不自动证明来源合规。'
                    '检查结论所依赖的配置与输入条件；excerpt_only 的材料可按 read_ref 继续读取。',
            'items': items, 'ambiguous_references': ambiguous, 'limited': limited,
            'table_relations': table_evidence(store, view, [item['path'] for item in items]),
            'repository_ref': f"{view['view_id']}:files:index"}
