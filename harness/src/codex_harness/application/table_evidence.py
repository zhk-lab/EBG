"""Join identifier columns in linked frozen tables, preserving original row evidence.

Equal values suggest a relation; they do not prove provenance or actual use. The
consumer must interpret these joins against the code and experiment requirements.
"""

from collections import Counter, deque
import csv
import io
from pathlib import PurePosixPath
import re


def _table(content, path, max_rows, notes):
    lines = content.splitlines()
    reader = csv.reader(io.StringIO(content), delimiter='\t' if path.endswith('.tsv') else ',')
    header = next(reader, [])
    if not header or len(header) > 32 or len(set(header)) != len(header):
        return None
    rows = []
    first = reader.line_num + 1
    for values in reader:
        if len(values) != len(header) or len(rows) >= max_rows:
            notes.append(f'{path}: malformed or over {max_rows} rows; omitted from joins')
            return None
        end = reader.line_num
        # csv.reader.line_num counts physical lines, including quoted newlines.
        rows.append({'values': dict(zip(header, values)), 'first': first, 'end': end,
                     'content': '\n'.join(lines[first - 1:end])})
        first = end + 1
    if not rows:
        return None
    keys, categories = {}, []
    for column in header:
        values = [r['values'][column] for r in rows]
        unique = set(values)
        if re.search(r'(^|_)(id|key)$', column, re.I):
            if '' not in unique and len(unique) == len(rows):
                keys[column] = {v: i for i, v in enumerate(values)}
            else:
                notes.append(f'{path}.{column}: nonunique or empty identifiers; join not inferred')
        elif 1 <= len(unique) <= 8 and all(v and not re.fullmatch(r'[+-]?[\d.eE]+', v) for v in unique):
            categories.append(column)
    return {'rows': rows, 'keys': keys, 'categories': categories}


def table_evidence(store, view, paths, *, max_rows=5000, max_hops=3, max_groups=12):
    """Summarize categorical values reached by exact, unique ID/key joins only."""
    notes, tables = [], {}
    for path in sorted(set(paths)):
        if PurePosixPath(path).suffix.lower() not in {'.csv', '.tsv'}:
            continue
        try:
            table = _table(store.file(view['files'][path])['content'], path, max_rows, notes)
        except csv.Error:
            notes.append(f'{path}: CSV parsing failed; omitted from joins')
            continue
        if table:
            tables[path] = table
    edges = {p: [] for p in tables}
    for i, left in enumerate(tables):
        for right in list(tables)[i + 1:]:
            for a, left_index in tables[left]['keys'].items():
                for b, right_index in tables[right]['keys'].items():
                    common = left_index.keys() & right_index.keys()
                    if not common:
                        continue
                    edges[left].append((right, a, b, {left_index[v]: right_index[v] for v in common}))
                    edges[right].append((left, b, a, {right_index[v]: left_index[v] for v in common}))
    groups, limited = [], False

    def quote(path, index):
        row = tables[path]['rows'][index]
        span = str(row['first']) if row['first'] == row['end'] else f"{row['first']}-{row['end']}"
        return {'source': f'{path}@{span}', 'content': row['content'],
                'read_ref': f"{view['view_id']}:repo:{path}"}

    for start, table in tables.items():
        queue = deque([(start, [(i,) for i in range(len(table['rows']))], [start], [])])
        visited = {start}
        while queue:
            path, matches, route, joins = queue.popleft()
            if joins:
                for column in tables[path]['categories']:
                    if len(groups) >= max_groups:
                        limited = True
                        break
                    counts = Counter(tables[path]['rows'][m[-1]]['values'][column] for m in matches)
                    examples = {}
                    for match in matches:
                        category = tables[path]['rows'][match[-1]]['values'][column]
                        examples.setdefault(category, match)
                    groups.append({'from': start, 'to': path, 'column': column, 'joins': joins,
                                   'matched_rows': len(matches), 'unmatched_rows': len(table['rows']) - len(matches),
                                   'counts': [{'value': v, 'rows': counts[v],
                                               'example': [quote(p, index) for p, index in zip(route, examples[v])]}
                                              for v in sorted(counts)]})
            if len(joins) >= max_hops:
                continue
            for target, left_key, right_key, mapping in sorted(edges[path], key=lambda e: -len(e[3])):
                if target in visited:
                    continue
                following = [(*m, mapping[m[-1]]) for m in matches if m[-1] in mapping]
                if not following:
                    continue
                visited.add(target)
                queue.append((target, following, route + [target],
                              joins + [f'{path}.{left_key} = {target}.{right_key}']))
    return {'note': 'Candidate table relations join equal values in unique ID/key columns, favoring short paths. They are not declared foreign keys, do not enumerate all paths, and do not prove actual use or provenance semantics. Group counts come from matched rows, with one complete join example per group. Interpret them alongside code, requirements, and execution.',
            'groups': groups, 'notes': notes, 'limited': limited}
