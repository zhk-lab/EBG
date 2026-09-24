"""Join explicitly typed experiment receipts and claims from frozen repository text.

These are comparisons of supplied records, not verification of their execution or
causal explanations. Unrecognized data is left available through repository reads.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import PurePosixPath
from typing import Any

from .storage import Store


def _records(store: Store, view: dict):
    for path, file_id in sorted(view['files'].items()):
        suffix = PurePosixPath(path).suffix.lower()
        if suffix not in {'.json', '.jsonl'}:
            continue
        content = store.file(file_id)['content']
        chunks = enumerate(content.splitlines(), 1) if suffix == '.jsonl' else [(1, content)]
        for line, chunk in chunks:
            try:
                value = json.loads(chunk)
            except (json.JSONDecodeError, ValueError):
                continue
            if not isinstance(value, dict) or value.get('record_type') not in ('experiment', 'claim'):
                continue
            last = line + max(1, len(chunk.splitlines())) - 1
            yield value, {'source': f'{path}@{line}-{last}',
                          'read_ref': f"{view['view_id']}:repo:{path}"}


def _valid_ids(*values) -> bool:
    return all(isinstance(ids, list) and all(isinstance(i, str) for i in ids) for ids in values)


def _overlap(record: dict) -> dict:
    left, right = record.get('training_row_ids'), record.get('evaluation_row_ids')
    if not _valid_ids(left, right):
        return {'status': 'unavailable'}
    common = sorted(set(left) & set(right))
    return {'status': 'compared', 'count': len(common), 'examples': common[:5],
            'training_unique': len(set(left)), 'evaluation_unique': len(set(right))}


def _population(left, right) -> dict:
    if not _valid_ids(left, right):
        return {'status': 'unavailable'}
    before, after = Counter(left), Counter(right)
    removed, added = before - after, after - before
    return {'status': 'compared', 'equal': before == after,
            'baseline_count': len(left), 'candidate_count': len(right),
            'removed_count': sum(removed.values()), 'added_count': sum(added.values()),
            'removed_examples': sorted(removed)[:5], 'added_examples': sorted(added)[:5],
            'note': 'Compare recorded sample ID multisets, retaining multiplicity and ignoring order. Matching filenames do not establish matching evaluation populations; matching IDs do not establish identical content or labels.'}


def _summary(record: dict, source: dict) -> dict:
    fields = ('experiment_id', 'revision', 'config', 'metric', 'parameters',
              'training_sources', 'evaluation_source', 'producer')
    return {**source, **{k: record[k] for k in fields if k in record},
            'row_id_overlap': _overlap(record)}


def _compare(field: str, left: dict, right: dict, key: str, *, right_key: str | None = None) -> dict:
    right_key = right_key or key
    result: dict[str, Any] = {'field': field}
    if key in left:
        result['left'] = left[key]
    if right_key in right:
        result['right'] = right[right_key]
    if key not in left:
        result['status'] = 'missing_in_baseline'
    elif right_key not in right:
        result['status'] = 'missing_in_record'
    else:
        result.update(status='compared', equal=left[key] == right[right_key])
    return result


def _mapping(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def research_context(store: Store, view: dict) -> dict | None:
    experiments, claims = [], []
    index: dict[str, list[dict]] = defaultdict(list)
    populations = {}
    for record, source in _records(store, view):
        if record['record_type'] == 'claim':
            claims.append((record, source))
        else:
            entry = _summary(record, source)
            populations[entry['source']] = record.get('evaluation_row_ids')
            experiments.append(entry)
            identifier = record.get('experiment_id')
            if isinstance(identifier, str):
                index[identifier].append(entry)
    if not experiments and not claims:
        return None

    def resolve(identifier):
        matches = index.get(identifier, []) if isinstance(identifier, str) else []
        if len(matches) == 1:
            return {'status': 'unique', 'record': matches[0]}
        return {'status': 'ambiguous' if matches else 'unresolved', 'records': matches}

    linked = []
    for claim, source in claims:
        candidate = resolve(claim.get('experiment_id'))
        item = {**source, 'claim': claim, 'candidate': candidate}
        if candidate['status'] == 'unique':
            receipt = candidate['record']
            comparisons = [_compare('revision', claim, receipt, 'revision')] if 'revision' in claim else []
            metric = _mapping(claim.get('metric'))
            comparisons.extend(_compare('metric.' + k, metric, _mapping(receipt.get('metric')), k) for k in metric)
            conditions = _mapping(claim.get('conditions'))
            parameters = _mapping(receipt.get('parameters'))
            comparisons.extend(_compare('conditions.' + k, conditions, parameters, k) for k in conditions)
            item['claim_comparisons'] = comparisons
        if 'baseline_experiment_id' in claim:
            baseline = resolve(claim['baseline_experiment_id'])
            item['baseline'] = baseline
            if baseline['status'] == candidate['status'] == 'unique':
                before, after = baseline['record'], candidate['record']
                item['evaluation_population'] = _population(populations[before['source']], populations[after['source']])
                comparisons = [_compare(k, before, after, k) for k in
                               ('training_sources', 'evaluation_source') if k in before or k in after]
                left, right = _mapping(before.get('parameters')), _mapping(after.get('parameters'))
                comparisons.extend(_compare('parameters.' + k, left, right, k) for k in sorted(left.keys() | right.keys()))
                # Metric values are allowed to improve; compare the metric identity separately.
                left, right = _mapping(before.get('metric')), _mapping(after.get('metric'))
                comparisons.extend(_compare('metric.' + k, left, right, k) for k in ('name', 'direction')
                                   if k in left or k in right)
                item['baseline_comparisons'] = comparisons
        linked.append(item)
    return {
        'note': 'Linked from collected JSON/JSONL entries explicitly marked record_type=claim/experiment. Values are source records, not proof of execution; differences do not automatically establish a violation or explain improvement. row_id_overlap compares string IDs only: check their namespaces, contents, and uses. Ordinary training labels do not themselves imply leakage. Duplicate experiment IDs remain ambiguous.',
        'comparison_order': 'claim_comparisons: left=claim, right=corresponding experiment; baseline_comparisons: left=baseline, right=candidate. Missing fields remain unknown.',
        'claims': linked, 'experiments': experiments,
    }
