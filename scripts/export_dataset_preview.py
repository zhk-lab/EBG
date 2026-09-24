"""Export one browsing row per frozen evaluation input for Hugging Face."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ('specgap', 'silentswap', 'feedbacktrace')


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def repository_tree(files: list[str]) -> str:
    tree = {}
    for name in sorted(files):
        if not name.startswith('repository/'):
            continue
        node = tree
        for part in name.split('/')[1:]:
            node = node.setdefault(part, {})
    lines = ['repository/']

    def render(node: dict, prefix: str) -> None:
        items = sorted(node.items(), key=lambda item: (not bool(item[1]), item[0]))
        for index, (name, children) in enumerate(items):
            last = index == len(items) - 1
            lines.append(prefix + ('`-- ' if last else '|-- ') + name + ('/' if children else ''))
            render(children, prefix + ('    ' if last else '|   '))

    render(tree, '')
    return '\n'.join(lines)


def render_event(event: dict) -> str:
    roles = {'user_prompt': 'User', 'assistant_response': 'Assistant',
             'tool_exchange': 'Tool', 'tool_result': 'Tool'}
    kind = event['event_type']
    label = roles.get(kind, kind)
    if event.get('tool_name'):
        label += ' / ' + event['tool_name']
    header = f"Turn {event.get('turn_number', '?')} | {label}"
    if event.get('evidence_id'):
        header += ' | ' + event['evidence_id']
    content = event.get('content', '')
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False, indent=2)
    return header + '\n' + content


def render_location(location: dict) -> str:
    symbol = location.get('symbol', '')
    if isinstance(symbol, dict):
        symbol = symbol.get('qualified_name', [])
        if isinstance(symbol, list):
            symbol = '.'.join(symbol)
    ranges = ', '.join(str(r['start']) if r['start'] == r['end']
                       else f"{r['start']}-{r['end']}"
                       for r in location.get('line_ranges', []))
    path = location.get('file', '') + (':' + ranges if ranges else '')
    return path + (' -> ' + symbol if symbol else '')


def session_trace(input_id: str, events: list[dict]) -> list[dict]:
    """Map visible events to HF Session Trace Format without adding Gold."""
    records = [{'type': 'session', 'harness': 'feedbacktrace', 'id': input_id,
                'name': input_id + ' (pre-feedback trajectory)'}]
    roles = {'user_prompt': 'user', 'assistant_response': 'assistant'}
    for index, event in enumerate(events):
        kind = event['event_type']
        content = event['content']
        metadata = {key: event[key] for key in ('turn_number', 'evidence_id') if key in event}
        if kind == 'tool_exchange':
            invocation, separator, result = content.removeprefix('Tool invocation:\n').partition('\n\nTool result:')
            # Incomplete source events remain readable verbatim; never invent a result.
            try:
                call = json.loads(invocation)
            except json.JSONDecodeError:
                records.append({'type': 'message', **metadata,
                                'message': {'role': 'tool', 'content': content}})
                continue
            call_id = event.get('evidence_id', f'event_{index}')
            records.append({'type': 'message', **metadata, 'message': {
                'role': 'assistant', 'content': '', 'toolCalls': [{
                    'id': call_id, 'function': {'name': call.get('tool_name', event.get('tool_name', 'Tool')),
                    'arguments': json.dumps(call.get('input', {}), ensure_ascii=False)}}]}})
            if separator:
                records.append({'type': 'message', **metadata, 'message': {
                    'role': 'tool', 'content': result, 'toolCallId': call_id}})
        else:
            records.append({'type': 'message', **metadata,
                            'message': {'role': roles[kind], 'content': content}})
    return records


def make_row(source: Path, benchmark: str, input_id: str) -> dict:
    artifacts = source / benchmark / 'artifacts'
    bundle = artifacts / 'visible_bundles' / input_id
    manifest = read_json(bundle / 'input_manifest.json')
    gold = read_json(artifacts / 'hidden_gold' / f'{input_id}.json')
    for record in (manifest, gold):
        if record['input_id'] != input_id or record['benchmark'] != benchmark:
            raise ValueError(f'Mismatched input or Gold: {benchmark}/{input_id}')
    files = manifest['visible_files']
    if benchmark == 'feedbacktrace':
        if 'trace/model_input.json' not in files:
            raise ValueError(f'Missing visible trace: {input_id}')
        events = read_json(bundle / 'trace/model_input.json')['events']
        evidence = []
        for evidence_id in gold['evidence_ids']:
            matches = [event for event in events if event.get('evidence_id') == evidence_id]
            if len(matches) != 1:
                raise ValueError(f'Expected one visible event for {input_id}/{evidence_id}')
            evidence.append(render_event(matches[0]))
        return {'id': input_id,
                'trace': session_trace(input_id, events),
                'verification_point': gold['verification_point'],
                'evidence': '\n\n'.join(evidence),
                'criticality': gold['criticality']}
    documents = [name for name in files if name.startswith('documents/')]
    if len(documents) != 1:
        raise ValueError(f'Expected one visible document: {input_id}')
    row = {'id': input_id, 'document': (bundle / documents[0]).read_text(encoding='utf-8'),
           'repository_tree': repository_tree(files)}
    findings, evidence = [], []
    if benchmark == 'specgap':
        for condition in gold['conditions']:
            label = condition['condition_id']
            findings.append(f"[{label}] {condition['normalized_condition']}")
            lines = [f'[{label}]']
            for key, title in [('implementation_locations', 'Implementation'),
                               ('test_locations', 'Tests')]:
                locations = condition.get(key, [])
                if locations:
                    lines.append(title + ':')
                    lines.extend('- ' + render_location(item) for item in locations)
            evidence.append('\n'.join(lines))
        row['missing_requirements'] = '\n\n'.join(findings)
    elif benchmark == 'silentswap':
        for index, swap in enumerate(gold['swaps'], 1):
            label = f'change_{index}'
            findings.append(f"[{label}]\nExpected: {swap['original_semantics']}\n"
                            f"Changed: {swap['swapped_semantics']}\n"
                            f"Impact: {swap['why_different']}")
            lines = [f'[{label}]']
            if swap.get('localization'):
                lines.append(render_location(swap['localization']))
            lines.extend(f'{number}. {text}' for number, text in enumerate(swap.get('evidence', []), 1))
            evidence.append('\n'.join(lines))
        row['semantic_changes'] = '\n\n'.join(findings)
    else:
        raise ValueError(f'Unknown benchmark: {benchmark}')
    row['evidence'] = '\n\n'.join(evidence)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    selection = json.loads((ROOT / 'experiments/main/configs/samples.json').read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    for benchmark in BENCHMARKS:
        rows = [make_row(args.source, benchmark, input_id) for input_id in selection[benchmark]]
        target = args.output / f'{benchmark}.jsonl'
        target.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows),
                          encoding='utf-8')
        print(f'{benchmark}: {len(rows)} rows')


if __name__ == '__main__':
    main()
