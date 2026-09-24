"""Export one browsing row per frozen evaluation input for Hugging Face."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREVIEW_LIMIT = 4000
TASKS = {
    'specgap': 'Identify omitted requirements using repository evidence.',
    'silentswap': 'Identify semantic substitutions that preserve existing tests.',
    'feedbacktrace': 'Identify consequential decisions before subsequent user feedback.',
}


def make_row(source: Path, benchmark: str, input_id: str) -> dict:
    relative = f'{benchmark}/artifacts/visible_bundles/{input_id}'
    bundle = source / relative
    manifest = json.loads((bundle / 'input_manifest.json').read_text(encoding='utf-8'))
    if manifest['input_id'] != input_id or manifest['benchmark'] != benchmark:
        raise ValueError(f'Mismatched manifest: {relative}')
    files = manifest['visible_files']
    if benchmark == 'feedbacktrace':
        preview_file = 'trace/model_input.json'
        if preview_file not in files:
            raise ValueError(f'Missing visible trace: {relative}')
        trace = json.loads((bundle / preview_file).read_text(encoding='utf-8'))
        parts = []
        length = 0
        for event in trace['events']:
            content = event.get('content', '')
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            part = f"[{event.get('event_type', 'event')}] {content}"
            parts.append(part)
            length += len(part) + 2
            if length > PREVIEW_LIMIT:
                break
        preview = '\n\n'.join(parts)
    else:
        preview_file = next(name for name in files if name.startswith('documents/'))
        preview = (bundle / preview_file).read_text(encoding='utf-8')
    return {
        'input_id': input_id,
        'benchmark': benchmark,
        'task': TASKS[benchmark],
        'input_preview': preview[:PREVIEW_LIMIT],
        'preview_truncated': len(preview) > PREVIEW_LIMIT,
        'visible_file_count': len(files),
        'preview_file': preview_file,
        'archive': f'{benchmark}.tar.gz',
        'bundle_path': relative,
        'gold_path': f'{benchmark}/artifacts/hidden_gold/{input_id}.json',
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    selection = json.loads((ROOT / 'experiments/main/configs/samples.json').read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    for benchmark in TASKS:
        rows = [make_row(args.source, benchmark, input_id) for input_id in selection[benchmark]]
        target = args.output / f'{benchmark}.jsonl'
        target.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows),
                          encoding='utf-8')
        print(f'{benchmark}: {len(rows)} rows')


if __name__ == '__main__':
    main()
