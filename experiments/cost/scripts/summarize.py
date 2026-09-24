"""Summarize accepted prediction input tokens, including cached input (no API calls)."""
import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from agentloop.silentswap_source_review import load_stage1

MODELS = {'kimi-k3': 'Kimi-K3', 'luna': 'Luna', 'glm-5-3': 'GLM-5-3'}
BENCHMARKS = ('specgap', 'silentswap', 'feedbacktrace')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def input_tokens(usage):
    # Both provider names include cache hits; do not add cached_tokens again.
    return usage['input_tokens'] if 'input_tokens' in usage else usage['prompt_tokens']


def accepted_tokens(run, source_review=False):
    batch = read(run / 'batch_result.json')
    if batch['status'] != 'complete':
        raise ValueError(f'Incomplete prediction: {run}')
    if batch.get('usage_scope') == 'successful_attempt_with_citation_repair':
        # This historical repair stores accepted-call usage in the batch record;
        # the original state intentionally retains its failed finish.
        return input_tokens(batch['usage'])
    if source_review:
        state = read(run / 'state.json')
        if state['status'] != 'complete':
            raise ValueError(f'Incomplete source review: {run}')
        # Corrections replace the one-shot answer; rejected responses are excluded.
        responses = sorted((run / 'responses').glob('turn_*.json'))
        response = read(responses[-1])
        return input_tokens(response['usage'])
    _, state, _ = load_stage1(run)
    records = [*state.get('records', []), state['terminal_record']]
    return sum(input_tokens(record['usage']) for record in records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true',
                        help='Reuse completed groups only if source batches have not changed')
    args = parser.parse_args()
    rows, samples = [], []
    out = ROOT / 'outputs/analysis/cost/results'
    cache = out / 'groups_input'
    cache.mkdir(parents=True, exist_ok=True)
    for model, label in MODELS.items():
        for benchmark in BENCHMARKS:
            directory = 'gpt5.6' if model == 'luna' and benchmark != 'feedbacktrace' else model
            methods = ('EBG', 'baseline') if benchmark == 'feedbacktrace' else ('EBG', 'baseline', 'RepoGraph')
            ids_by_method = []
            for method in methods:
                base = ROOT / 'outputs/main' / ('repograph' if method == 'RepoGraph' else method) / benchmark / directory
                if method == 'RepoGraph':
                    base /= 'RepoGraph'
                two_stage = benchmark == 'silentswap' and method == 'EBG'
                runs = (base / 'stage1' if two_stage else base) / 'runs'
                manifest = read(runs.parent / 'manifest.json')
                ids = manifest['selected_ids'][benchmark]
                cases = [runs / sample for sample in sorted(ids)]
                if len(cases) != 100 or len(set(ids)) != 100:
                    raise ValueError(f'Expected 100 predictions: {runs}')
                ids_by_method.append({p.name for p in cases})
                checkpoint = cache / f'{model}_{benchmark}_{method}.json'
                if args.resume and checkpoint.exists():
                    saved = read(checkpoint)
                    if {r['input_id'] for r in saved['samples']} != set(ids):
                        raise ValueError(f'Checkpoint sample mismatch: {checkpoint}')
                    rows.append(saved['group'])
                    samples.extend(saved['samples'])
                    continue
                total = 0
                group_samples = []
                for run in cases:
                    tokens = accepted_tokens(run)
                    if two_stage:
                        tokens += accepted_tokens(base / 'stage2/runs' / run.name, source_review=True)
                    total += tokens
                    group_samples.append(dict(model=label, benchmark=benchmark, method=method,
                                        input_id=run.name, input_tokens=tokens))
                rows.append(dict(model=label, benchmark=benchmark, method=method, samples=100,
                                 total_input_tokens=total, mean_input_tokens=total / 100,
                                 source=base.relative_to(ROOT).as_posix()))
                samples.extend(group_samples)
                checkpoint.write_text(json.dumps({'group': rows[-1], 'samples': group_samples}, indent=2) + '\n', encoding='utf-8')
                print(label, benchmark, method, f'{total / 100:,.1f}', flush=True)
            if any(ids != ids_by_method[0] for ids in ids_by_method):
                raise ValueError(f'Unpaired samples: {model}/{benchmark}')
    (out / 'token_usage.json').write_text(json.dumps({'metric': 'input_tokens', 'includes_cached_input': True,
                                                   'groups': rows, 'samples': samples}, indent=2) + '\n', encoding='utf-8')
    with (out / 'token_usage.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
