"""Inventory per-KEY failure candidates from frozen judges; no API calls.

This is screening, not human causal attribution. Unknowns remain explicit.
"""
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
MODELS = ('kimi-k3', 'luna', 'glm-5-3')
JUDGES = ('qwen3.7-max-2026-06-08', 'glm-5-2')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def screen(component, gold, result):
    """Return one screening record per gold, keeping partial credit separate."""
    if component == 'specgap':
        matches = {m['condition_id']: m for m in result['llm_judge']['judgment']['matches']}
        return [dict(key_id=g['condition_id'], gold=g,
                     outcome=('full' if matches.get(g['condition_id'], {}).get('match_score', 0) == 1
                              else 'partial' if matches.get(g['condition_id'], {}).get('match_score', 0) > 0
                              else 'unmatched'),
                     judgment=matches.get(g['condition_id'])) for g in gold['conditions']
                if g.get('implementation_locations')]
    if component == 'silentswap':
        checks = {c['reference_swap_number']: c for c in result['llm_judge']['location_checks']}
        return [dict(key_id=f'swap_{i}', gold=g,
                     outcome=('unknown' if i not in checks else
                              'full' if checks[i]['fully_correct'] else 'incorrect'),
                     judgment=checks.get(i)) for i, g in enumerate(gold['swaps'], 1)]
    value = result['verification_point_alignment']
    return [dict(key_id='verification_point', gold=gold,
                 outcome='full' if value == 1 else 'partial' if value > 0 else 'incorrect',
                 judgment=result)]


def collect():
    records, summaries = [], []
    for component in ('specgap', 'silentswap', 'feedbacktrace'):
        for model in MODELS:
            directory = 'gpt5.6' if model == 'luna' and component != 'feedbacktrace' else model
            base = ROOT / 'outputs/main' / 'EBG' / component / directory
            for judge in JUDGES:
                disk_judge = 'glm-5.2' if component == 'feedbacktrace' and judge == 'glm-5-2' else judge
                stage = base / 'stage1' if component == 'silentswap' else base
                paths = sorted((stage / 'judges' / disk_judge).glob('*/result.json'))
                assert len(paths) == 100, (component, model, judge, len(paths))
                group = []
                excluded = 0
                for path in paths:
                    sample = path.parent.name
                    gold_path = ROOT / 'data' / 'prepared' / component / 'artifacts/hidden_gold' / f'{sample}.json'
                    result, gold = read(path), read(gold_path)
                    if component == 'specgap':
                        excluded += sum(not g.get('implementation_locations') for g in gold['conditions'])
                    run = stage / 'runs' / sample
                    prediction_path = run / 'prediction.json'
                    prediction = read(prediction_path) if prediction_path.exists() else None
                    pairing = None
                    for item in screen(component, gold, result):
                        item.update(benchmark=component, model=model, judge=judge, sample_id=sample,
                                    gold_path=str(gold_path.relative_to(ROOT)),
                                    result_path=str(path.relative_to(ROOT)),
                                    prediction=prediction,
                                    prediction_path=str(prediction_path.relative_to(ROOT)),
                                    run_path=str(run.relative_to(ROOT)),
                                    stage_pairing_note=pairing,
                                    category=None,
                                    review_status='not_required' if item['outcome'] == 'full' else 'pending',
                                    review_reason=None if item['outcome'] == 'full' else
                                    'Requires per-KEY validation and evidence-sufficiency review; score alone does not identify the failing stage.')
                        group.append(item)
                counts = Counter(r['outcome'] for r in group)
                summaries.append(dict(benchmark=component, model=model, judge=judge,
                                      samples=len(paths), gold_keys=len(group), excluded_no_location=excluded,
                                      full=counts['full'],
                                      partial=counts['partial'], unmatched_or_incorrect=counts['unmatched']+counts['incorrect'],
                                      unknown=counts['unknown'], failure_candidates=len(group)-counts['full']-counts['unknown'],
                                      attributed=0, pending=len(group)-counts['full'],
                                      file_selection='not_applicable' if component == 'feedbacktrace' else None))
                records.extend(group)
    return records, summaries


def main():
    records, summary = collect()
    (ROOT / 'outputs/analysis/failure_analysis/data').mkdir(exist_ok=True)
    for name, data in [('key_screening.json', records), ('screening_summary.json', summary)]:
        (ROOT / 'outputs/analysis/failure_analysis/data' / name).write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print('Saved per-KEY screening and summaries.')


if __name__ == '__main__':
    main()
