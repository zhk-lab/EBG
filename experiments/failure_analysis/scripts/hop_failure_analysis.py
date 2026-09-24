"""Classify every SpecGap failure at each existing expansion depth."""
from collections import Counter

from attribute import CATEGORIES, VERSION, audit_run, classify, write
from summarize import HERE, ROOT, MODELS, read, screen


def main():
    baseline = [r for r in read(ROOT / 'outputs/analysis/failure_analysis/data/attributed_keys.json') if r['benchmark'] == 'specgap' and r['judge'] == 'glm-5-2']
    records = [dict(r, hops=1) for r in baseline]
    for model in MODELS:
        for hops in (2, 3):
            stage = ROOT / 'outputs/analysis/sensitivity/results/conditions' / model / 'specgap' / f'graph_h{hops}_r8' / 'stage1'
            paths = sorted((stage / 'judges/glm-5-2').glob('*/result.json'))
            assert len(paths) == 100
            for path in paths:
                sample = path.parent.name
                gold = read(ROOT / 'data/prepared/specgap/artifacts/hidden_gold' / f'{sample}.json')
                failures = [r for r in screen('specgap', gold, read(path)) if r['outcome'] != 'full']
                if not failures:
                    continue
                run = stage / 'runs' / sample
                prediction = read(run / 'prediction.json')
                assert read(path.with_name('judge_input.json'))['prediction'] == prediction
                base = dict(benchmark='specgap', model=model, sample_id=sample, run_path=run.relative_to(ROOT).as_posix(), prediction=prediction)
                cache = ROOT / 'outputs/analysis/failure_analysis/data/hop_control_audits' / model / f'h{hops}' / f'{sample}.json'
                audit = read(cache) if cache.exists() else None
                if audit is None or audit.get('version') != VERSION:
                    audit = audit_run(base)
                    audit['version'] = VERSION
                    write(cache, audit)
                for item in failures:
                    row = dict(base, **item)
                    category, reason, evidence = classify(row, audit)
                    records.append(dict(model=model, sample_id=sample, key_id=item['key_id'], hops=hops, outcome=item['outcome'], category=category, reason=reason, evidence=evidence, result_path=path.relative_to(ROOT).as_posix(), request_audit_path=cache.relative_to(ROOT).as_posix()))
            print(f'Completed {model} h{hops}', flush=True)
    summary = []
    for model in MODELS:
        for hops in (1, 2, 3):
            rows = [r for r in records if r['model'] == model and r['hops'] == hops]
            counts = Counter((r['category'] for r in rows))
            summary.append(dict(model=model, hops=hops, failures=len(rows), pending=counts[None], **{c: counts[c] for c in CATEGORIES}))
    write(ROOT / 'outputs/analysis/failure_analysis/data/hop_failure_keys.json', records)
    write(ROOT / 'outputs/analysis/failure_analysis/data/hop_failure_summary.json', summary)
    title = '## Hop expansion and evidence omissions'
    text = '\n\nSpecGAP, GLM-5.2 judge: 100 inputs and 482 conditions with localization references per model and setting. Each run is attributed separately; percentages use attributed failures as the denominator.\n\n'
    text += '| Model | Hops | Failures | File selection | Evidence presentation | Model judgment | Unresolved |\n|---|---:|---:|---:|---:|---:|---:|\n'
    labels = {'kimi-k3': 'K3', 'glm-5-3': 'GLM-5-3', 'luna': 'Luna'}
    for r in summary:
        n = r['failures'] - r['pending']
        cells = [f'{r[c]}（{r[c] / n:.1%}）' if n else str(r[c]) for c in CATEGORIES]
        text += f'| {labels[r['model']]} | {r['hops']} | {r['failures']} | ' + ' | '.join(cells) + f' | {r['pending']} |\n'
    text += '\nModel judgment remains the largest failure category. Increasing hops does not consistently reduce omissions or total failures.\n'
    text += '\nSettings are independent runs with potentially different failed inputs. Counts describe distributions, not recovery rates for a fixed set of failures.\n'
    text += '\nRun `python experiments/failure_analysis/scripts/hop_failure_analysis.py`; details and summaries are in `data/hop_failure_keys.json` and `data/hop_failure_summary.json`.\n'
    report = ROOT / "outputs/analysis/failure_analysis/REPORT.md"
    content = (report.read_text(encoding='utf-8') if report.exists() else '# Analysis report\n').split(title)[0].rstrip()
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(content + '\n\n' + title + text, encoding='utf-8')
    for row in summary:
        print(row)


if __name__ == '__main__':
    main()
