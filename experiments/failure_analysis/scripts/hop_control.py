"""Pair existing deeper-hop runs with one-hop evidence-omission failures."""
from collections import defaultdict

from attribute import VERSION, audit_run, classify, get_requests, write
from summarize import HERE, ROOT, read, screen


def summarize_group(rows):
    valid = [r for r in rows if not r.get('error')]
    return dict(
        total=len(rows), verified=len(valid), pending=len(rows)-len(valid),
        evidence_completed=sum(r['evidence_complete'] for r in valid),
        repaired=sum(r['outcome'] == 'full' for r in valid),
        completed_and_repaired=sum(r['evidence_complete'] and r['outcome'] == 'full' for r in valid),
        completed_but_failed=sum(r['evidence_complete'] and r['outcome'] != 'full' for r in valid),
    )


def main():
    baseline = [r for r in read(ROOT / 'outputs/analysis/failure_analysis/data/attributed_keys.json')
                if r['benchmark'] == 'specgap' and r['judge'] == 'glm-5-2'
                and r['category'] in ('file_selection', 'evidence_presentation')]
    grouped = defaultdict(list)
    for row in baseline:
        grouped[(row['model'], row['sample_id'])].append(row)
    output = []
    for index, ((model, sample), cohort) in enumerate(sorted(grouped.items()), 1):
        original, error = get_requests(cohort[0])
        if error:
            raise ValueError(error)
        original_state = read(original[0])
        for hops in (2, 3):
            stage = ROOT/'outputs/analysis/sensitivity/results/conditions'/model/'specgap'/f'graph_h{hops}_r8'/'stage1'
            run = stage/'runs'/sample
            prediction = read(run/'prediction.json')
            result_path = stage/'judges/glm-5-2'/sample/'result.json'
            assert read(result_path.with_name('judge_input.json'))['prediction'] == prediction
            gold = read(ROOT/cohort[0]['gold_path'])
            outcomes = {r['key_id']: r for r in screen('specgap', gold, read(result_path))}
            row = dict(cohort[0], run_path=run.relative_to(ROOT).as_posix(), prediction=prediction)
            cache = ROOT / 'outputs/analysis/failure_analysis/data/hop_control_audits'/model/f'h{hops}'/f'{sample}.json'
            audit = read(cache) if cache.exists() else None
            if audit is None or audit.get('version') != VERSION:
                audit = audit_run(row)
                audit['version'] = VERSION
                write(cache, audit)
            config_checks = {}
            if not audit.get('error'):
                state = read(ROOT/audit['state'])
                config_checks = {k: original_state[k] == state[k]
                                 for k in ('config', 'max_rounds', 'model_profile', 'system')}
            for old in cohort:
                category, reason, evidence = classify(dict(old, prediction=prediction), audit)
                locations = evidence.get('locations', [])
                record = dict(model=model, sample_id=sample, key_id=old['key_id'], hops=hops,
                              baseline_category=old['category'], baseline_outcome=old['outcome'],
                              outcome=outcomes[old['key_id']]['outcome'], category=category,
                              evidence_complete=bool(locations) and all(x['complete'] for x in locations),
                              error=audit.get('error'), reason=reason, evidence=evidence,
                              config_checks=config_checks, result_path=result_path.relative_to(ROOT).as_posix(),
                              request_audit_path=cache.relative_to(ROOT).as_posix())
                output.append(record)
        if index % 20 == 0:
            print(f'Checked {index}/{len(grouped)} model/sample pairs', flush=True)
    groups = defaultdict(list)
    for row in output:
        groups[(row['baseline_category'], row['model'], row['hops'])].append(row)
    summary = [dict(baseline_category=k[0], model=k[1], hops=k[2], **summarize_group(v))
               for k, v in sorted(groups.items())]
    write(ROOT / 'outputs/analysis/failure_analysis/data/hop_control_keys.json', output)
    write(ROOT / 'outputs/analysis/failure_analysis/data/hop_control_summary.json', summary)
    if not all(not r['error'] and all(r['config_checks'].values()) for r in output):
        raise ValueError('Resolve request/config mismatches before writing the report')
    for item in summary:
        print(item)
    mismatches = {k: sum(not r['config_checks'].get(k, False) for r in output)
                  for k in ('config', 'max_rounds', 'model_profile', 'system')}
    print('Config mismatch counts:', mismatches)


if __name__ == '__main__':
    main()
