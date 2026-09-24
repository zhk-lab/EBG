"""Render the request-evidence audit without regenerating predictions or figures."""
from collections import Counter

from summarize import HERE, ROOT, read
from attribute import CATEGORIES, write


def main():
    rows = read(ROOT / 'outputs/analysis/failure_analysis/data/attributed_keys.json')
    summary = read(ROOT / 'outputs/analysis/failure_analysis/data/attribution_summary.json')
    selected = [('specgap', 'sg_009', 'kc_004'), ('specgap', 'sg_023', 'kc_001'), ('specgap', 'sg_003', 'kc_008'), ('silentswap', 'ss_006', 'swap_2'), ('feedbacktrace', 'ft_032_long', 'verification_point')]
    explanations = ['parse_resource_id returns a name dictionary when ARM parsing fails. The run read authentication, cloud configuration, and polling files but not msrestazure/tools.py, which contains the Gold fallback implementation. The output also omitted that rule.', 'AlignedStream obtains its default buffer size from DISSECT_STREAM_BUFFER_SIZE, falling back to io.DEFAULT_BUFFER_SIZE. The graph for stream.py was read, but line 11 defining this constant and the environment variable name were absent from the actual input. The output focused on other rules such as seek.', 'pretend.py:126-132 was fully presented. Line 129 appends the call record before line 130 invokes func, supporting retention of the record on exceptions. The output discussed call comparison and hashing but omitted the record/exception relationship.', 'Stage 1 read graphql_query/templates/field.jinja2 (F0013), but graph units returned in turns 1 and 2 had empty source, behavior_total=0, and groundable=false. Gold line 8 was not shown. The first-stage output did not locate this template, so the case is attributed to evidence presentation.', 'Gold concerns nonzero gate exit when a baseline file is absent from current LCOV. Although e_101_assistant_response_26 was split across excerpts, both the nonzero-exit statement and its condition were present in the current task input. The model selected the TS-only scope instead; both judges classified it as a different decision.']
    cases = []
    for (component, sample, key), explanation in zip(selected, explanations):
        row = next((r for r in rows if r['benchmark'] == component and r['model'] == 'kimi-k3' and (r['sample_id'] == sample) and (r['key_id'] == key) and r['judge'].startswith('qwen')))
        cases.append(dict(row, case_explanation=explanation))
    write(ROOT / 'outputs/analysis/failure_analysis/results/cases.json', cases)
    labels = dict(file_selection='File-selection omission', evidence_presentation='Evidence-presentation omission', model_judgment='Model-judgment failure')
    case_lines = ['# Failure case review', '', 'These cases are assistant reviews of saved records, not independent human double coding. cases.json retains Gold, predictions, scores, request locations, and attribution reasons.', '']
    for c in cases:
        audit = read(ROOT / c['request_audit_path'])
        case_lines += [f'## {c['benchmark']} / K3 / {c['sample_id']} / {c['key_id']}', '', f'Category: {labels[c['category']]}。', '', c['case_explanation'], '', f'- Gold: [record](../../../{c['gold_path'].replace(chr(92), '/')})', f'- Judgment: [record](../../../{c['result_path'].replace(chr(92), '/')})', f'- Prediction: [record](../../../{c['prediction_path'].replace(chr(92), '/')})', f'- Input review: [record](../../../{c['request_audit_path'].replace(chr(92), '/')})', '']
        for request in audit['requests']:
            case_lines.append(f'- Actual request: [JSON](../../../{request.replace(chr(92), '/')})')
        case_lines.append('')
    (ROOT / 'outputs/analysis/failure_analysis/results/CASES.md').write_text('\n'.join(case_lines), encoding='utf-8')
    report = ROOT / "outputs/analysis/failure_analysis/REPORT.md"
    original = (report.read_text(encoding='utf-8') if report.exists() else '# Analysis report\n')
    prefix = original.split('## Attribution results', 1)[0]
    lines = ['## Attribution results', '', 'Cells show counts and percentages of attributed failures. Unresolved cases are excluded from percentage denominators; judges are reported separately. SilentSwap candidates use first-stage location_checks.fully_correct, and evidence coverage uses only the corresponding first-stage request.', '']
    for judge, label in [('qwen3.7-max-2026-06-08', 'Qwen'), ('glm-5-2', 'GLM')]:
        lines += [f'### {label} judgment', '', '| Benchmark | Model | Included targets | Failure candidates | Attributed | Unresolved | File selection | Evidence presentation | Model judgment |', '|---|---|---:|---:|---:|---:|---:|---:|---:|']
        for s in summary:
            if s['judge'] != judge:
                continue
            cells = [s['benchmark'], s['model'], str(s['gold_keys']), str(s['failure_candidates']), str(s['attributed']), str(s['pending'])]
            for k in CATEGORIES:
                v = s['categories'][k]
                cells.append('N/A' if k == 'file_selection' and s['benchmark'] == 'feedbacktrace' else f'{v['count']}（{v['percent']:.1f}%）')
            lines.append('| ' + ' | '.join(cells) + ' |')
        lines.append('')
    lines += ['## Interpretation', '', '- SpecGAP failures are mainly in model judgment, with additional file-selection and graph-presentation omissions. Both relevant-file coverage and semantic review of presented evidence matter.', '- SilentSwap uses first-stage localization failures: unread files, missing reference code in graph output, and incorrect localization despite presented reference code form the three categories. Full source from stage 2 is outside this analysis.', '- In the analyzed FeedbackTrace failures, Gold events were fully present in the current task input. Failures mainly involve selecting a different decision or incomplete disclosure. This coverage definition found no omitted Gold events; cross-event interpretation and decision selection remain relevant.', '', 'Model-judgment failure includes attention, interpretation, and output formulation. It does not identify a specific cognitive cause or establish sufficiency of all other context. These categories are operational diagnostics, not human-confirmed causal proportions.', '', '## Exclusions and supplementary decisions', '', '- SpecGAP sg_024/kc_005 (coordinate error bound) and sg_084/kc_011 (Windows event-loop policy) lack Gold implementation_locations. Excluding them leaves 482 of 484 conditions. They are not counted as successes, failures, or unresolved cases.', '- The five conditions of Luna / SpecGAP sg_095 are assigned to model-judgment failure by supplementary author adjudication for both judges. This decision is recorded separately and does not claim verified complete evidence coverage. The current unresolved count is zero.', '', '## Cases', '', 'See [case review](results/CASES.md) for file-selection omissions, missing constants, omissions despite complete code, empty template graphs, and selection of a different trajectory decision.', '', '## Artifacts and reproduction', '', '- [Per-target attribution and reasons](data/attributed_keys.json)', '- [Category summary](data/attribution_summary.json)', '- [Per-target screening and partial matches](data/key_screening.json)', '- Review rules are in experiments/failure_analysis/configs/evidence_review_rules.json.', '- Supplementary adjudications are in experiments/failure_analysis/configs/user_adjudications.json.', '- data/request_audits/ stores selected files, presented code lines, Gold-event coverage, and request locations for each run.', '- Run summarize.py, attribute.py, then report_attribution.py from experiments/failure_analysis/scripts/. Per-run input reviews are cached for resume.', '']
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(prefix + '\n'.join(lines), encoding='utf-8')
    assert all((s['attributed'] + s['pending'] == s['failure_candidates'] for s in summary))
    assert all((sum((v['count'] for v in s['categories'].values())) == s['attributed'] for s in summary))
    print('Rendered and validated 18 groups and 5 reviewed cases.')


if __name__=='__main__':
    main()
