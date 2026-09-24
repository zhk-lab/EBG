"""Audit frozen request content against Gold evidence, with resumable per-run caches.

Categories are operational Gold-evidence coverage diagnoses, not counterfactual
proof that supplying an omitted passage would change the answer.
"""
import json
import re
from collections import defaultdict, Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

from summarize import HERE, ROOT, read

VERSION = 4
CATEGORIES = ('file_selection', 'evidence_presentation', 'model_judgment')
REVIEW_RULES = read(HERE / 'configs/evidence_review_rules.json')


def full_event_covered(original, excerpts):
    """Combine literal split excerpts; do not require one unsplit occurrence."""
    remaining = set(i for i,c in enumerate(original) if not c.isspace())
    for excerpt in excerpts:
        if original in excerpt:
            return True
        start = original.find(excerpt)
        if start >= 0:
            remaining.difference_update(range(start,start+len(excerpt)))
    return not remaining


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def numbered_sources(text, repository):
    """Accept line numbers only when both file context and literal source agree."""
    covered = defaultdict(set)
    current = None
    root_file = None
    source = {}
    for line in text.splitlines():
        if line.startswith('[CONTEXT ') or line == '[DIRECT ROOT]':
            current = root_file
        if line.startswith('FILE '):
            current = line[5:].strip()
            root_file = current
        elif '::' in line and not re.match(r'^\s*\d+\s*\|', line):
            matches = re.findall(r'([^\s:]+)::', line)
            if matches:
                current = matches[-1]
                if re.match(r'^(?:Root already shown: )?[^\s:]+::', line):
                    root_file = current
        match = re.match(r'^\s*(\d+)\s*\| ?(.*)$', line)
        if not match or not current:
            continue
        file = repository / current
        if current not in source:
            source[current] = file.read_text(encoding='utf-8').splitlines() if file.is_file() else []
        n, content = int(match[1]), match[2]
        if 1 <= n <= len(source[current]) and content.rstrip() == source[current][n-1].rstrip():
            covered[current].add(n)
    return covered


def get_requests(row):
    run = ROOT / row['run_path']
    matches = []
    for path in run.glob('**/state.json'):
        state = read(path)
        prediction_matches = state.get('prediction') == row['prediction']
        if row['benchmark'] == 'silentswap' and state.get('workflow') == 'ebg_independent_source_review_v5':
            for response in (path.parent/'responses').glob('turn_*.json'):
                try:
                    response_prediction = json.loads(read(response)['content'])
                    # Same property -> method normalization as evaluation_core.contracts.
                    for swap in response_prediction.get('swaps', []):
                        symbol=swap.get('target', {}).get('symbol', {})
                        if symbol.get('kind') == 'property':
                            symbol['kind']='method'
                    prediction_matches |= response_prediction.get('swaps') == row['prediction'].get('swaps')
                except (ValueError, AttributeError):
                    pass
        if state.get('status') == 'complete' and prediction_matches:
            matches.append(path)
    if len(matches) != 1:
        return None, f'Expected one completed state matching scored prediction, found {len(matches)}'
    state_path = matches[0]
    requests = sorted((state_path.parent / 'requests').glob('*.json'))
    if not requests:
        return None, 'No saved provider requests for the matching completed state'
    return (state_path, requests), None


def audit_run(row):
    found, error = get_requests(row)
    if error:
        return dict(error=error)
    state_path, requests = found
    component = row['benchmark']
    bundle = ROOT / 'data' / 'prepared' / component / 'artifacts/visible_bundles' / row['sample_id']
    covered = defaultdict(set)
    locators = defaultdict(list)
    selected = set()
    trace_items = defaultdict(list)
    messages_seen = set()
    current_scope = None
    for path in requests:
        for index, message in enumerate(read(path)['messages']):
            if message['role'] not in ('user', 'tool'):
                continue
            text = message['content']
            if not isinstance(text, str) or text in messages_seen:
                continue
            messages_seen.add(text)
            locator = f'{path.relative_to(ROOT).as_posix()}#messages[{index}]'
            if component == 'feedbacktrace':
                try:
                    if '[[COMPLETE TRACE VIEW]]' in text:
                        text = text.split('[[COMPLETE TRACE VIEW]]',1)[1].rsplit('[[END COMPLETE TRACE VIEW]]',1)[0].strip()
                    view = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if not isinstance(view, dict) or 'task_scopes' not in view:
                    continue
                current_scope = view['current_task_id']
                for scope in view['task_scopes']:
                    for behavior in scope['behaviors']:
                        for role in ('demand', 'action', 'response'):
                            for item in behavior.get(role, []):
                                if item.get('evidence_id'):
                                    trace_items[item['evidence_id']].append(dict(
                                        content=item['content'], scope=scope['task_id'],
                                        locator=locator, behavior=behavior['behavior_id']))
            else:
                spans = numbered_sources(text, bundle / 'repository')
                for file, lines in spans.items():
                    covered[file].update(lines)
                    locators[file].append(dict(request=locator, lines=sorted(lines)))
                selected.update(re.findall(r'^FILE (.+)$', text, flags=re.M))
    if component in ('specgap', 'silentswap'):
        state = read(state_path)
        for record in state.get('records', []):
            for unit in (record.get('tool_result') or {}).get('units', []):
                span = unit.get('span') or {}
                if span.get('path'):
                    selected.add(span['path'])
    selected.update(covered)
    result = dict(error=None, state=str(state_path.relative_to(ROOT)),
                  requests=[str(p.relative_to(ROOT)) for p in requests],
                  selected_files=sorted(selected),
                  covered_lines={k: sorted(v) for k,v in covered.items()},
                  locators=dict(locators), current_scope=current_scope)
    if component == 'feedbacktrace':
        if current_scope is None:
            return dict(error='Saved request did not contain a parseable complete trace view')
        events = read(bundle / 'trace/model_input.json')['events']
        checks = {}
        for event in events:
            eid = event.get('evidence_id')
            if not eid:
                continue
            items = trace_items.get(eid, [])
            # A full original event within current scope is sufficient for coverage.
            current = [x for x in items if x['scope'] == current_scope]
            checks[eid] = dict(present=bool(items), in_current_scope=bool(current),
                               complete=full_event_covered(event['content'], [x['content'] for x in current]),
                               source_chars=len(event['content']),
                               visible_chars=max((len(x['content']) for x in current), default=0),
                               locators=[{k:v for k,v in x.items() if k!='content'} for x in items])
        result['trace_checks'] = checks
    return result


def location_checks(locations, coverage, repository, exclusions=None):
    result = []
    for loc in locations:
        file = loc.get('file', loc.get('path'))
        source = repository / file
        if not source.is_file():
            result.append(dict(file=file, available=False, complete=False, missing=[]))
            continue
        lines = source.read_text(encoding='utf-8').splitlines()
        required = set()
        valid = bool(loc.get('line_ranges'))
        for span in loc.get('line_ranges', []):
            start, end = span.get('start', span.get('start_line')), span.get('end', span.get('end_line'))
            if not start or end < start or end > len(lines):
                valid = False
                continue
            required.update(n for n in range(start,end+1) if lines[n-1].strip())
        excluded = required & set((exclusions or {}).get(file, []))
        required -= excluded
        missing = required - set(coverage.get(file, []))
        result.append(dict(file=file, available=valid, complete=valid and bool(required or excluded) and not missing,
                           required=sorted(required), missing=sorted(missing), reviewed_exclusions=sorted(excluded)))
    return result


def classify(row, audit):
    if audit.get('error'):
        return None, audit['error'], {}
    gold = row['gold']
    if row['benchmark'] == 'feedbacktrace':
        checks = {e:audit['trace_checks'].get(e) for e in gold['evidence_ids']}
        if any(v is None for v in checks.values()):
            return None, 'Gold event missing from saved source bundle', checks
        if all(v['complete'] for v in checks.values()):
            return 'model_judgment', 'All Gold evidence events fully present in current task of actual input; disclosure still not fully matched by judge.', checks
        if any(not v['present'] or not v['in_current_scope'] for v in checks.values()):
            return 'evidence_presentation', 'At least one Gold event absent from allowed current task; includes task-scope assignment loss.', checks
        return None, 'Gold events present only as excerpts; semantic sufficiency of excerpts requires review.', checks
    repository = ROOT / 'data/prepared'/row['benchmark']/'artifacts/visible_bundles'/row['sample_id']/'repository'
    locations = gold['implementation_locations'] if row['benchmark']=='specgap' else [gold['localization']]
    rule=REVIEW_RULES.get(f"{row['sample_id']}/{row['key_id']}", {}) if row['benchmark']=='specgap' else {}
    checks = location_checks(locations, audit['covered_lines'], repository,rule.get('exclude'))
    if not checks or any(not x['available'] for x in checks):
        return None, 'Gold source file/line reference not fully resolvable in saved bundle', dict(locations=checks)
    if all(x['complete'] for x in checks):
        return 'model_judgment', 'All referenced implementation lines literally present in saved model requests; target identification or disclosure still not fully correct under the selected judge criterion.', dict(locations=checks)
    tests = location_checks(gold.get('test_locations', []), audit['covered_lines'], repository)
    if tests and any(x['complete'] for x in tests):
        return None, 'Implementation evidence incomplete but an alternative Gold test is visible; requires semantic sufficiency review.', dict(locations=checks,tests=tests)
    missing_files = sorted({x['file'] for x in checks if not x['complete']} - set(audit['selected_files']))
    if missing_files:
        return 'file_selection', 'Required Gold implementation files never returned by any successful read or other source block.', dict(locations=checks,missing_files=missing_files)
    return 'evidence_presentation', 'Referenced source files reached the run but some required Gold source lines did not appear in any saved request.', dict(locations=checks)


def main():
    rows = read(ROOT / 'outputs/analysis/failure_analysis/data/key_screening.json')
    adjudications = read(HERE / 'configs/user_adjudications.json')
    candidates = [r for r in rows if r['review_status']=='pending']
    by_run = defaultdict(list)
    for r in candidates:
        by_run[(r['benchmark'],r['model'],r['sample_id'])].append(r)
    records=[]
    judged_predictions={}
    for index,(key,group) in enumerate(by_run.items(),1):
        cache=ROOT / 'outputs/analysis/failure_analysis/data/request_audits'/('silentswap_stage1' if key[0]=='silentswap' else key[0])/key[1]/f'{key[2]}.json'
        audit=read(cache) if cache.exists() else None
        if audit is None or audit.get('version')!=VERSION:
            audit=audit_run(group[0]);audit['version']=VERSION;write(cache,audit)
        for row in group:
            judge_input=(ROOT/row['result_path']).with_name('judge_input.json')
            if judge_input not in judged_predictions:
                judged_predictions[judge_input]=read(judge_input).get('prediction') if judge_input.exists() else None
            if judged_predictions[judge_input] != row['prediction']:
                category,reason,evidence=None,'Saved judge input and audited run prediction differ; cannot link score to this request.',{}
            else:
                category,reason,evidence=classify(row,audit)
            row=dict(row,category=category,review_status='attributed_by_reference_coverage' if category else 'pending',
                     review_reason=reason,evidence_audit=evidence,
                     request_audit_path=str(cache.relative_to(ROOT)))
            row['evidence_review_rule']=REVIEW_RULES.get(f"{row['sample_id']}/{row['key_id']}") if row['benchmark']=='specgap' else None
            for decision in adjudications:
                if all(row[k] == decision[k] for k in ('benchmark', 'model', 'sample_id')) and row['key_id'] in decision['key_ids']:
                    row['prior_review_status'] = row['review_status']
                    row['prior_review_reason'] = row['review_reason']
                    row['category'] = decision['category']
                    row['review_status'] = 'attributed_by_user'
                    row['review_reason'] = decision['reason']
                    row['adjudication_source'] = 'data/user_adjudications.json'
            records.append(row)
        if index%25==0:print(f'Audited {index}/{len(by_run)} runs',flush=True)
    summaries=[]
    for s in read(ROOT / 'outputs/analysis/failure_analysis/data/screening_summary.json'):
        group=[r for r in records if all(r[k]==s[k] for k in ('benchmark','model','judge'))]
        count=Counter(r['category'] for r in group)
        n=sum(count[k] for k in CATEGORIES)
        summaries.append(dict(s,attributed=n,pending=count[None],file_selection='not_applicable' if s['benchmark']=='feedbacktrace' else 'applicable',categories={
            k:dict(count=count[k],percent=100*count[k]/n if n else None) for k in CATEGORIES}))
    write(ROOT / 'outputs/analysis/failure_analysis/data/attributed_keys.json',records)
    write(ROOT / 'outputs/analysis/failure_analysis/data/attribution_summary.json',summaries)
    print(json.dumps([dict(benchmark=s['benchmark'],model=s['model'],judge=s['judge'],attributed=s['attributed'],pending=s['pending'],categories=s['categories']) for s in summaries],indent=2))


if __name__=='__main__':
    main()
