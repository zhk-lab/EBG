"""Freeze review inputs, investigate evidence on demand and record judgments."""

from __future__ import annotations

import json
from importlib.resources import files as resource_files
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from .repository import capture
from .requirements import normalize_requirements
from .storage import HarnessError, dumps

if TYPE_CHECKING:
    from .service import Harness


TRIGGERS = {'result', 'adjustment', 'ambiguity'}


def _review_protocol() -> dict[str, str]:
    skill = resource_files('codex_harness').joinpath('skills', 'beg-disclose', 'SKILL.md')
    _, marker, section = skill.read_text(encoding='utf-8').partition('## Autoresearch review\n')
    if not marker:
        raise HarnessError('beg-disclose Skill is missing its Autoresearch review section.')
    return {'source': 'beg-disclose/SKILL.md :: Autoresearch review',
            'content': section.split('\n## ', 1)[0].strip(),
            'note': 'Apply to research decisions and conclusions at the relevant disclosure stage. This is review guidance, not a user requirement or a finding.'}

REVIEW_GUIDANCE = {
    'result': (
        '按 Skill 的结果审查部分核对实际执行、实际验证和结果分析，成功、失败、部分完成或不确定均需检查。'
        '追踪实际执行分支，核实测试及断言是否检验成功条件，判断数据、配置、资源与结果选择是否支持提升归因。'
        '采用结果前先检查；不必每次内部试验都披露，在本轮汇报中说明实际完成情况与仍影响结论的重要限制。'
        '若下一步需要解释关键歧义或实质改变原要求，先创建 ambiguity 或 adjustment 检查，再作决定。'
    ),
    'adjustment': (
        '按 Skill 的提前披露部分，核对原要求、受阻证据、拟采取的替代方案、影响和已有授权。'
        '工具失败只是检查信号，不证明需要变更方案或向用户披露。'
        '准备作出影响原要求的重要调整时，在落实前说明问题、做法及影响；需要用户取舍时先澄清。'
        '普通重试或已修复并验证的临时失败不必反复报告，替代操作成功不等于原要求已满足。'
    ),
    'ambiguity': (
        '按 Skill 的提前披露部分，区分用户明确要求、Agent 自拟计划与未确定假设。'
        '明确歧义、拟采用的解释及其对目标、约束、验收或实验结论的影响，在落实关键解释前披露。'
        '没有明文规定不自动等于违规；已有授权内的常规决定可以继续，需要用户作出关键取舍时先澄清。'
    ),
}


class Checks:
    def __init__(self, harness: Harness) -> None:
        self.harness = harness
        self.store = harness.store
        self.log = harness.sessions

    def all(self) -> list[dict[str, Any]]:
        with self.store.connect() as db:
            return [json.loads(row[0]) for row in db.execute('SELECT data FROM checkpoints ORDER BY rowid')]

    def get(self, check_id: str) -> dict[str, Any]:
        with self.store.connect() as db:
            row = db.execute('SELECT data FROM checkpoints WHERE id=?', (check_id,)).fetchone()
        if row is None:
            raise HarnessError('Unknown check_id in the current session.')
        check = json.loads(row[0])
        self.log.current(check['session_id'])
        return check

    def put(self, check: dict[str, Any], *, new: bool = False) -> None:
        with self.store.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            self.store.require_session(db, check['session_id'])
            if new:
                # Capture must not silently include later execution events.
                if self.log._cutoff(check['session_id']) != check['cutoff']:
                    raise HarnessError('Events changed during checkpoint capture; retry.')
                check['revision'] = 1
                db.execute('INSERT INTO checkpoints VALUES (?, ?)', (check['id'], dumps(check)))
            else:
                row = db.execute('SELECT data FROM checkpoints WHERE id=?', (check['id'],)).fetchone()
                if row is None or json.loads(row[0])['revision'] != check['revision']:
                    raise HarnessError('Checkpoint changed concurrently; reload and retry.')
                check['revision'] += 1
                db.execute('UPDATE checkpoints SET data=? WHERE id=?', (dumps(check), check['id']))

    def create(self, trigger: str, focus: str, event_ids: list[str] | None = None,
               plan_ids: list[str] | None = None) -> dict[str, Any]:
        if trigger not in TRIGGERS or not focus.strip():
            raise HarnessError('Use result, adjustment or ambiguity and a nonempty focus.')
        session = self.log.current()
        turn = self.log.turn(session)
        if turn is None:
            raise HarnessError('Active checks need a recorded Prompt; enable hooks before execution.')
        events = self.log.events(session)
        cutoff = max((e['seq'] for e in events), default=0)
        anchors = sorted(set(event_ids or []))
        if set(anchors) - {e['id'] for e in events}:
            raise HarnessError('event_ids must refer to recorded events in this session.')
        files, notes, uncollected = capture(self.store, self.log.root(session),
            max_file_bytes=self.harness.max_file_bytes, session_id=session)
        snapshots = [s for s in self.log.snapshots(session) if s['event_cutoff'] <= cutoff]
        sources = [{'id': f"P{e['turn']}", 'kind': 'user', 'label': f"Prompt P{e['turn']}",
                    'content': e['content']} for e in events if e['kind'] == 'user' and not e.get('internal')]
        plans = {}
        for snap in snapshots + [{'files': files, 'snapshot_id': 'checkpoint'}]:
            for path, file_id in snap['files'].items():
                if Path(path).suffix.lower() in {'.md', '.markdown'}:
                    plans[f'L{file_id}'] = {'id': f'L{file_id}', 'kind': 'plan', 'path': path,
                        'label': f"{path} ({snap['snapshot_id']}; origin unknown)",
                        'content': self.store.file(file_id)['content']}
        if set(plan_ids or []) - plans.keys():
            raise HarnessError('plan_ids must be saved Markdown versions at this checkpoint.')
        # Direct path references select Plan candidates; other Markdown is available
        # for expansion, not silently declared a task requirement.
        reference_text = '\n'.join([focus, *[s['content'] for s in sources],
                                   *[e['content'] for e in events if e['id'] in anchors]])
        selected_plans = set(plan_ids or []) | {p['id'] for p in plans.values() if p['path'] in reference_text}
        key = {'trigger': trigger, 'focus': focus.strip(), 'event_ids': [e['id'] for e in events],
               'anchors': anchors, 'plans': sorted(selected_plans), 'files': files,
               'uncollected_files': uncollected, 'notes': notes}
        previous = next((c for c in reversed(self.all()) if c['session_id'] == session and c['key'] == key), None)
        if self.log._cutoff(session) != cutoff:
            raise HarnessError('Events changed during checkpoint capture; retry.')
        if previous:
            return previous
        before = next((s for s in snapshots if s['turn_id'] == turn['turn_id'] and s['phase'] == 'before'), None)
        check = {'id': 'C' + uuid4().hex[:12], 'session_id': session, 'turn_id': turn['turn_id'],
                 'trigger': trigger, 'focus': focus.strip(), 'cutoff': cutoff, 'key': key,
                 'files': files, 'notes': notes, 'uncollected_files': uncollected,
                 'baseline': before['files'] if before else {},
                 'baseline_uncollected_files': before.get('uncollected_files', {}) if before else {},
                 'events': events, 'sources': sources + list(plans.values()),
                 'selected_plans': sorted(selected_plans), 'anchors': anchors, 'snapshots': snapshots,
                 'assessment': None, 'review_protocol': _review_protocol()}
        self.put(check, new=True)
        return check

    def review(self, check_id: str | None = None, *, trigger: str | None = None, focus: str | None = None,
               event_ids: list[str] | None = None, plan_ids: list[str] | None = None) -> str:
        if check_id and any(v is not None for v in (trigger, focus, event_ids, plan_ids)):
            raise HarnessError('Read an existing checkpoint or create a new one, not both.')
        if not check_id and (not trigger or not focus):
            raise HarnessError('Supply check_id from the hook, or trigger and focus.')
        check = self.get(check_id) if check_id else self.create(trigger, focus, event_ids, plan_ids)
        owner = f"review:{check['session_id']}"
        events = check['events']
        anchors = set(check['anchors'])
        if not anchors:
            anchors = {e['id'] for e in events if e['turn_id'] == check['turn_id'] and e['kind'] != 'user'}
        calls = {e.get('call_id') for e in events if e['id'] in anchors} - {None}
        trace = [e for e in events if e['id'] in anchors or e.get('call_id') in calls]
        if check['trigger'] == 'ambiguity':
            # Include the question's whole turn, not an arbitrary last-N tail.
            turns = {e['turn_id'] for e in trace} or {check['turn_id']}
            trace = [e for e in events if e['turn_id'] in turns and e['kind'] != 'user']
        latest = next((c for c in reversed(self.all()) if c.get('verification_call_id')), None)
        newer = latest is not None and latest['cutoff'] > check['cutoff']
        payload = {'check_id': check['id'], 'trigger': check['trigger'], 'focus': check['focus'],
                   'cutoff': check['cutoff'], 'assessment': check['assessment'],
                   'execution_scope': {
                       'covers_latest_verification': not newer,
                       'latest_check_id': latest['id'] if latest else None,
                       'note': ('本检查点早于新的验证，不能支持该次验证的完成声明；请读取 latest_check_id。'
                                if newer else '仅核对当前冻结材料；之后的新执行需重新检查。'),
                   },
                   'note': '先核对要求、执行记录和检查条目，不默认展开代码。发现疑点，即使尚不确定，也必须调用 beg_evidence；核实后用 beg_record 保存判断。读取或记录不等于向用户披露。',
                   'review_guidance': REVIEW_GUIDANCE[check['trigger']] +
                       '只披露证据支持且影响结论或决策的问题，不把尚未检查的可能性写成事实。',
                   'review_protocol': check.get('review_protocol', {}),
                   'prompts': [s for s in check['sources'] if s['kind'] == 'user'],
                   'plans': [s for s in check['sources'] if s['id'] in check['selected_plans']],
                   'trace': trace,
                   'plan_candidates': [{k: s[k] for k in ('id', 'path', 'label')}
                                       for s in check['sources'] if s['kind'] == 'plan'],
                   'scope_note': 'Prompt 保留当前会话全部用户原文，可能含其他任务，由模型判断适用范围。默认 Trace 按关联调用或当前轮选取；完整上下文可展开。Markdown 候选不自动视为 Plan。',
                   'prior_assessments': [{'check_id': c['id'], 'focus': c['focus'], 'assessment': c['assessment']}
                                         for c in self.all() if c['id'] != check['id'] and c['assessment']
                                         and c['cutoff'] <= check['cutoff']],
                   'all_sources': check['sources'], 'all_trace': events,
                   'collection_notes': check['notes']}
        ref = self.harness.pages.save(owner, payload)
        # Avoid returning all Trace/Plan text twice; expose frozen expansion links.
        for key in ('all_sources', 'all_trace'):
            payload[key] = {'read_ref': f'{ref}#/{key}'}
        ref = self.harness.pages.save(owner, payload)
        return self.harness.pages.read(owner, ref)

    def record(self, check_id: str, conclusion: str, summary: str) -> str:
        """Save the agent's judgment without rereading sources or building evidence."""
        if conclusion not in {'clear', 'issue', 'uncertain'} or not summary or not summary.strip():
            raise HarnessError('Assessment needs clear/issue/uncertain and a nonempty summary.')
        check = self.get(check_id)
        check['assessment'] = {'conclusion': conclusion, 'summary': summary.strip(),
                               'origin': 'agent judgment; not an independently verified verdict'}
        self.put(check)
        return dumps({'check_id': check_id, 'recorded': True, 'conclusion': conclusion,
                      'note': 'Recorded agent judgment; this does not establish user-facing disclosure.'})

    def evidence(self, check_id: str, question: str | None = None, refs: list[dict[str, Any]] | None = None,
                 *, read_ref: str | None = None, offset: int = 0) -> str:
        check = self.get(check_id)
        task_id = 'check_' + check_id
        if read_ref:
            if question is not None or refs is not None:
                raise HarnessError('Read saved evidence or submit a question, not both.')
            # Review pages and implementation evidence share one public reader.
            owner = f"review:{check['session_id']}"
            try:
                root = self.harness.pages.resolve(owner, read_ref.split('#', 1)[0])
            except HarnessError:
                return self.harness.build_evidence_groups(task_id, read_ref=read_ref, offset=offset)
            if root.get('check_id') != check_id:
                raise HarnessError('Review reference belongs to a different checkpoint.')
            return self.harness.pages.read(owner, read_ref, offset)
        if offset or not question or not question.strip():
            raise HarnessError('Provide a question (refs optional), or read_ref for continuation.')
        sources = check['sources'] + [
            {'id': e['id'], 'kind': 'trace', 'label': f"{e['kind']} {e.get('tool_name', '')}",
             'content': e['content']} for e in check['events'] if e['kind'] != 'user']
        if refs is None:
            # The checkpoint already fixes the original context. A precise question
            # should not fail simply because the caller did not copy an anchor.
            calls = {e.get('call_id') for e in check['events'] if e['id'] in check['anchors']}
            trace_ids = {e['id'] for e in check['events'] if e['kind'] == 'tool_call'
                         and (e['turn_id'] == check['turn_id'] or e.get('call_id') in calls)}
            refs = [{'source_id': s['id'], 'quote': s['content'], 'start': 0}
                    for s in sources if (s['kind'] == 'user' or s['id'] in check['selected_plans'] or s['id'] in trace_ids)
                    and s['content'].strip()]
        known = {s['id'] for s in sources}
        for ref in refs:
            identifier = ref.get('source_id', '')
            if identifier in known or not identifier.startswith('V'):
                continue
            view_id, _, material_id = identifier.partition(':')
            view = self.store.view(view_id)
            if view['scope'].get('check_id') != check_id:
                raise HarnessError('Evidence reference belongs to a different checkpoint.')
            kind, _, path = material_id.partition(':')
            if kind not in {'repo', 'context', 'diff'}:
                raise HarnessError('Quote a returned repo/context/diff reference, or expand it with read_ref.')
            material = self.harness.material(view_id, material_id)
            if kind == 'context':
                path = material['file_ref'].split(':repo:', 1)[1]
            sources.append({'id': identifier, 'kind': 'repo', 'path': path,
                            'label': identifier, 'content': material['content']})
            known.add(identifier)
        normalized = normalize_requirements([{'id': 'R1', 'check': question, 'refs': refs}], sources)
        try:
            task = self.store.task(task_id)
        except HarnessError:
            task = {'id': task_id, 'repo_path': str(self.log.root(check['session_id'])),
                    'scope': {'mode': 'checkpoint', 'session_id': check['session_id'],
                              'check_id': check_id, 'event_ids': [e['id'] for e in check['events']],
                              'repo_before': 'turn start', 'repo_after': check_id},
                    'baseline': check['baseline'], 'current_files': check['files'], 'sources': sources,
                    'requirements': normalized, 'collection_notes': check['notes'],
                    'history_snapshots': check['snapshots'],
                    'uncollected_files': check['uncollected_files'],
                    'baseline_uncollected_files': check['baseline_uncollected_files']}
            self.store.put_task(task, new=True)
        else:
            # Earlier queries may have introduced code references.
            # Keep new evidence anchors available to subsequent normalization.
            existing = {s['id'] for s in task['sources']}
            additions = [s for s in sources if s['id'] not in existing]
            if additions:
                task['sources'].extend(additions)
                self.store.put_task(task)
        return self.harness.build_evidence_groups(task_id, [{'id': 'R1', 'check': question, 'refs': refs}])
