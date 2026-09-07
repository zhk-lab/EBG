"""Freeze in-flight context, attach relevant code and support evidence follow-ups."""

from __future__ import annotations

import json
from importlib.resources import files as resource_files
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import yaml

from .repository import SOURCE_EXTENSIONS, capture
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
            'note': 'Apply to research/optimization conclusions. This is review guidance, not a user requirement or a finding.'}

REVIEW_GUIDANCE = {
    'result': (
        '从有效要求中找出拟采用或汇报的结论成立所需的前提，并对应实际执行证据。'
        '验证完成声明需对应实际执行路径、覆盖范围与详细结果，不能以通过数量代替；'
        '如要求真实外部调用，应核对实际请求及有效响应，替代实现的流程通过不满足该前提。'
        '比较提升时，优先检查相对基线改变的路径及其输入来源是否满足适用条件；'
        '评测文件固定、指标计算正确或命令成功，不足以证明实验可比。'
        '涉及额外数据、缓存或替代实现时，沿项目提供的来源和依赖关系核实，'
        '不能仅凭文件名、配置开关或表面编号判断条件成立。'
        '需要核实实现时，将尚未得到支持的具体前提与原文引用交给 beg_evidence。'
    ),
    'adjustment': (
        '对应受阻步骤、有效要求和拟采用的替代方案，检查替代方案是否改变验证范围、'
        '资源条件或可交付结论。已恢复的临时失败不继续当作交付问题；'
        '替代命令成功也不自动证明原验证要求已满足。'
    ),
    'ambiguity': (
        '区分用户明确要求、Agent 自拟计划与尚未确定的假设，'
        '检查不同解释是否影响目标、范围或实验可比性。'
        '没有明文规定不自动等于违规，也不自动等于任何实现都能支持原目标；'
        '按已有授权处理，只有需要用户决定时才提问。'
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

    def context(self, check_id: str | None = None, *, trigger: str | None = None, focus: str | None = None,
                event_ids: list[str] | None = None, plan_ids: list[str] | None = None,
                read_ref: str | None = None, offset: int = 0,
                conclusion: str | None = None, summary: str | None = None) -> str:
        if check_id and any(v is not None for v in (trigger, focus, event_ids, plan_ids)):
            raise HarnessError('Read an existing checkpoint or create a new one, not both.')
        if not check_id and (read_ref or offset or conclusion or summary):
            raise HarnessError('Continuation and assessment require check_id.')
        if read_ref and (conclusion is not None or summary is not None):
            raise HarnessError('Read saved context or record an assessment, not both.')
        if not read_ref and offset:
            raise HarnessError('Continuation requires read_ref.')
        if not check_id and (not trigger or not focus):
            raise HarnessError('Supply check_id from the hook, or trigger and focus.')
        check = self.get(check_id) if check_id else self.create(trigger, focus, event_ids, plan_ids)
        owner = f"context:{check['session_id']}"
        if read_ref:
            root = self.harness.pages.resolve(owner, read_ref.split('#')[0])
            if root.get('check_id') != check['id']:
                raise HarnessError('Context reference belongs to a different checkpoint.')
            return self.harness.pages.read(owner, read_ref, offset)
        if conclusion is not None or summary is not None:
            if conclusion not in {'clear', 'issue', 'uncertain'} or not summary or not summary.strip():
                raise HarnessError('Assessment needs clear/issue/uncertain and a nonempty summary.')
            check['assessment'] = {'conclusion': conclusion, 'summary': summary.strip(),
                                   'origin': 'agent judgment; not an independently verified verdict'}
            self.put(check)
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
        evidence = self._initial_evidence(check, trace)
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
                   'note': '先核对原文与附带代码；需要补查时调用 beg_evidence。无匹配不等于无行为。核对后用 beg_context 记录 conclusion 和 summary；读取本身不代表完成。',
                   'review_guidance': REVIEW_GUIDANCE[check['trigger']] +
                       '只披露证据支持且影响结论或决策的问题，不把尚未检查的可能性写成事实。',
                   'review_protocol': check.get('review_protocol', {}),
                   'prompts': [s for s in check['sources'] if s['kind'] == 'user'],
                   'plans': [s for s in check['sources'] if s['id'] in check['selected_plans']],
                   'code_evidence': evidence['code'],
                   'table_relations': evidence.get('tables', {}),
                   'related_materials': evidence['materials'],
                   'evidence_expansion': {
                       'read_ref': evidence['read_ref'],
                       'note': '上述代码与材料来自检查点快照。补查或展开这些证据时调用 beg_evidence(check_id, read_ref)，'
                               '不是 beg_context；引用关系不证明实际使用或条件合规。',
                   },
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

    def _initial_evidence(self, check: dict[str, Any], trace: list[dict[str, Any]]) -> dict[str, Any]:
        """Build once from original sources, so the first read includes code evidence."""
        if 'initial_evidence' in check:
            return check['initial_evidence']
        selected = set(check['selected_plans'])
        sources = [s for s in check['sources'] if s['kind'] == 'user' or s['id'] in selected]
        sources += [e for e in trace if e['kind'] == 'tool_call']
        refs = [{'source_id': s['id'], 'quote': s['content'], 'start': 0}
                for s in sources if s['content'].strip()]
        question = f"核对本次要求、执行入口及相关实现是否支持这一结果或决定：{check['focus']}"
        # Use the same persisted evidence path as explicit follow-up questions.
        page = yaml.safe_load(self.evidence(check['id'], question, refs))
        ref = page['read_ref'].split('#', 1)[0]
        root = self.harness.pages.resolve('evidence:check_' + check['id'], ref)
        check['initial_evidence'] = {
            'read_ref': ref,
            'code': [entry for group in root['evidence_groups'].values()
                     for entry in group.get('actual', {}).get('repo', [])
                     if (path := entry['source'].split(':', 1)[-1].split('::', 1)[0]) in check['files']
                     and Path(path).suffix.lower() in SOURCE_EXTENSIONS],
            'materials': root.get('linked_artifacts', {}).get('items', []),
            'tables': root.get('linked_artifacts', {}).get('table_relations', {}),
        }
        self.put(check)
        return check['initial_evidence']

    def evidence(self, check_id: str, question: str | None = None, refs: list[dict[str, Any]] | None = None,
                 *, read_ref: str | None = None, offset: int = 0) -> str:
        check = self.get(check_id)
        task_id = 'check_' + check_id
        if read_ref:
            if question is not None or refs is not None:
                raise HarnessError('Read saved evidence or submit a question, not both.')
            return self.harness.build_evidence_groups(task_id, read_ref=read_ref, offset=offset)
        if offset or not question or not question.strip():
            raise HarnessError('Provide a question (refs optional), or read_ref for continuation.')
        sources = check['sources'] + [
            {'id': e['id'], 'kind': 'trace', 'label': f"{e['kind']} {e.get('tool_name', '')}",
             'content': e['content']} for e in check['events'] if e['kind'] != 'user']
        if refs is None:
            # The checkpoint already fixes the original context. A precise question
            # should not fail simply because the caller did not copy an anchor.
            refs = [{'source_id': s['id'], 'quote': s['content'], 'start': 0}
                    for s in sources if (s['kind'] == 'user' or s['id'] in check['selected_plans'])
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
            # Initial context built this task before the model could cite its code.
            # Keep new evidence anchors available to subsequent normalization.
            existing = {s['id'] for s in task['sources']}
            additions = [s for s in sources if s['id'] not in existing]
            if additions:
                task['sources'].extend(additions)
                self.store.put_task(task)
        return self.harness.build_evidence_groups(task_id, [{'id': 'R1', 'check': question, 'refs': refs}])
