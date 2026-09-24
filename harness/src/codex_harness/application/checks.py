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
from .runtime import runtime, pause

if TYPE_CHECKING:
    from .service import Harness


TRIGGERS = {'result', 'adjustment', 'ambiguity'}


def _review_protocol(trigger: str) -> dict[str, str]:
    name = {'result': 'ebg-result-review', 'ambiguity': 'ebg-ambiguity', 'adjustment': 'ebg-adjustment'}[trigger]
    resources = resource_files('codex_harness').joinpath('skills')
    skill = resources.joinpath(name, 'SKILL.md')
    _, marker, section = skill.read_text(encoding='utf-8').partition('## Autoresearch review\n')
    if not marker:
        raise HarnessError(f'{name} Skill is missing its Autoresearch review section.')
    shared = resources.joinpath('ebg-review', 'references', 'review-rules.md').read_text(encoding='utf-8')
    return {'source': f'{name}/SKILL.md :: Autoresearch review',
            'shared_source': 'ebg-review/references/review-rules.md',
            'content': section.split('\n## ', 1)[0].strip() + '\n\n' + shared.strip(),
            'note': 'Apply to research decisions and conclusions at the relevant disclosure stage. This is review guidance, not a user requirement or a finding.'}

REVIEW_GUIDANCE = {
    'result': (
        'Use ebg-result-review to check actual execution, verification, and result analysis, including failed, partial, and uncertain outcomes. Trace executed branches and verify that tests and assertions establish the success criteria. Check whether data, configuration, resources, and result selection support the claimed improvement. Review results before adopting them. Report completed work and material limitations in the task response. Every review requires ebg_evidence for relevant code, data, and execution evidence; scores and success markers do not establish experimental validity. Pause and clarify material ambiguity promptly. Disclosure alone does not resolve a decision that still requires user confirmation.'
    ),
    'adjustment': (
        'Use ebg-adjustment before completion to review the original requirements, blocking evidence, actual adjustment or fallback, consequences, and existing authorization. Tool failure is a review signal, not proof that a plan change or disclosure is required. Record authorized adjustments during execution and review their evidence before completion. Ask immediately when an unresolved tradeoff requires authorization. Routine retries and repaired, verified transient failures need no repeated reporting. A successful substitute does not establish that the original requirement was met.'
    ),
    'ambiguity': (
        'Use ebg-ambiguity before implementing a plan. Distinguish explicit user requirements, Agent proposals, and unresolved assumptions. Identify the ambiguity, proposed interpretation, and effects on goals, constraints, acceptance, or experimental conclusions. Clarify material choices before implementing them. Gather only the evidence needed to resolve the ambiguity while dependent work is paused. A default or disclosed assumption is not user confirmation. Record pending choices and pauses; do not mark a dependent decision clear merely because it was disclosed. Missing instructions do not automatically imply a violation. Continue routine authorized decisions and do not ask again about choices already resolved by existing instructions.'
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
        with runtime(self.store, session) as state:
            observations = state.get('observations', [])
        key = {'trigger': trigger, 'focus': focus.strip(), 'event_ids': [e['id'] for e in events],
               'anchors': anchors, 'plans': sorted(selected_plans), 'files': files,
               'uncollected_files': uncollected, 'notes': notes, 'observations': observations}
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
                 'observations': observations, 'assessment': None, 'review_protocol': _review_protocol(trigger)}
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
        # Checkpoints are scoped to the active session.  ``all()`` is a
        # diagnostic API and may include rows from another session when a
        # state database is inspected or migrated without the normal session
        # cleanup.  Never use a foreign result checkpoint to describe the
        # execution scope of this review.
        session_checks = [c for c in self.all() if c['session_id'] == check['session_id']]
        latest = next((c for c in reversed(session_checks) if c['trigger'] == 'result'), None)
        newer = any(e['seq'] > check['cutoff'] and e['kind'] == 'tool_result'
                    for e in self.log.events(check['session_id']))
        latest_result_seq = max((e['seq'] for e in self.log.events(check['session_id'])
                                 if e['kind'] == 'tool_result'), default=0)
        if latest and latest['cutoff'] < latest_result_seq:
            latest = None
        payload = {'check_id': check['id'], 'trigger': check['trigger'], 'focus': check['focus'],
                   'cutoff': check['cutoff'], 'assessment': check['assessment'],
                   'execution_scope': {
                       'covers_latest_verification': not newer,
                       'latest_check_id': latest['id'] if latest else None,
                       'note': ('This checkpoint precedes new tool execution. Read the applicable latest_check_id or create a checkpoint if none exists.'
                                if newer else 'Review only the frozen materials at this checkpoint. Subsequent execution requires another review.'),
                   },
                   'note': 'Every review requires ebg_evidence. For adjustment/result, inspect relevant code and execution evidence instead of relying only on traces or scores. Save the conclusion with ebg_record. Reading or recording does not itself disclose anything to the user.',
                   'review_guidance': REVIEW_GUIDANCE[check['trigger']] +
                       'Disclose only evidence-supported issues that affect conclusions or decisions. Do not present unchecked possibilities as facts.',
                   'review_protocol': check.get('review_protocol', {}),
                   'prompts': [s for s in check['sources'] if s['kind'] == 'user'],
                   'plans': [s for s in check['sources'] if s['id'] in check['selected_plans']],
                   'trace': trace,
                   'observations': check.get('observations', []),
                   'plan_candidates': [{k: s[k] for k in ('id', 'path', 'label')}
                                       for s in check['sources'] if s['kind'] == 'plan'],
                   'scope_note': 'Prompt preserves all user messages in the current session, potentially spanning several tasks; determine which apply. The default Trace selects linked calls or the current turn, and full context can be expanded. Markdown candidates are not automatically plans.',
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

    def record(self, check_id: str | None = None, conclusion: str | None = None, summary: str = '', *,
               waiting_for_user: bool = False, resolution: str | None = None,
               note_kind: str | None = None, decision_status: str = 'proposed') -> str:
        """Save the agent's judgment without rereading sources or building evidence."""
        if note_kind is not None:
            if (note_kind not in {'adjustment', 'ambiguity', 'limitation'} or not summary.strip()
                    or decision_status not in {'proposed', 'executed'}
                    or check_id is not None or conclusion is not None or waiting_for_user or resolution is not None):
                raise HarnessError('A process note needs note_kind, summary and proposed/executed status; no assessment fields.')
            session = self.log.current()
            note = {'kind': note_kind, 'summary': summary.strip(), 'status': decision_status,
                    'cutoff': self.log._cutoff(session)}
            with runtime(self.store, session) as state:
                observations = state.setdefault('observations', [])
                if note not in observations:
                    observations.append(note)
            return dumps({'recorded': True, 'note': 'The process event has been recorded for review before completion. This is neither a review conclusion nor execution authorization.'})
        if conclusion not in {'clear', 'issue', 'uncertain'} or not summary or not summary.strip():
            raise HarnessError('Assessment needs clear/issue/uncertain and a nonempty summary.')
        check = self.get(check_id)
        if not check.get('evidence_queries'):
            raise HarnessError('Call ebg_evidence with a concrete question before recording any assessment.')
        if waiting_for_user and (conclusion == 'clear' or resolution is not None):
            raise HarnessError('A pending user choice cannot be clear or resolved at the same time.')
        with runtime(self.store, check['session_id']) as state:
            waiting = state['waiting']
        if resolution is not None:
            if not resolution.strip() or not waiting or waiting['check_id'] != check_id:
                raise HarnessError('Resolve the existing waiting checkpoint with a nonempty explanation.')
            if not any(e['kind'] == 'user' and not e.get('internal') and e['seq'] > waiting['cutoff']
                       for e in self.log.events(check['session_id'])):
                raise HarnessError('Wait for a new user reply before resolving the choice.')
        elif waiting and conclusion == 'clear':
            raise HarnessError('Supply resolution after the user reply before clearing this choice.')
        check['assessment'] = {'conclusion': conclusion, 'summary': summary.strip(),
                               'waiting_for_user': waiting_for_user, 'resolution': resolution,
                               'origin': 'agent judgment; not an independently verified verdict'}
        self.put(check)
        with runtime(self.store, check['session_id']) as state:
            if check['trigger'] == 'ambiguity':
                state['plan_seen'] = True
            if waiting_for_user:
                state['waiting'] = {'check_id': check_id, 'question': summary.strip(),
                                    'cutoff': self.log._cutoff(check['session_id'])}
                pause(state)
            elif resolution is not None:
                state['waiting'] = None
        return dumps({'check_id': check_id, 'recorded': True, 'conclusion': conclusion,
                      'note': ('Ask the user for clarification now, end the response, and wait for an answer. Do not implement work that depends on the pending choice.'
                               if waiting_for_user else 'Recorded agent judgment; this does not establish user-facing disclosure.')})

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
        result = self.harness.build_evidence_groups(task_id, [{'id': 'R1', 'check': question, 'refs': refs}])
        check = self.get(check_id)
        queries = check.setdefault('evidence_queries', [])
        if question not in queries:
            queries.append(question)
            self.put(check)
        return result
