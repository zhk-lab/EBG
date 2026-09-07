"""Freeze historical Prompt intervals or current code for Plan-only inspection."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

from .repository import capture
from .sessions import SessionLog
from .storage import HarnessError


def select_plan_task(log: SessionLog, session_id: str, plan_ids: list[str]) -> dict[str, Any]:
    """Freeze current code for a Plan-only review; there is no historical baseline."""
    if not plan_ids or len(set(plan_ids)) != len(plan_ids):
        raise HarnessError('Select at least one Plan, with no duplicate versions.')
    root = log.root(session_id)
    sources = []
    for identifier in plan_ids:
        if not identifier.startswith('L') or not identifier[1:].isdigit():
            raise HarnessError('Use Plan IDs from beg_list_task_sources.')
        plan = log.store.file(int(identifier[1:]))
        if Path(plan['root']) != root or Path(plan['path']).suffix.lower() not in {'.md', '.markdown'}:
            raise HarnessError('Plan must belong to the selected repository and be a Markdown file.')
        sources.append({'id': identifier, 'kind': 'plan', 'path': plan['path'],
                        'label': f"{plan['path']} (selected version; origin unknown)", 'content': plan['content']})
    if not any(s['content'].strip() for s in sources):
        raise HarnessError('Prompt and Plan cannot both be empty; selected Plans contain no text.')
    files, notes, uncollected = capture(log.store, root, max_file_bytes=log.max_file_bytes, session_id=session_id)
    identifier = uuid4().hex[:12]
    return {'id': f'task_{identifier}', 'repo_path': str(root), 'baseline': {}, 'current_files': files,
            'sources': sources, 'requirements': [], 'collection_notes': notes, 'uncollected_files': uncollected,
            'scope': {'mode': 'current', 'session_id': session_id, 'plan_ids': plan_ids,
                      'repo_after': f'current_{identifier}', 'event_ids': []}}


def select_task(log: SessionLog, session_id: str, start_prompt: str, end_prompt: str,
                plan_ids: list[str]) -> dict[str, Any]:
    events = log.events(session_id)
    prompts = {f"P{e['turn']}": e for e in events if e['kind'] == 'user' and not e.get('internal')}
    start, end = prompts.get(start_prompt), prompts.get(end_prompt)
    if not start or not end or start['turn'] > end['turn']:
        raise HarnessError("Select existing start_prompt/end_prompt in chronological order.")
    snapshots = log.snapshots(session_id)
    before = next((s for s in snapshots if s['turn_id'] == start['turn_id'] and s['phase'] == 'before'), None)
    after = next((s for s in snapshots if s['turn_id'] == end['turn_id'] and s['phase'] == 'after'), None)
    if not before or not after:
        raise HarnessError("Historical boundary snapshot missing; select a completed end turn. Live files cannot replace history.")
    selected = [e for e in events if start['turn'] <= e['turn'] <= end['turn'] and e['seq'] <= after['event_cutoff']]
    sources = [{'id': f"P{e['turn']}", 'kind': 'user', 'label': f"Prompt P{e['turn']}", 'content': e['content']}
               for e in selected if e['kind'] == 'user' and not e.get('internal')]
    available = {p['id']: p for p in log.plans(session_id, cutoff=after['event_cutoff'], last_turn=end['turn'])}
    if len(set(plan_ids)) != len(plan_ids):
        raise HarnessError("Select each Plan version only once.")
    for identifier in plan_ids:
        plan = available.get(identifier)
        if not plan:
            raise HarnessError(f"Plan {identifier} was not captured before the selected end; use a listed historical version.")
        sources.append({'id': identifier, 'kind': 'plan', 'path': plan['path'],
                        'label': f"{plan['path']} ({plan['snapshot_id']}; origin unknown)",
                        'content': log.store.file(plan['file_id'])['content']})
    scope = {'session_id': session_id, 'start_prompt': start_prompt, 'end_prompt': end_prompt,
             'plan_ids': plan_ids, 'repo_before': before['snapshot_id'], 'repo_after': after['snapshot_id'],
             'event_ids': [e['id'] for e in selected]}
    notes = list(dict.fromkeys([*before['notes'], *after['notes']]))
    return {'id': f"task_{uuid4().hex[:12]}", 'repo_path': str(log.root(session_id)),
            'baseline': before['files'], 'sources': sources, 'requirements': [],
            'collection_notes': notes, 'scope': scope,
            'uncollected_files': after.get('uncollected_files', {}),
            'baseline_uncollected_files': before.get('uncollected_files', {})}
