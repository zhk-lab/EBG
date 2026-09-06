"""Record session events; task selection happens later during disclosure."""

from __future__ import annotations

import json
from typing import Any

from ..application.service import Harness
from ..application.storage import HarnessError


def _content(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)


def handle_hook(harness: Harness, payload: dict[str, Any]) -> dict[str, Any]:
    name, session = payload.get('hook_event_name'), payload.get('session_id')
    if not session:
        raise HarnessError('Hook input needs session_id.')
    log = harness.sessions
    if name == 'SessionStart':
        harness.store.activate_session(session)
        return _context(name, session)
    if name == 'UserPromptSubmit':
        log.start(session, payload['turn_id'], payload['cwd'], payload['prompt'])
        return _context(name, session)
    if session != harness.store.current_session():
        return {}  # Late events from a replaced session must not restore old data.
    turn = log.turn(session, payload.get('turn_id'))
    if turn is None:
        return {}
    if name in {'PreToolUse', 'PostToolUse'}:
        tool = payload['tool_name']
        if tool.rsplit('__', 1)[-1].startswith('beg_'):
            return {}
        original_id = payload['tool_use_id']
        call_id = f"{turn['number']}:{original_id}"
        event = {'kind': 'tool_call', 'call_id': call_id, 'original_call_id': original_id,
                 'tool_name': tool, 'content': _content(payload['tool_input'])}
        log.append(session, turn['turn_id'], event, f'call:{original_id}')
        if name == 'PostToolUse':
            log.append(session, turn['turn_id'], {**event, 'kind': 'tool_result',
                       'content': _content(payload['tool_response'])}, f'result:{original_id}')
    elif name == 'Stop':
        log.stop(session, turn['turn_id'], payload.get('last_assistant_message'))
    return {}


def _context(event_name: str, session_id: str) -> dict[str, Any]:
    return {'hookSpecificOutput': {
        'hookEventName': event_name,
        'additionalContext': f'BEG 正在记录 session_id={session_id}。用户要求披露检查时使用 beg-disclose Skill：'
                             'beg_list_task_sources → beg_select_task → beg_build_evidence_groups。'
                             '选择 Prompt 区间和/或 Plan，至少一项非空；仅有 Plan 时核对当前 Repo。整理要求后核对证据。',
    }}
