"""Record execution; review Plans after reading and experiments at Stop."""

from __future__ import annotations

import json
import re
import time
from typing import Any

from ..application.service import Harness
from ..application.storage import HarnessError
from ..application.runtime import runtime, pause, elapsed
from ..application.repository import capture


def _content(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)


def _context(name: str, message: str) -> dict[str, Any]:
    return {'hookSpecificOutput': {'hookEventName': name, 'additionalContext': message}}


def _execution_plan(prompt: str) -> str | None:
    """Recognize explicit execution requests only; ambiguous intent stays with the agent."""
    if re.match(r'\s*(?:please\s+|请)?(?:explain\b|describe\b|review\b|summarize\b|complete\b\s+(?:(?:a|an|the)\s+)?(?:review|summary|description|explanation)\b|解释|评审|总结|讨论)', prompt, re.I):
        return None
    if re.search(r'(?:不要|不必|do not|don.t)\s*(?:执行|implement|execute|follow|complete)', prompt, re.I):
        return None
    if not re.search(r'(?:执行|按照|按|落实|\bimplement\b|\bexecute\b|\bfollow\b|\bcomplete\b)[^\n。]{0,100}(?:plan|计划|方案|\.md\b)', prompt, re.I):
        return None
    paths = re.findall(r'[\w./\\-]+\.(?:md|markdown)\b', prompt, re.I)
    return paths[0] if paths else '@context'


def _read_plan(tool: str, inputs: Any, plan: str) -> bool:
    if not isinstance(inputs, dict) or plan == '@context':
        return False
    name = tool.rsplit('__', 1)[-1].lower()
    path = inputs.get('path', inputs.get('file_path', ''))
    if name in {'read', 'read_file'}:
        actual, expected = str(path).replace('\\', '/'), plan.replace('\\', '/')
        return actual == expected or actual.endswith('/' + expected)
    command = inputs.get('cmd', inputs.get('command', ''))
    return (isinstance(command, str) and plan.casefold() in command.casefold()
            and bool(re.search(r'(?:^|[;\n])\s*(?:Get-Content|cat|type|head|sed)\b', command, re.I)))


def _plan_check(harness: Harness, plan: str, event_ids: list[str] | None=None) -> dict[str, Any] | None:
    session = harness.sessions.current()
    path = harness.sessions.root(session) / plan
    content = path.read_text(encoding='utf-8') if plan != '@context' and path.is_file() else next((e['content'] for e in reversed(harness.sessions.events(session)) if e['kind'] == 'user' and (not e.get('internal'))))
    request = next((e['content'] for e in reversed(harness.sessions.events(session)) if e['kind'] == 'user' and (not e.get('internal'))))
    with runtime(harness.store, session) as state:
        previous = state['plan_checks'].get(plan)
    if previous and previous['content'] == content and (previous.get('request') == request):
        return None
    check = harness.checks.create('ambiguity', f'Before implementing the plan, clarify goals, data splits, baselines, and acceptance criteria: {plan}', event_ids=event_ids)
    with runtime(harness.store, session) as state:
        state['plan_checks'][plan] = {'content': content, 'request': request, 'check_id': check['id']}
        state['plan_seen'] = True
    return _context('PostToolUse', f'EBG: use ebg-review. Read ebg_review(check_id="{check["id"]}") and use ebg_evidence to verify goals and implementation assumptions. Expand materials as needed and finish the review before implementing the plan. If a user choice is required, record ebg_record(waiting_for_user=true) and ask immediately.')


def handle_hook(harness: Harness, payload: dict[str, Any]) -> dict[str, Any]:
    name, session = (payload.get('hook_event_name'), payload.get('session_id'))
    if not session:
        raise HarnessError('Hook input needs session_id.')
    log = harness.sessions
    if name == 'SessionStart':
        harness.store.activate_session(session)
        with runtime(harness.store, session) as state:
            state['active_since'] = None
        return _context(name, f'EBG session_id={session} autoresearch: use the ebg-review coordinator skill; retain statistics and checks per session.')
    if name == 'UserPromptSubmit':
        internal = any((c.get('stop_reason') == payload['prompt'] for c in harness.checks.all() if c['session_id'] == session))
        log.start(session, payload['turn_id'], payload['cwd'], payload['prompt'], internal=internal)
        plan = None if internal else _execution_plan(payload['prompt'])
        with runtime(harness.store, session) as state:
            if payload['turn_id'] not in state['prompts']:
                state['prompts'].append(payload['turn_id'])
                if state['active_since'] is None:
                    state['active_since'] = time.time()
                if not internal:
                    state['plan'] = plan
                if plan:
                    state['plan_seen'] = True
            waiting = state['waiting']
        if waiting:
            return _context(name, 'EBG: a decision still awaits clarification. Check the current reply and clear the waiting state with ebg_record(resolution=how_the_reply_resolves_the_issue) only when it resolves that decision. Do not assume confirmation.')
        if plan == '@context':
            result = _plan_check(harness, plan)
            if result:
                result['hookSpecificOutput']['hookEventName'] = name
                return result
        return _context(name, f'EBG session_id={session}: run ambiguity review only before implementing a plan, after reading the applicable plan. For plan execution intent missed by hooks, call ebg_review(trigger="ambiguity", focus=the_specific_plan). Ordinary prompts do not trigger ambiguity review. Record execution events first; at Stop, evaluate adjustment/result triggers separately and check only triggered stages, adjustment before result when both apply.')
    if session != harness.store.current_session():
        return {}
    turn = log.turn(session, payload.get('turn_id'))
    if turn is None:
        return {}
    if name in {'PreToolUse', 'PostToolUse'}:
        tool = payload['tool_name']
        if tool.rsplit('__', 1)[-1].startswith('ebg_'):
            return {}
        original_id = payload['tool_use_id']
        call_id = f'{turn["number"]}:{original_id}'
        event = {'kind': 'tool_call', 'call_id': call_id, 'original_call_id': original_id, 'tool_name': tool, 'content': _content(payload['tool_input'])}
        log.append(session, turn['turn_id'], event, f'call:{original_id}')
        if name == 'PostToolUse':
            log.append(session, turn['turn_id'], {**event, 'kind': 'tool_result', 'content': _content(payload['tool_response'])}, f'result:{original_id}')
        with runtime(harness.store, session) as state:
            if state['active_since'] is not None:
                pause(state)
                state['active_since'] = time.time()
            if call_id not in state['calls']:
                state['calls'].append(call_id)
            if name == 'PostToolUse' and _failed(payload['tool_response']):
                state['failed'] = True
            plan = state['plan']
        if name == 'PostToolUse' and plan and (not _failed(payload['tool_response'])) and _read_plan(tool, payload['tool_input'], plan):
            ids = [e['id'] for e in log.events(session) if e.get('call_id') == call_id]
            return _plan_check(harness, plan, ids) or {}
    elif name == 'Stop':
        return _stop(harness, payload, turn)
    return {}


def _stop(harness: Harness, payload: dict[str, Any], turn: dict[str, Any]) -> dict[str, Any]:
    session, log = (payload['session_id'], harness.sessions)
    with runtime(harness.store, session) as state:
        if turn['turn_id'] not in state['stops']:
            pause(state)
            state['stops'].append(turn['turn_id'])
        waiting = state['waiting']
        observations = state.get('observations', [])
        adjustment_needed = state['failed'] or any((note['kind'] in {'adjustment', 'ambiguity'} for note in observations))
        result_needed = state['plan_seen'] or any((note['kind'] == 'limitation' for note in observations)) or len(state['calls']) >= harness.review_call_threshold or (elapsed(state) >= harness.review_seconds_threshold)
        batch = state['batch']
    checks = [check for check in harness.checks.all() if check['session_id'] == session]
    for check in checks:
        assessment = check['assessment']
        if check['trigger'] == 'ambiguity':
            result_needed = True
            if assessment and assessment['conclusion'] != 'clear':
                adjustment_needed = True
        elif not assessment or assessment['conclusion'] != 'clear':
            if check['trigger'] == 'adjustment':
                adjustment_needed = True
            elif check['trigger'] == 'result':
                result_needed = True
    triggers = [stage for stage, needed in (('adjustment', adjustment_needed), ('result', result_needed)) if needed]
    reason = None
    internal = bool(payload.get('stop_hook_active')) or any((e.get('internal') for e in log.events(session) if e['turn_id'] == turn['turn_id']))
    if internal:
        # Only one Stop continuation is allowed. Preserve its actual ending state
        # without creating another review batch that the agent cannot execute.
        log.stop(session, turn['turn_id'], payload.get('last_assistant_message'))
        return {}
    if not waiting and triggers:
        events = log.events(session)
        signature = [e['id'] for e in events if e['kind'] in {'tool_call', 'tool_result'} or (e['kind'] == 'user' and (not e.get('internal')))]
        files, notes, missing = capture(harness.store, log.root(session), max_file_bytes=harness.max_file_bytes, session_id=session)
        key = {'events': signature, 'files': files, 'missing': missing, 'notes': notes, 'observations': observations, 'triggers': triggers}
        key['claim'] = payload.get('last_assistant_message', '')
        if not batch or batch['key'] != key:
            ids = [e['id'] for e in events if e['kind'] in {'tool_call', 'tool_result'}]
            focuses = {'adjustment': 'Review experimental adjustments, fallback reasons, and comparability for this session.', 'result': key.get('claim') or 'Review experimental results, verification, and attribution of improvements.'}
            batch = {'key': key, 'checks': [harness.checks.create(stage, focuses[stage], event_ids=ids)['id'] for stage in triggers]}
            with runtime(harness.store, session) as state:
                state['batch'] = batch
        pending = [identifier for identifier in batch['checks'] if not harness.checks.get(identifier)['assessment']]
        if pending:
            reason = 'EBG autoresearch: before completion, review in order: ' + '、'.join(pending) + '. For each item call ebg_review(check_id=...), inspect code, data, and execution evidence with ebg_evidence, then save ebg_record. Check only triggered stages, adjustment before result when both apply. The final response must answer the original experimental task, explain whether its goal was met, and state retained or reverted choices, supporting evidence, and limitations. Do not respond only with internal check IDs or completion notices. This is a system continuation reminder, not a new user requirement.'
            for identifier in batch['checks']:
                check = harness.checks.get(identifier)
                check['stop_reason'] = reason
                harness.checks.put(check)
    if not reason:
        log.stop(session, turn['turn_id'], payload.get('last_assistant_message'))
    return {'decision': 'block', 'reason': reason} if reason else {}


def _failed(response: Any) -> bool:
    """Recognize structured failures and standard shell envelopes, not arbitrary prose."""
    if isinstance(response, dict):
        if response.get('isError') is True:
            return True
        for key in ('exit_code', 'exitCode'):
            code = response.get(key)
            if isinstance(code, int) and not isinstance(code, bool) and code != 0:
                return True
    elif isinstance(response, str):
        try:
            decoded = json.loads(response)
        except ValueError:
            return bool(re.search(r'(?m)^Process exited with code (?!0\b)-?\d+\s*$', response))
        return _failed(decoded) if isinstance(decoded, dict) else False
    return False
