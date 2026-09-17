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
    if re.match(r'\s*(?:please\s+|请)?(?:explain\b|describe\b|review\b|summarize\b|解释|评审|总结|讨论)', prompt, re.I):
        return None
    if re.search(r'(?:不要|不必|do not|don.t)\s*(?:执行|implement|execute|follow)', prompt, re.I):
        return None
    if not re.search(r'(?:执行|按照|按|落实|\bimplement\b|\bexecute\b|\bfollow\b)[^\n。]{0,100}(?:plan|计划|方案|\.md\b)', prompt, re.I):
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


def _plan_check(harness: Harness, plan: str, event_ids: list[str] | None = None) -> dict[str, Any] | None:
    session = harness.sessions.current()
    # Plan identity includes captured content; a changed Plan gets a fresh check.
    path = harness.sessions.root(session) / plan
    content = (path.read_text(encoding='utf-8') if plan != '@context' and path.is_file()
               else next(e['content'] for e in reversed(harness.sessions.events(session))
                         if e['kind'] == 'user' and not e.get('internal')))
    request = next(e['content'] for e in reversed(harness.sessions.events(session))
                   if e['kind'] == 'user' and not e.get('internal'))
    with runtime(harness.store, session) as state:
        previous = state['plan_checks'].get(plan)
    if previous and previous['content'] == content and previous.get('request') == request:
        return None
    check = harness.checks.create('ambiguity', f'执行 Plan 前澄清目标、数据划分、基线和验收：{plan}',
                                  event_ids=event_ids)
    with runtime(harness.store, session) as state:
        state['plan_checks'][plan] = {'content': content, 'request': request, 'check_id': check['id']}
        state['plan_seen'] = True
    return _context('PostToolUse', f'EBG：使用 ebg-review。读取 ebg_review(check_id="{check["id"]}")，'
                    '必须 ebg_evidence 核实目标与实现前提；材料不足可继续读，执行 Plan 前完成检查。'
                    '需要用户选择时用 ebg_record(waiting_for_user=true) 记录并立即提问。')


def handle_hook(harness: Harness, payload: dict[str, Any]) -> dict[str, Any]:
    name, session = payload.get('hook_event_name'), payload.get('session_id')
    if not session:
        raise HarnessError('Hook input needs session_id.')
    log = harness.sessions
    if name == 'SessionStart':
        harness.store.activate_session(session)
        with runtime(harness.store, session) as state:
            # A restored process must not charge the disconnected interval as research time.
            state['active_since'] = None
        return _context(name, f'EBG session_id={session} autoresearch：使用 ebg-review 总 Skill；统计和检查按 session 保留。')
    if name == 'UserPromptSubmit':
        internal = any(c.get('stop_reason') == payload['prompt'] for c in harness.checks.all()
                       if c['session_id'] == session)
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
            return _context(name, 'EBG：仍有待澄清决定。核对本次答复；只有解决后才用 '
                            'ebg_record(resolution=答复如何解决问题) 清除等待状态，不能自动视为已确认。')
        if plan == '@context':
            result = _plan_check(harness, plan)
            if result:
                result['hookSpecificOutput']['hookEventName'] = name
                return result
        return _context(name, f'EBG session_id={session}：仅执行 Plan 时做 ambiguity；读完待执行 Plan 后检查。'
                        '未被 Hook 识别的 Plan 执行意图，由 Codex 主动 ebg_review(trigger="ambiguity", focus=具体计划)。'
                        '普通 Prompt 不做 ambiguity；执行中先记录，Stop 分别判断 adjustment/result，'
                        '只检查满足触发条件的阶段；两类都触发时先 adjustment、后 result。')
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
        call_id = f"{turn['number']}:{original_id}"
        event = {'kind': 'tool_call', 'call_id': call_id, 'original_call_id': original_id,
                 'tool_name': tool, 'content': _content(payload['tool_input'])}
        log.append(session, turn['turn_id'], event, f'call:{original_id}')
        if name == 'PostToolUse':
            log.append(session, turn['turn_id'], {**event, 'kind': 'tool_result',
                       'content': _content(payload['tool_response'])}, f'result:{original_id}')
        with runtime(harness.store, session) as state:
            if state['active_since'] is not None:
                pause(state)
                state['active_since'] = time.time()
            if call_id not in state['calls']:
                state['calls'].append(call_id)
            if name == 'PostToolUse' and _failed(payload['tool_response']):
                state['failed'] = True
            plan = state['plan']
        if (name == 'PostToolUse' and plan and not _failed(payload['tool_response'])
                and _read_plan(tool, payload['tool_input'], plan)):
            ids = [e['id'] for e in log.events(session) if e.get('call_id') == call_id]
            return _plan_check(harness, plan, ids) or {}
    elif name == 'Stop':
        return _stop(harness, payload, turn)
    return {}


def _stop(harness: Harness, payload: dict[str, Any], turn: dict[str, Any]) -> dict[str, Any]:
    session, log = payload['session_id'], harness.sessions
    with runtime(harness.store, session) as state:
        if turn['turn_id'] not in state['stops']:
            pause(state)
            state['stops'].append(turn['turn_id'])
        waiting = state['waiting']
        observations = state.get('observations', [])
        adjustment_needed = state['failed'] or any(
            note['kind'] in {'adjustment', 'ambiguity'} for note in observations)
        result_needed = (state['plan_seen'] or any(note['kind'] == 'limitation' for note in observations)
                         or len(state['calls']) >= harness.review_call_threshold
                         or elapsed(state) >= harness.review_seconds_threshold)
        batch = state['batch']
    checks = harness.checks.all()
    for check in checks:
        assessment = check['assessment']
        if check['trigger'] == 'ambiguity':
            result_needed = True
            if assessment and assessment['conclusion'] != 'clear':
                adjustment_needed = True
        elif not assessment or assessment['conclusion'] != 'clear':
            # Preserve explicitly requested or unresolved checks in their own stage.
            if check['trigger'] == 'adjustment':
                adjustment_needed = True
            elif check['trigger'] == 'result':
                result_needed = True
    triggers = [stage for stage, needed in (('adjustment', adjustment_needed), ('result', result_needed)) if needed]
    reason = None
    internal = bool(payload.get('stop_hook_active')) or any(
        e.get('internal') for e in log.events(session) if e['turn_id'] == turn['turn_id'])
    if not waiting and triggers:
        events = log.events(session)
        # Internal prompts and generated replies do not invalidate an experiment review.
        signature = [e['id'] for e in events if e['kind'] in {'tool_call', 'tool_result'}
                     or (e['kind'] == 'user' and not e.get('internal'))]
        files, notes, missing = capture(harness.store, log.root(session),
            max_file_bytes=harness.max_file_bytes, session_id=session)
        key = {'events': signature, 'files': files, 'missing': missing, 'notes': notes,
               'observations': observations, 'triggers': triggers}
        # A normal new report can change the claim; an internal continuation is the same report.
        if not internal:
            key['claim'] = payload.get('last_assistant_message', '')
        elif batch:
            key['claim'] = batch['key'].get('claim', '')
        if not batch or batch['key'] != key:
            ids = [e['id'] for e in events if e['kind'] in {'tool_call', 'tool_result'}]
            focuses = {'adjustment': '复核本 session 的实验调整、回退原因及可比性。',
                       'result': key.get('claim') or '复核实验结果、验证与改进归因。'}
            batch = {'key': key, 'checks': [harness.checks.create(stage, focuses[stage], event_ids=ids)['id']
                                            for stage in triggers]}
            with runtime(harness.store, session) as state:
                state['batch'] = batch
        pending = [identifier for identifier in batch['checks'] if not harness.checks.get(identifier)['assessment']]
        if pending and not payload.get('stop_hook_active') and not internal:
            reason = ('EBG autoresearch：结束前依次检查 ' + '、'.join(pending) +
                      '。对每项调用 ebg_review(check_id=...)，必须 ebg_evidence 核对代码、数据和执行验证证据，'
                      '再 ebg_record。只检查已触发的阶段；两类都触发时先 adjustment、后 result。最终完整回答原始实验任务，说明目标是否达成、'
                      '实际保留或回退的方案、关键依据及限制。不要只回复检查已完成或内部编号。'
                      '这是系统续跑提示，不是新的用户任务要求。')
            for identifier in batch['checks']:
                check = harness.checks.get(identifier)
                check['stop_reason'] = reason
                harness.checks.put(check)
    # A blocked Stop can resume tools in the same turn; freeze only when it ends.
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
