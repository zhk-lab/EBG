"""Record execution and surface active checks without replacing tool results."""

from __future__ import annotations

import json
import re
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
        return _context(name, session, harness)
    if name == 'UserPromptSubmit':
        internal = any(c.get('stop_reason') == payload['prompt'] for c in harness.checks.all()
                       if c['session_id'] == session)
        log.start(session, payload['turn_id'], payload['cwd'], payload['prompt'], internal=internal)
        return _context(name, session, harness)
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
            if _failed(payload['tool_response']):
                if any(c.get('failure_call_id') == call_id for c in harness.checks.all()):
                    return {}  # Replayed delivery of the same failure, even after later events.
                recorded = [e for e in log.events(session) if e.get('call_id') == call_id]
                check = harness.checks.create('adjustment', '工具返回失败信号，判断是否影响后续方案或验证范围。',
                                               event_ids=[e['id'] for e in recorded])
                if not check.get('notified'):
                    check['notified'] = True
                    check['failure_call_id'] = call_id
                    harness.checks.put(check)
                    return {'hookSpecificOutput': {'hookEventName': name, 'additionalContext':
                        f"BEG 检查点 {check['id']}：调用 beg_context(check_id=\"{check['id']}\")。"
                        '失败信号不等于任务问题；先判断影响，必要时查代码并披露。'}}
            elif _verification_command(payload['tool_input']):
                if any(c.get('verification_call_id') == call_id for c in harness.checks.all()):
                    return {}
                recorded = [e for e in log.events(session) if e.get('call_id') == call_id]
                check = harness.checks.create('result', '验证命令已返回，核对实际验证范围与完成声明。',
                                              event_ids=[e['id'] for e in recorded])
                check['verification_call_id'] = call_id
                harness.checks.put(check)
                return {'hookSpecificOutput': {'hookEventName': name, 'additionalContext':
                    f"BEG 检查点 {check['id']}：在声称验证通过或完成前，调用 beg_context(check_id=\"{check['id']}\")。"
                    '对照原始成功条件、实际执行路径与详细结果；命令返回不等于要求满足。'
                    '此检查点包含刚返回的验证，不要用执行前的旧检查点代替。'}}
    elif name == 'Stop':
        # Stop continuation is a new prompt in Codex. Never request an endless
        # chain of reviews, and never close the turn before freezing its check.
        already_stopped = log._snapshot_for(session, turn['turn_id'], 'after')
        internal = any(e.get('internal') for e in log.events(session) if e['turn_id'] == turn['turn_id'])
        check = None
        if not already_stopped and not payload.get('stop_hook_active') and not internal:
            check = harness.checks.create('result', payload.get('last_assistant_message') or '准备结束当前任务并汇报结果。')
        elif already_stopped and not payload.get('stop_hook_active') and not internal:
            check = next((c for c in reversed(harness.checks.all())
                          if c['turn_id'] == turn['turn_id'] and c.get('stop_reason')), None)
        reason = None
        if check and not check['assessment']:
            reason = (f"BEG 检查点 {check['id']}：结束前调用 beg_context(check_id=\"{check['id']}\")，"
                      '核对准备汇报的结果；需要代码时调用 beg_evidence。'
                      '有影响结论的问题才主动披露；核对后用 beg_context 记录 conclusion 和 summary，再完成汇报。'
                      '这是系统续跑提示，不是新的用户任务要求。')
            if check.get('stop_reason') != reason:
                check['stop_reason'] = reason
                harness.checks.put(check)
        log.stop(session, turn['turn_id'], payload.get('last_assistant_message'))
        if reason:
            return {'decision': 'block', 'reason': reason}
    return {}


def _verification_command(tool_input: Any) -> bool:
    """Recognize direct Python verification invocations, not mentions in file reads."""
    if not isinstance(tool_input, dict):
        return False
    command = tool_input.get('command', tool_input.get('cmd', ''))
    if not isinstance(command, str):
        return False
    return bool(re.search(
        r'(?:^|[;\n]\s*)\s*(?:python(?:\d+(?:\.\d+)?)?(?:\.exe)?|py)\s+'
        r'(?:-m\s+(?:pytest|unittest)\b|(?:[\w./\\-]+[/\\])?'
        r'(?:smoke|test(?:_[\w-]+)?|verify|validate|check(?:_[\w-]+)?)\.py\b)', command))


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


def _context(event_name: str, session_id: str, harness: Harness) -> dict[str, Any]:
    pending = [c['id'] for c in harness.checks.all() if c['assessment'] is None]
    reminder = (' 尚未记录结论的检查点（最近三个）：' + '、'.join(pending[-3:])) if pending else ''
    return {'hookSpecificOutput': {
        'hookEventName': event_name,
        'additionalContext': f'BEG 正在记录 session_id={session_id}。采用结果或汇报前、重要方案调整或关键歧义时，'
                             '使用 beg_context(trigger=result/adjustment/ambiguity, focus=拟作出的声明或决定)；'
                             '收到检查点编号则直接读取。需要核实代码时调用 beg_evidence；核对后记录 conclusion/summary。'
                             '只披露影响结论或决策的问题，不重复提醒。执行普通任务使用上述两个接口；'
                             '只有用户另行要求完整审查时才使用 Skill 中的完整任务核对流程。' + reminder,
    }}
