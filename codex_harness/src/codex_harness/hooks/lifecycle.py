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
            # Completed verification needs a result check even when it failed.
            # Failure alone does not establish an intention to change the plan.
            if _verification_command(payload['tool_input']):
                if any(c.get('verification_call_id') == call_id for c in harness.checks.all()):
                    return {}
                recorded = [e for e in log.events(session) if e.get('call_id') == call_id]
                check = harness.checks.create('result', '验证命令已返回，核对实际执行、验证内容及结果分析。',
                                              event_ids=[e['id'] for e in recorded])
                check['verification_call_id'] = call_id
                harness.checks.put(check)
                return {'hookSpecificOutput': {'hookEventName': name, 'additionalContext':
                    f"BEG 检查点 {check['id']}：采用或汇报结果前，调用 beg_review(check_id=\"{check['id']}\")。"
                    '成功、失败或部分完成都需检查；核对实际执行、验证是否检验目标效果，以及比较和归因是否成立。'
                    '内部试验不必逐次披露；本轮汇报时说明实际完成情况及仍影响结论的重要限制。'
                    '若准备实质改变原要求，先用 adjustment/ambiguity 检查并按影响提前披露。'
                    '此检查点包含刚返回的验证，不要用执行前的旧检查点代替。'}}
            elif _failed(payload['tool_response']):
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
                        f"BEG 检查点 {check['id']}：调用 beg_review(check_id=\"{check['id']}\")。"
                        '失败信号不等于任务问题，也不代表必须向用户披露。'
                        '核对原要求、阻碍、拟采取的做法及影响；准备重要方案变更时在落实前披露，'
                        '已有授权内的普通修复可以继续，仍有重要限制则在本轮汇报时说明。'}}
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
            reason = (f"BEG 检查点 {check['id']}：结束前调用 beg_review(check_id=\"{check['id']}\")，"
                      '核对实际执行、验证内容和结果分析，失败或部分完成也要据实汇报；发现疑点即使尚不确定，也必须调用 beg_evidence 核实。'
                      '将实际完成情况及仍影响结论的重要限制随结果说明，不逐次重复内部失败。'
                      '核对后用 beg_record 记录 conclusion 和 summary，再完成汇报。'
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
        'additionalContext': f'BEG 正在记录 session_id={session_id}。采用或汇报结果前检查；'
                             '主要在本轮汇报时披露仍影响结论的重要问题，关键歧义或重要方案变更在落实前提前披露。'
                             '使用 beg_review(trigger=result/adjustment/ambiguity, focus=拟作出的声明或决定)；'
                             '收到检查点编号则直接读取。发现疑点必须调用 beg_evidence，即使尚不确定；核对后用 beg_record 保存 conclusion/summary。'
                             '只披露影响结论或决策的问题，不重复提醒。审查统一使用上述三个接口。' + reminder,
    }}
