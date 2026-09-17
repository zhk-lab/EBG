"""Behavior Atomization: build deterministic minimal Behaviors."""

from __future__ import annotations

import ast
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Iterable

from .core.errors import BehaviorError
from .core.model import BehaviorCandidate, RepoArtifact, VisibleBundle
from .core.syntax import parse_python
from .evidence_intake import source_symbol_spans


# ---------------------------------------------------------------------------
# Public behavior construction
# ---------------------------------------------------------------------------

RESULT_KINDS = {"return", "raise", "state_write", "output", "external_call", "yield"}
ARTIFACT_KINDS = {"source", "runtime_template", "executable", "configuration"}


def build_behaviors(
    bundle: VisibleBundle,
    evidence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if bundle.benchmark == "feedbacktrace":
        behaviors = _build_trace_behaviors(evidence)
    else:
        behaviors = _build_code_behaviors(bundle, evidence)
    validate_behaviors(bundle, evidence, behaviors)
    return behaviors


def validate_behaviors(
    bundle: VisibleBundle,
    evidence: list[dict[str, Any]],
    behaviors: list[dict[str, Any]],
) -> None:
    known = {str(item["evidence_id"]): item for item in evidence}
    if len(known) != len(evidence):
        raise BehaviorError("Evidence IDs must be unique")
    prefix = "T" if bundle.benchmark == "feedbacktrace" else "B"
    expected_ids = [f"{prefix}{index:04d}" for index in range(1, len(behaviors) + 1)]
    if [item.get("behavior_id") for item in behaviors] != expected_ids:
        raise BehaviorError("Behavior IDs must be contiguous and deterministic")
    for behavior in behaviors:
        if bundle.benchmark == "feedbacktrace":
            _validate_trace_behavior(behavior, known)
        else:
            _validate_repo_behavior(behavior, known)
    if bundle.benchmark == "feedbacktrace":
        _validate_trace_scope_order(behaviors)


def _build_code_behaviors(
    bundle: VisibleBundle,
    evidence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    evidence_index = EvidenceLineIndex(evidence)
    by_path: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in evidence:
        if item["source_type"] == "code":
            by_path[str(item["locator"]["path"])].append(item)
    artifacts = {artifact.path: artifact for artifact in bundle.repo_artifacts}
    candidates: list[BehaviorCandidate] = []
    for path in sorted(by_path):
        artifact = artifacts.get(path)
        if artifact is None:
            raise BehaviorError(f"Evidence has no retained artifact: {path}")
        if _is_python(artifact):
            try:
                candidates.extend(analyze_python_artifact(artifact, evidence_index))
            except SyntaxError:
                continue
        elif artifact.kind == "configuration":
            for item in by_path[path]:
                locator = item["locator"]
                candidates.append(
                    BehaviorCandidate(
                        path=path,
                        symbol=str(locator["symbol"]),
                        result_kind="state_write",
                        result_line=int(locator["line_start"]),
                        evidence_ids=(str(item["evidence_id"]),),
                    )
                )
        elif artifact.kind == "runtime_template":
            for item in by_path[path]:
                content = str(item["content"]).strip()
                if re.fullmatch(r"\{%.*%}", content, flags=re.DOTALL):
                    continue
                locator = item["locator"]
                candidates.append(
                    BehaviorCandidate(
                        path=path,
                        symbol=str(locator["symbol"]),
                        result_kind="output",
                        result_line=int(locator["line_start"]),
                        evidence_ids=(str(item["evidence_id"]),),
                    )
                )
    return _materialize_repo_behaviors(bundle, evidence, candidates)


def _build_trace_behaviors(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: list[list[dict[str, Any]]] = []
    for item in evidence:
        event_type = item["locator"]["event_type"]
        if event_type == "user_prompt":
            groups.append([])
        if groups:
            groups[-1].append(item)
    result: list[dict[str, Any]] = []
    task_profiles: dict[str, dict[str, Any]] = {}
    current_task_id: str | None = None
    sequence_by_task: dict[str, int] = defaultdict(int)
    for group in groups:
        if len(group) < 2:
            continue
        first = group[0]
        events = [
            item
            for item in group[1:]
            if item["locator"]["event_type"] in {
                "assistant_response", "tool_exchange"
            }
        ]
        if not events:
            continue
        demand_spans = _trace_demand_spans(str(first["content"]))
        partitions = _partition_trace_events(
            str(first["content"]), demand_spans, events
        )
        if partitions is None:
            partitions = [
                (
                    None,
                    [item for item in events if item["locator"]["event_type"] == "tool_exchange"],
                    [item for item in events if item["locator"]["event_type"] == "assistant_response"],
                )
            ]
        drafts = [
            (char_range, actions, responses)
            for char_range, actions, responses in partitions
            if actions or responses
        ]
        shared_task_id: str | None = None
        if _trace_partitions_share_task(str(first["content"]), drafts):
            shared_task_id = _trace_task_id(
                str(first["content"]),
                [item for _, actions, _ in drafts for item in actions],
                [item for _, _, responses in drafts for item in responses],
                current_task_id,
                task_profiles,
            )
        for char_range, actions, responses in drafts:
            if not actions and not responses:
                continue
            demand_content = str(first["content"])
            if char_range is not None:
                demand_content = demand_content[char_range[0]:char_range[1]]
            task_id = shared_task_id or _trace_task_id(
                demand_content,
                actions,
                responses,
                current_task_id,
                task_profiles,
            )
            current_task_id = task_id
            _update_trace_task_profile(
                task_profiles,
                task_id,
                demand_content,
                actions,
                responses,
            )
            sequence_by_task[task_id] += 1
            referenced = [first, *actions, *responses]
            demand_ref: dict[str, Any] = {
                "evidence_id": str(first["evidence_id"]),
            }
            if char_range is not None:
                demand_ref["char_range"] = list(char_range)
            result.append(
                {
                    "behavior_id": f"T{len(result) + 1:04d}",
                    "task_id": task_id,
                    "sequence_index": sequence_by_task[task_id],
                    "start_turn": min(int(item["locator"]["turn"]) for item in referenced),
                    "end_turn": max(int(item["locator"]["turn"]) for item in referenced),
                    "demand_refs": [demand_ref],
                    "action_evidence_ids": [str(item["evidence_id"]) for item in actions],
                    "response_refs": [_trace_response_ref(item) for item in responses],
                }
            )
    return result


_FOLLOWUP_PREFIX = re.compile(
    r"^(?:yes\b|no\b|ok(?:ay)?\b|sure\b|exactly\b|right\b|agreed\b|do it\b|apply\b|continue\b|"
    r"proceed\b|go on\b|try that\b|verify (?:it|this|that)\b|check (?:it|this|that|again)\b|"
    r"next\b|now\s+(?:let's|lets|we can|please)\b|only keep\b|"
    r"implement\s+(?:(?:option|item)\s+)?\d+\b|[1-9]\b|"
    r"(?:check|see)\s+(?:the\s+)?(?:latest|new|last)\s+(?:review\s+)?comments?\b|"
    r"got it\b|i see\b|great\b|awesome\b|thanks?\b|actually\b|and\b|also\b|then\b|"
    r"[a-e]\b|"
    r"好(?:的)?|是的|对|没错|明白|继续|进行|就这样|再试|然后|同意|实际上|"
    r"はい|いいえ|了解|お願い(?:します)?|進めて|続けて|そのまま|そう|いいですね|"
    r"ok)",
    re.IGNORECASE,
)
_STATUS_FOLLOWUP = re.compile(
    r"^(?:how(?:'s| is) (?:it|the (?:job|run|eval|evaluation)) going|"
    r"how(?:'s| is) the progress|how are we doing|what about now|any (?:result|progress|completed)|"
    r"monitor (?:the )?progress|"
    r"进展(?:如何|怎么样)|现在怎么样|完成了多少|有结果了吗)",
    re.IGNORECASE,
)
_SHORT_CONTEXT_FOLLOWUP = re.compile(
    r"^(?:why|how|how (?:do i|can (?:i|we)|to) (?:fix|resolve)(?: it| this| that)?|"
    r"what about (?:that|it|this)|为什么|怎么|那这个呢|这个呢)[?？.!。]?$",
    re.IGNORECASE,
)
_CONTEXT_REFERENCE = re.compile(
    r"\b(?:this|that|these|those|it|there|the same|the previous|the above|the rest|"
    r"the (?:job|run|eval|evaluation|result|results|comment|plan|fix|change))\b|"
    r"(?:这个|那个|这些|那些|上述|之前的|刚才的|同一个|其余|剩余|该任务|该结果|是不是因为)|"
    r"(?:これ|それ|あれ|この|その|前の|上記|同じ|全部|どれ|もっと|もう一度)",
    re.IGNORECASE,
)
_OUTCOME_FOLLOWUP = re.compile(
    r"\b(?:failed|failure|still (?:fails|failing|broken|exists)|does not work|doesn't work|"
    r"not working|problem still|workflow failed|error persists|"
    r"now\b.{0,60}\b(?:password|credentials))\b|"
    r"(?:仍然失败|还是失败|还是不行|问题仍然|问题依旧|又报错|工作流失败)|"
    r"(?:まだ|失敗|エラー|見えていない|見えない)",
    re.IGNORECASE,
)
_CROSS_TASK_INPUT = re.compile(
    r"(?:based on|using|from|according to) (?:the )?"
    r"(?:previous|earlier|prior) (?:result|decision|choice|output|artifact|constraint)|"
    r"(?:根据|基于|使用|沿用)(?:之前|先前|已有|上一个)(?:结果|决定|选择|产物|约束)",
    re.IGNORECASE,
)
_TASK_SWITCH = re.compile(
    r"\b(?:shift (?:our )?attention to|move on to|switch to|start (?:a )?new task)\b|"
    r"\bcome back to .{0,80} later.{0,120}\b(?:shift|switch|move on)\b|"
    r"(?:先|暂时)(?:别|不要|停止|放下).{0,80}(?:改|转|换|处理)|"
    r"(?:换成|转向|另一个任务)",
    re.IGNORECASE | re.DOTALL,
)
_RESUME_PREFIX = re.compile(
    r"^(?:verify|check again|re-?check|re-?test|re-?run|resume|follow up|"
    r"验证|重新检查|重新测试|重跑|恢复|继续处理)",
    re.IGNORECASE,
)
_ASSISTANT_REQUESTS_INPUT = re.compile(
    r"(?:[?？]\s*(?:$|\n))|"
    r"\b(?:which (?:option|approach)|what do you|would you|should we|do you want|want me to|can you|"
    r"please (?:choose|pick|provide|share)|let me know|choose [a-e]|waiting for your)\b|"
    r"(?:请选择|你希望|你想要|是否要|告诉我你的选择|どれ|どちら|選んで|どうしますか)",
    re.IGNORECASE | re.MULTILINE,
)
_ISSUE_OR_PR_URL = re.compile(
    r"https?://[^\s`]+/(?:issues|pull)/\d+",
    re.IGNORECASE,
)
_HASH_REFERENCE = re.compile(r"(?<![A-Za-z0-9])#\d{2,}\b")
_EXPLICIT_ID = re.compile(
    r"\b(?P<label>task|job|run|execution)[ _-]?id[`\"']?\s*[=:：#-]?\s*[`\"']?"
    r"(?P<value>[A-Za-z0-9][A-Za-z0-9._/-]{0,79})|"
    r"\b(?P<series>trial|shard)\s*[=:：#-]?\s*(?P<number>\d+[A-Za-z0-9._-]*)|"
    r"\bbranch\s+(?P<branch>[A-Za-z0-9][A-Za-z0-9._/-]{0,79})",
    re.IGNORECASE,
)
_TRACE_OBJECT = re.compile(
    r"\$[A-Za-z_][A-Za-z0-9_]*|"
    r"https?://[^\s`]+|"
    r"[A-Za-z]:\\[^\s`]+|"
    r"(?:^|(?<=\s)|(?<=[`'\"]))(?:[A-Za-z0-9_.-]+/){1,}[A-Za-z0-9_./-]+|"
    r"`([^`\n]{2,120})`",
    re.IGNORECASE | re.MULTILINE,
)
_DEMAND_MARKER = re.compile(
    r"(?m)^(?P<indent>[ \t]*)(?:[-*•]|\d+[.)]|[A-Za-z][.)])\s+"
)
_INLINE_DEMAND_MARKER = re.compile(
    r"(?<=[.!?。！？]) (?P<bullet>[-*])\s+(?=\*\*)"
)
_TRACE_WORD = re.compile(r"\$?[A-Za-z0-9_./\\-]{3,}|[\u4e00-\u9fff]{2,}")
_TRACE_STOPWORDS = {
    "and", "the", "this", "that", "with", "from", "into", "then", "also",
    "please", "should", "could", "would", "change", "update", "make", "add",
    "remove", "tool", "invocation", "result", "response", "assistant", "user",
    "complete", "completed", "done", "修改", "调整", "然后", "这个", "那个",
    "需要", "可以", "进行",
}

_TOOL_OPERATION = {
    "read": "inspect",
    "grep": "inspect",
    "glob": "inspect",
    "search": "inspect",
    "webfetch": "inspect",
    "websearch": "inspect",
    "edit": "modify",
    "write": "modify",
    "create": "modify",
    "delete": "modify",
    "notebookedit": "modify",
}
_OPERATION_PATTERNS = {
    "publish": re.compile(
        r"\b(?:commit(?:ted)?|push(?:ed)?|publish(?:ed)?|deploy(?:ed)?|open(?:ed)?\s+(?:a\s+)?pr)\b|"
        r"(?:提交|推送|发布|部署)",
        re.IGNORECASE,
    ),
    "verify": re.compile(
        r"\b(?:test(?:ed|ing)?|pytest|unittest|verify|verified|validat(?:e|ed|ion)|"
        r"check(?:ed|ing)?|lint(?:ed|ing)?|mypy|ruff|preflight|build)\b|"
        r"(?:测试|验证|检查|校验)",
        re.IGNORECASE,
    ),
    "modify": re.compile(
        r"\b(?:add(?:ed|ing)?|chang(?:e|ed|ing)|edit(?:ed|ing)?|fix(?:ed|ing)?|"
        r"implement(?:ed|ing)?|remov(?:e|ed|ing)|delet(?:e|ed|ing)|replac(?:e|ed|ing)|"
        r"rewrit(?:e|ten|ing)|writ(?:e|ten|ing)|creat(?:e|ed|ing)|set)\b|"
        r"(?:修改|新增|添加|删除|替换|实现|写入|创建|设置)",
        re.IGNORECASE,
    ),
    "inspect": re.compile(
        r"\b(?:read|review(?:ed|ing)?|inspect(?:ed|ing)?|search(?:ed|ing)?|"
        r"find|found|look(?:ed|ing)?|investigat(?:e|ed|ing)|compar(?:e|ed|ing)?)\b|"
        r"(?:读取|查看|审查|搜索|查找|调查|比较)",
        re.IGNORECASE,
    ),
}
_SHELL_VERIFY = re.compile(
    r"(?:^|[\s;&|])(?:pytest|python\s+-m\s+(?:pytest|unittest|mypy)|"
    r"ruff|mypy|tox|nox|npm\s+(?:run\s+)?test|pnpm\s+(?:run\s+)?test|"
    r"cargo\s+test|go\s+test|.*(?:test|check|lint|preflight).*\.py)\b",
    re.IGNORECASE,
)
_SHELL_PUBLISH = re.compile(
    r"\bgit\s+(?:commit|push|tag)\b|\bgh\s+pr\s+create\b|\bdeploy\b",
    re.IGNORECASE,
)
_SHELL_MODIFY = re.compile(
    r"(?:^|[\s;&|])(?:rm|del|move|mv|copy|cp|mkdir|touch)\b|"
    r"(?:^|[\s;&|])git\s+(?:add|restore|checkout|switch|merge|rebase)\b",
    re.IGNORECASE,
)
_INDEPENDENT_QUESTION = re.compile(r"[^?？\n]+[?？](?:\s*$|\s*\n)")
_RESPONSE_SENTENCE = re.compile(
    r".+?(?:[!?\u3002\uff01\uff1f]+(?=\s|$)|"
    r"\.(?=\s+(?=[A-Z0-9\u4e00-\u9fff])|$)|\n+|$)",
    re.DOTALL,
)
_PATH_TOKEN = re.compile(
    r"(?:[A-Za-z]:[\\/]|\.?\.?[\\/])[^\s`'\"<>|]+|"
    r"(?:[A-Za-z0-9_.-]+[\\/])+(?:[A-Za-z0-9_.-]+)",
    re.IGNORECASE,
)
_FILE_TOKEN = re.compile(
    r"(?<![A-Za-z0-9_.-])[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)+(?::\d+)?"
)
_IDENTIFIER_TOKEN = re.compile(
    r"(?<![A-Za-z0-9])(?:[A-Za-z_][A-Za-z0-9_]*\.)+[A-Za-z_][A-Za-z0-9_]*"
)
_TOOL_OBJECT_KEYS = {
    "file", "file_path", "filepath", "path", "paths", "symbol", "line", "line_number",
    "test", "test_name", "branch", "job_id", "run_id", "execution_id", "task_id",
    "issue", "issue_id", "pr", "pr_id", "url", "command", "cmd", "pattern", "query",
}


@dataclass(frozen=True, slots=True)
class _TraceSignal:
    operations: frozenset[str]
    objects: frozenset[str]
    terms: frozenset[str]


def _trace_text_operations(content: str) -> frozenset[str]:
    return frozenset(
        operation
        for operation, pattern in _OPERATION_PATTERNS.items()
        if pattern.search(content)
    )


def _trace_object_aliases(value: str) -> set[str]:
    normalized = value.strip("`'\"()[]{}.,:; ").casefold().replace("\\", "/")
    if not normalized or len(normalized) > 240:
        return set()
    aliases = {normalized}
    if "/" in normalized:
        parts = [part for part in normalized.split("/") if part and part not in {".", ".."}]
        if parts:
            aliases.add(parts[-1])
        if len(parts) >= 2:
            aliases.add("/".join(parts[-2:]))
        if len(parts) >= 3:
            aliases.add("/".join(parts[-3:]))
    return {item for item in aliases if len(item) >= 2}


def _trace_text_objects(content: str) -> frozenset[str]:
    objects: set[str] = set()
    for value in _trace_stable_keys(content) | _trace_object_keys(content):
        objects.update(_trace_object_aliases(value))
    for match in _PATH_TOKEN.finditer(content):
        objects.update(_trace_object_aliases(match.group(0)))
    for match in _FILE_TOKEN.finditer(content):
        objects.update(_trace_object_aliases(match.group(0)))
    for match in _IDENTIFIER_TOKEN.finditer(content):
        objects.update(_trace_object_aliases(match.group(0)))
    return frozenset(objects)


def _trace_tool_payload(content: str) -> dict[str, Any] | None:
    invocation = content.split("\n\nTool result:", 1)[0]
    if "\n" not in invocation:
        return None
    candidate = invocation.split("\n", 1)[1].strip()
    try:
        value = json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def _trace_payload_objects(value: Any, *, key: str | None = None) -> set[str]:
    objects: set[str] = set()
    if isinstance(value, dict):
        for child_key, child in value.items():
            objects.update(_trace_payload_objects(child, key=str(child_key).casefold()))
        return objects
    if isinstance(value, list):
        for child in value:
            objects.update(_trace_payload_objects(child, key=key))
        return objects
    if key not in _TOOL_OBJECT_KEYS or not isinstance(value, (str, int)):
        return objects
    text = str(value)
    objects.update(_trace_object_aliases(text))
    objects.update(_trace_text_objects(text))
    return objects


def _trace_tool_operation(item: dict[str, Any]) -> str:
    tool_name = str(item["locator"].get("tool_name") or "").casefold()
    compact_name = re.sub(r"[^a-z]", "", tool_name)
    for name, operation in _TOOL_OPERATION.items():
        if compact_name == name or compact_name.endswith(name):
            return operation
    invocation = _trace_profile_content(item)
    if _SHELL_PUBLISH.search(invocation):
        return "publish"
    if _SHELL_VERIFY.search(invocation):
        return "verify"
    if _SHELL_MODIFY.search(invocation):
        return "modify"
    if "task" in compact_name or "agent" in compact_name:
        return "execute"
    return "execute"


def _trace_signal(item: dict[str, Any]) -> _TraceSignal:
    content = str(item["content"])
    char_range = item.get("_behavior_char_range")
    if char_range is not None:
        content = content[char_range[0]:char_range[1]]
    event_type = str(item["locator"]["event_type"])
    objects = set(_trace_text_objects(_trace_profile_content(item)))
    if event_type == "tool_exchange":
        payload = _trace_tool_payload(content)
        if payload is not None:
            objects.update(_trace_payload_objects(payload))
        operations = frozenset({_trace_tool_operation(item)})
    else:
        operations = _trace_text_operations(content)
    return _TraceSignal(operations, frozenset(objects), frozenset(_trace_terms(content)))


def _combine_trace_signals(items: list[dict[str, Any]]) -> _TraceSignal:
    signals = [_trace_signal(item) for item in items]
    return _TraceSignal(
        frozenset(operation for signal in signals for operation in signal.operations),
        frozenset(obj for signal in signals for obj in signal.objects),
        frozenset(term for signal in signals for term in signal.terms),
    )


def _trace_signal_score(left: _TraceSignal, right: _TraceSignal) -> int:
    object_overlap = len(left.objects & right.objects)
    operation_overlap = len(left.operations & right.operations)
    term_overlap = len(left.terms & right.terms)
    return object_overlap * 100 + operation_overlap * 10 + min(term_overlap, 5)


def _trace_signal_matches(left: _TraceSignal, right: _TraceSignal) -> bool:
    """Return whether two visible signals identify the same fulfillment unit."""

    if left.objects and right.objects:
        return bool(left.objects & right.objects)
    return bool(
        left.operations & right.operations
        and left.terms & right.terms
    )


def _trace_task_id(
    demand_content: str,
    actions: list[dict[str, Any]],
    responses: list[dict[str, Any]],
    current_task_id: str | None,
    task_profiles: dict[str, dict[str, Any]],
) -> str:
    normalized = demand_content.strip()
    demand_keys = _trace_stable_keys(demand_content)
    stable_keys = _trace_stable_keys(
        "\n".join(
            [demand_content, *(_trace_profile_content(item) for item in actions)]
        )
    )
    if _TASK_SWITCH.search(normalized):
        return _new_trace_task_id(task_profiles)
    if _CROSS_TASK_INPUT.search(normalized):
        return _new_trace_task_id(task_profiles)
    demand_matches = _trace_stable_task_matches(demand_keys, task_profiles)
    if len(demand_matches) == 1:
        return demand_matches[0]
    if current_task_id is not None and (
        _FOLLOWUP_PREFIX.match(normalized)
        or _SHORT_CONTEXT_FOLLOWUP.fullmatch(normalized)
    ):
        return current_task_id
    if current_task_id is not None and _STATUS_FOLLOWUP.match(normalized):
        return current_task_id
    if current_task_id is not None and bool(task_profiles[current_task_id]["awaiting_user"]):
        return current_task_id
    if (
        current_task_id is not None
        and (
            _OUTCOME_FOLLOWUP.search(normalized[:200])
            or (
                len(normalized) <= 500
                and _OUTCOME_FOLLOWUP.search(normalized)
            )
        )
    ):
        return current_task_id
    signature_matches = [
        task_id
        for task_id, profile in task_profiles.items()
        if _trace_demand_signature(demand_content) in profile["demand_signatures"]
    ]
    if len(signature_matches) == 1:
        return signature_matches[0]
    object_keys = _trace_object_keys(demand_content)
    if current_task_id is not None and _CONTEXT_REFERENCE.search(normalized):
        contextual = _trace_unique_object_match(object_keys, task_profiles)
        if contextual is None:
            contextual = _trace_unique_term_match(
                _trace_terms(demand_content), task_profiles, minimum_overlap=3
            )
        return contextual or current_task_id
    stable_matches = _trace_stable_task_matches(stable_keys, task_profiles)
    if len(stable_matches) == 1:
        return stable_matches[0]
    object_match = _trace_unique_object_match(object_keys, task_profiles)
    if object_match is not None:
        return object_match
    if _RESUME_PREFIX.match(normalized):
        resumed = _trace_unique_term_match(
            _trace_terms(demand_content), task_profiles, minimum_overlap=2
        )
        if resumed is not None:
            return resumed
    return _new_trace_task_id(task_profiles)


def _new_trace_task_id(task_profiles: dict[str, dict[str, Any]]) -> str:
    return f"K{len(task_profiles) + 1:04d}"


def _trace_stable_keys(content: str) -> set[str]:
    keys = {match.group(0).casefold().rstrip(".,);]") for match in _ISSUE_OR_PR_URL.finditer(content)}
    keys.update(match.group(0).casefold() for match in _HASH_REFERENCE.finditer(content))
    for match in _EXPLICIT_ID.finditer(content):
        if match.group("label") is not None:
            keys.add(f"{match.group('label').casefold()}:{match.group('value').casefold()}")
        elif match.group("series") is not None:
            keys.add(f"{match.group('series').casefold()}:{match.group('number').casefold()}")
        else:
            branch = match.group("branch").casefold()
            if branch not in {"are", "is", "do", "does", "should", "to", "from", "for", "on"}:
                keys.add(f"branch:{branch}")
    return keys


def _trace_object_keys(content: str) -> set[str]:
    keys: set[str] = set()
    for match in _TRACE_OBJECT.finditer(content):
        value = match.group(1) or match.group(0)
        value = value.strip("`'\"()[]{}.,:; ").casefold().replace("\\", "/")
        if value:
            keys.add(value)
    return keys


def _trace_demand_signature(content: str) -> str:
    return " ".join(content.casefold().split())


def _trace_stable_task_matches(
    stable_keys: set[str],
    task_profiles: dict[str, dict[str, Any]],
) -> list[str]:
    if not stable_keys:
        return []
    scored = [
        (len(stable_keys & set(profile["stable_keys"])), task_id)
        for task_id, profile in task_profiles.items()
    ]
    best = max((score for score, _ in scored), default=0)
    return [task_id for score, task_id in scored if score == best and score > 0]


def _trace_unique_term_match(
    terms: set[str],
    task_profiles: dict[str, dict[str, Any]],
    *,
    minimum_overlap: int,
) -> str | None:
    scored: list[tuple[int, str]] = []
    for task_id, profile in task_profiles.items():
        overlap = max(
            (len(terms & snapshot) for snapshot in profile["term_snapshots"]),
            default=0,
        )
        if overlap >= minimum_overlap:
            scored.append((overlap, task_id))
    best = max((score for score, _ in scored), default=0)
    winners = [task_id for score, task_id in scored if score == best]
    return winners[0] if len(winners) == 1 else None


def _trace_unique_object_match(
    object_keys: set[str],
    task_profiles: dict[str, dict[str, Any]],
) -> str | None:
    if not object_keys:
        return None
    scored = [
        (len(object_keys & set(profile["object_keys"])), task_id)
        for task_id, profile in task_profiles.items()
    ]
    best = max((score for score, _ in scored), default=0)
    minimum = 1 if any(key.startswith("$") for key in object_keys) else 2
    winners = [task_id for score, task_id in scored if score == best and score >= minimum]
    return winners[0] if len(winners) == 1 else None


def _update_trace_task_profile(
    task_profiles: dict[str, dict[str, Any]],
    task_id: str,
    demand_content: str,
    actions: list[dict[str, Any]],
    responses: list[dict[str, Any]],
) -> None:
    profile = task_profiles.setdefault(
        task_id,
        {
            "stable_keys": set(),
            "object_keys": set(),
            "demand_signatures": set(),
            "term_snapshots": [],
            "awaiting_user": False,
        },
    )
    visible_content = "\n".join(
        [
            demand_content,
            *(_trace_profile_content(item) for item in actions),
            *(str(item["content"]) for item in responses),
        ]
    )
    profile["stable_keys"].update(_trace_stable_keys(visible_content))
    profile["object_keys"].update(_trace_object_keys(visible_content))
    profile["demand_signatures"].add(_trace_demand_signature(demand_content))
    profile["term_snapshots"].append(_trace_terms(demand_content))
    profile["term_snapshots"] = profile["term_snapshots"][-3:]
    if responses:
        recent_responses = "\n".join(
            str(item["content"])[-2000:] for item in responses[-3:]
        )
        profile["awaiting_user"] = bool(
            _ASSISTANT_REQUESTS_INPUT.search(recent_responses)
        )

def _trace_partitions_share_task(
    user_content: str,
    drafts: list[
        tuple[
            tuple[int, int] | None,
            list[dict[str, Any]],
            list[dict[str, Any]],
        ]
    ],
) -> bool:
    if len(drafts) < 2:
        return False
    # Multiple fulfillment units derived from one unsplit demand are one task.
    if all(char_range is None for char_range, _, _ in drafts):
        return True
    keys_by_draft = []
    for char_range, actions, _ in drafts:
        demand = user_content
        if char_range is not None:
            demand = demand[char_range[0]:char_range[1]]
        keys_by_draft.append(
            _trace_stable_keys(
                "\n".join([demand, *(_trace_profile_content(item) for item in actions)])
            )
        )
    nonempty = [keys for keys in keys_by_draft if keys]
    # A single prompt normally describes one task. Only explicit, mutually
    # distinct stable IDs justify assigning its demand parts to separate tasks.
    return not (
        len(nonempty) == len(drafts)
        and all(
            left.isdisjoint(right)
            for index, left in enumerate(nonempty)
            for right in nonempty[index + 1:]
        )
    )


def _trace_demand_spans(content: str) -> list[tuple[int, int]]:
    markers = [
        (
            match.start(),
            match.end(),
            len(match.group("indent").replace("\t", "    ")),
        )
        for match in _DEMAND_MARKER.finditer(content)
    ]
    markers.extend(
        (match.start(), match.end(), 0)
        for match in _INLINE_DEMAND_MARKER.finditer(content)
    )
    markers.sort()
    if markers:
        outermost = min(indent for _, _, indent in markers)
        markers = [
            (start, end, indent)
            for start, end, indent in markers
            if indent == outermost
        ]
    if len(markers) >= 2:
        spans: list[tuple[int, int]] = []
        for index, (_, marker_end, _) in enumerate(markers):
            start = marker_end
            end = markers[index + 1][0] if index + 1 < len(markers) else len(content)
            while start < end and content[start].isspace():
                start += 1
            while end > start and content[end - 1].isspace():
                end -= 1
            if start < end:
                spans.append((start, end))
        if len(spans) >= 2:
            return spans
    questions = [match.span() for match in _INDEPENDENT_QUESTION.finditer(content)]
    return questions if len(questions) >= 2 else []


def _trace_terms(content: str) -> set[str]:
    terms: set[str] = set()
    for match in _TRACE_WORD.finditer(content):
        term = match.group(0).casefold().replace("\\", "/").strip("./-")
        if term and term not in _TRACE_STOPWORDS:
            terms.add(term)
            # Preserve recognizable command/file components so a result such
            # as "preflight failed" can match run_pr_preflight.py without
            # requiring the response to repeat the complete path.
            terms.update(
                part
                for part in re.split(r"[/_.-]+", term)
                if len(part) >= 3 and part not in _TRACE_STOPWORDS
            )
    return terms


def _trace_profile_content(item: dict[str, Any]) -> str:
    content = str(item["content"])
    char_range = item.get("_behavior_char_range")
    if char_range is not None:
        content = content[char_range[0]:char_range[1]]
    if item["locator"]["event_type"] == "tool_exchange":
        content = content.split("\n\nTool result:", 1)[0]
    return content


def _partition_trace_events(
    user_content: str,
    demand_spans: list[tuple[int, int]],
    events: list[dict[str, Any]],
) -> list[
    tuple[tuple[int, int] | None, list[dict[str, Any]], list[dict[str, Any]]]
] | None:
    if len(demand_spans) >= 2:
        explicit = _partition_explicit_trace_demands(
            user_content, demand_spans, events
        )
        if explicit is not None:
            return explicit
    return None


def _partition_explicit_trace_demands(
    user_content: str,
    demand_spans: list[tuple[int, int]],
    events: list[dict[str, Any]],
) -> list[tuple[tuple[int, int], list[dict[str, Any]], list[dict[str, Any]]]] | None:
    demand_signals = [
        _TraceSignal(
            _trace_text_operations(user_content[start:end]),
            _trace_text_objects(user_content[start:end]),
            frozenset(_trace_terms(user_content[start:end])),
        )
        for start, end in demand_spans
    ]
    assigned: list[list[dict[str, Any]]] = [[] for _ in demand_spans]
    unassigned: list[list[dict[str, Any]]] = []
    actions = [
        event for event in events if _trace_event_type(event) == "tool_exchange"
    ]
    responses = [
        event for event in events if _trace_event_type(event) == "assistant_response"
    ]
    event_groups = _trace_action_groups(actions)
    for response in responses:
        split_response = _split_response_across_demands(response, demand_signals)
        if split_response is None:
            event_groups.append([response])
            continue
        for winner, fragment in split_response:
            assigned[winner].append(fragment)
    event_groups.sort(key=lambda group: _trace_event_position(group[0]))
    for group in event_groups:
        winner = _unique_trace_signal_match(
            _combine_trace_signals(group),
            demand_signals,
            allow_incidental_result_object=(
                _trace_event_type(group[0]) == "assistant_response"
            ),
        )
        if winner is None:
            unassigned.append(group)
        else:
            assigned[winner].extend(group)

    # Every demand needs its own observable fulfillment anchor. Otherwise the
    # list is descriptive context and remains one composite Behavior.
    if any(not bucket for bucket in assigned):
        return None
    for group in unassigned:
        assigned[_nearest_trace_bucket(group[0], assigned)].extend(group)
    for bucket in assigned:
        bucket.sort(key=_trace_event_position)
    return [
        (
            span,
            [
                event
                for event in bucket
                if _trace_event_type(event) == "tool_exchange"
            ],
            [
                event
                for event in bucket
                if _trace_event_type(event) == "assistant_response"
            ],
        )
        for span, bucket in zip(demand_spans, assigned)
    ]


def _split_response_across_demands(
    response: dict[str, Any],
    demand_signals: list[_TraceSignal],
) -> list[tuple[int, dict[str, Any]]] | None:
    """Split a short multi-result reply only when clauses match distinct demands."""

    content = str(response["content"])
    if len(content) > 2000 or "```" in content:
        return None
    spans = []
    for match in _RESPONSE_SENTENCE.finditer(content):
        start, end = match.span()
        while start < end and content[start].isspace():
            start += 1
        while end > start and content[end - 1].isspace():
            end -= 1
        if start < end:
            spans.append((start, end))
    if len(spans) < 2:
        return None

    winners = [
        _unique_trace_signal_match(
            _TraceSignal(
                _trace_text_operations(content[start:end]),
                _trace_text_objects(content[start:end]),
                frozenset(_trace_terms(content[start:end])),
            ),
            demand_signals,
            allow_incidental_result_object=True,
        )
        for start, end in spans
    ]
    matched = [(index, winner) for index, winner in enumerate(winners) if winner is not None]
    if len({winner for _, winner in matched}) < 2:
        return None

    for index, winner in enumerate(winners):
        if winner is None:
            _, nearest_winner = min(
                matched,
                key=lambda item: abs(item[0] - index),
            )
            winners[index] = nearest_winner

    result: list[tuple[int, dict[str, Any]]] = []
    for (start, end), winner in zip(spans, winners):
        if winner is None:
            raise AssertionError("response fragment assignment must be complete")
        fragment = dict(response)
        fragment["_behavior_char_range"] = (start, end)
        result.append((winner, fragment))
    return result


def _trace_action_groups(
    actions: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    signals: list[_TraceSignal] = []
    for action in actions:
        signal = _trace_signal(action)
        winner = _unique_trace_signal_match(signal, signals)
        if winner is None:
            groups.append([action])
            signals.append(signal)
        else:
            groups[winner].append(action)
            signals[winner] = _combine_trace_signals(groups[winner])
    return groups


def _unique_trace_signal_match(
    signal: _TraceSignal,
    candidates: list[_TraceSignal],
    *,
    allow_incidental_result_object: bool = False,
) -> int | None:
    scores = [
        _trace_signal_score(signal, candidate)
        if (
            _trace_signal_matches(signal, candidate)
            or (
                allow_incidental_result_object
                and signal.operations & candidate.operations
                and signal.terms & candidate.terms
            )
        )
        else 0
        for candidate in candidates
    ]
    best = max(scores, default=0)
    if best <= 0:
        return None
    winners = [index for index, score in enumerate(scores) if score == best]
    return winners[0] if len(winners) == 1 else None


def _nearest_trace_bucket(
    event: dict[str, Any],
    buckets: list[list[dict[str, Any]]],
) -> int:
    position = _trace_event_position(event)
    return min(
        range(len(buckets)),
        key=lambda index: min(
            abs(position[0] - _trace_event_position(candidate)[0])
            for candidate in buckets[index]
        ),
    )


def _trace_event_type(event: dict[str, Any]) -> str:
    return str(event["locator"]["event_type"])


def _trace_event_position(event: dict[str, Any]) -> tuple[int, int]:
    locator = event["locator"]
    return int(locator["event_index"]), int(locator["turn"])


def _trace_response_ref(item: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"evidence_id": str(item["evidence_id"])}
    char_range = item.get("_behavior_char_range")
    if char_range is not None:
        result["char_range"] = list(char_range)
    return result


def _materialize_repo_behaviors(
    bundle: VisibleBundle,
    evidence: list[dict[str, Any]],
    candidates: list[BehaviorCandidate],
) -> list[dict[str, Any]]:
    known = {str(item["evidence_id"]): item for item in evidence}
    artifacts = {item.path: item for item in bundle.repo_artifacts}
    spans = source_symbol_spans(bundle)
    unique: dict[tuple[Any, ...], dict[str, Any]] = {}
    for candidate in candidates:
        if candidate.result_kind not in RESULT_KINDS:
            continue
        trigger_ids, operation_ids, result_ids = _split_evidence_roles(candidate, known)
        if not result_ids:
            continue
        role_ids = [*trigger_ids, *operation_ids, *result_ids]
        role_lines = [
            (
                int(known[evidence_id]["locator"]["line_start"]),
                int(known[evidence_id]["locator"]["line_end"]),
            )
            for evidence_id in role_ids
        ]
        containing = [
            span
            for span in spans.get((candidate.path, candidate.symbol), [])
            if span.line_start <= min(start for start, _ in role_lines)
            and max(end for _, end in role_lines) <= span.line_end
        ]
        if not containing:
            raise BehaviorError(
                f"Behavior has no source scope: {candidate.path}::{candidate.symbol}"
            )
        scope = min(
            containing,
            key=lambda item: (item.line_end - item.line_start, item.line_start),
        )
        artifact = artifacts[candidate.path]
        behavior = {
            "artifact_kind": artifact.kind,
            "path": candidate.path,
            "symbol": candidate.symbol,
            "symbol_lines": [scope.line_start, scope.line_end],
            "result_type": candidate.result_kind,
            "trigger_evidence_ids": trigger_ids,
            "operation_evidence_ids": operation_ids,
            "result_evidence_ids": result_ids,
        }
        identity = (
            candidate.path,
            candidate.symbol,
            candidate.result_line,
            candidate.result_kind,
            tuple(trigger_ids),
            tuple(operation_ids),
            tuple(result_ids),
        )
        unique.setdefault(identity, behavior)
    ordered = [unique[key] for key in sorted(unique)]
    return [
        {"behavior_id": f"B{index:04d}", **behavior}
        for index, behavior in enumerate(ordered, start=1)
    ]


def _split_evidence_roles(
    candidate: BehaviorCandidate,
    known: dict[str, dict[str, Any]],
) -> tuple[list[str], list[str], list[str]]:
    ordered = list(candidate.evidence_ids)
    result_ids = [
        evidence_id
        for evidence_id in ordered
        if int(known[evidence_id]["locator"]["line_start"])
        <= candidate.result_line
        <= int(known[evidence_id]["locator"]["line_end"])
    ]
    result_set = set(result_ids)
    trigger_ids = [
        evidence_id
        for evidence_id in ordered
        if evidence_id not in result_set
        and _is_trigger_evidence(str(known[evidence_id]["content"]))
    ]
    trigger_set = set(trigger_ids)
    operation_ids = [
        evidence_id
        for evidence_id in ordered
        if evidence_id not in result_set and evidence_id not in trigger_set
    ]
    return trigger_ids, operation_ids, result_ids


def _is_trigger_evidence(content: str) -> bool:
    stripped = content.lstrip()
    return bool(
        re.match(
            r"^(?:if\b|elif\b|else\s*:|for\b|async\s+for\b|while\b|try\s*:|except\*?\b|finally\s*:|with\b|async\s+with\b|match\b|case\b)",
            stripped,
        )
    )


def _role_ids(behavior: dict[str, Any]) -> list[str]:
    return [
        *behavior["trigger_evidence_ids"],
        *behavior["operation_evidence_ids"],
        *behavior["result_evidence_ids"],
    ]


def _validate_role_arrays(behavior: dict[str, Any]) -> None:
    for field in (
        "trigger_evidence_ids",
        "operation_evidence_ids",
        "result_evidence_ids",
    ):
        values = behavior.get(field)
        if not isinstance(values, list) or any(
            not isinstance(value, str) or not value for value in values
        ):
            raise BehaviorError(f"Behavior {field} must be an Evidence ID array")
        if len(values) != len(set(values)):
            raise BehaviorError(f"Behavior {field} must not repeat Evidence IDs")


def _validate_repo_behavior(
    behavior: dict[str, Any], known: dict[str, dict[str, Any]]
) -> None:
    required = {
        "behavior_id", "artifact_kind", "path", "symbol", "symbol_lines",
        "result_type", "trigger_evidence_ids", "operation_evidence_ids",
        "result_evidence_ids",
    }
    if set(behavior) != required:
        raise BehaviorError("Repo Behavior has an unexpected field")
    _validate_role_arrays(behavior)
    if behavior["artifact_kind"] not in ARTIFACT_KINDS:
        raise BehaviorError("Repo Behavior has an unsupported artifact_kind")
    if not isinstance(behavior["path"], str) or not behavior["path"]:
        raise BehaviorError("Repo Behavior requires a path")
    if not isinstance(behavior["symbol"], str) or not behavior["symbol"]:
        raise BehaviorError("Repo Behavior requires a symbol")
    if behavior["result_type"] not in RESULT_KINDS:
        raise BehaviorError("Repo Behavior has an unsupported result_type")
    lines = behavior["symbol_lines"]
    if (
        not isinstance(lines, list)
        or len(lines) != 2
        or any(not isinstance(value, int) or isinstance(value, bool) for value in lines)
        or not 1 <= lines[0] <= lines[1]
    ):
        raise BehaviorError("Repo Behavior requires valid symbol_lines")
    if not behavior["result_evidence_ids"]:
        raise BehaviorError("Repo Behavior requires result Evidence")
    role_ids = _role_ids(behavior)
    if any(item not in known for item in role_ids):
        raise BehaviorError("Repo Behavior references unknown Evidence")
    for evidence_id in role_ids:
        item = known[evidence_id]
        locator = item["locator"]
        if (
            item["source_type"] != "code"
            or locator["path"] != behavior["path"]
            or locator["symbol"] != behavior["symbol"]
            or not lines[0] <= locator["line_start"] <= locator["line_end"] <= lines[1]
        ):
            raise BehaviorError("Repo Behavior crosses its path, symbol, or source range")


def _validate_trace_behavior(
    behavior: dict[str, Any], known: dict[str, dict[str, Any]]
) -> None:
    required = {
        "behavior_id", "task_id", "sequence_index", "start_turn", "end_turn",
        "demand_refs", "action_evidence_ids", "response_refs",
    }
    if set(behavior) != required:
        raise BehaviorError("Trace Behavior has an unexpected field")
    task_id = behavior["task_id"]
    if not isinstance(task_id, str) or re.fullmatch(r"K\d{4,}", task_id) is None:
        raise BehaviorError("Trace Behavior requires a stable task_id")
    sequence_index = behavior["sequence_index"]
    if (
        not isinstance(sequence_index, int)
        or isinstance(sequence_index, bool)
        or sequence_index < 1
    ):
        raise BehaviorError("Trace Behavior requires a positive sequence_index")
    if (
        not isinstance(behavior["start_turn"], int)
        or isinstance(behavior["start_turn"], bool)
        or not isinstance(behavior["end_turn"], int)
        or isinstance(behavior["end_turn"], bool)
        or not 0 <= behavior["start_turn"] <= behavior["end_turn"]
    ):
        raise BehaviorError("Trace Behavior requires a valid turn range")

    demand_refs = behavior["demand_refs"]
    if not isinstance(demand_refs, list) or not demand_refs:
        raise BehaviorError("Trace Behavior requires at least one demand_ref")
    demand_ids: list[str] = []
    for demand_ref in demand_refs:
        if not isinstance(demand_ref, dict) or set(demand_ref) not in (
            {"evidence_id"},
            {"evidence_id", "char_range"},
        ):
            raise BehaviorError("Trace demand_ref has an unexpected field")
        evidence_id = demand_ref.get("evidence_id")
        if not isinstance(evidence_id, str) or not evidence_id:
            raise BehaviorError("Trace demand_ref requires an Evidence ID")
        demand_ids.append(evidence_id)
        if "char_range" in demand_ref:
            char_range = demand_ref["char_range"]
            content = str(known.get(evidence_id, {}).get("content", ""))
            if (
                not isinstance(char_range, list)
                or len(char_range) != 2
                or any(
                    not isinstance(value, int) or isinstance(value, bool)
                    for value in char_range
                )
                or not 0 <= char_range[0] < char_range[1] <= len(content)
                or not content[char_range[0]:char_range[1]].strip()
            ):
                raise BehaviorError("Trace demand_ref requires a valid char_range")
    if len(demand_ids) != len(set(demand_ids)):
        raise BehaviorError("Trace Behavior must not repeat demand Evidence")

    action_ids = behavior["action_evidence_ids"]
    if not isinstance(action_ids, list) or any(
        not isinstance(value, str) or not value for value in action_ids
    ):
        raise BehaviorError(
            "Trace Behavior action_evidence_ids must be an Evidence ID array"
        )
    if len(action_ids) != len(set(action_ids)):
        raise BehaviorError(
            "Trace Behavior action_evidence_ids must not repeat Evidence IDs"
        )

    response_refs = behavior["response_refs"]
    if not isinstance(response_refs, list):
        raise BehaviorError("Trace Behavior response_refs must be an array")
    response_ids: list[str] = []
    response_identities: set[tuple[str, tuple[int, int] | None]] = set()
    for response_ref in response_refs:
        if not isinstance(response_ref, dict) or set(response_ref) not in (
            {"evidence_id"},
            {"evidence_id", "char_range"},
        ):
            raise BehaviorError("Trace response_ref has an unexpected field")
        evidence_id = response_ref.get("evidence_id")
        if not isinstance(evidence_id, str) or not evidence_id:
            raise BehaviorError("Trace response_ref requires an Evidence ID")
        response_ids.append(evidence_id)
        char_range_value = response_ref.get("char_range")
        identity_range: tuple[int, int] | None = None
        if char_range_value is not None:
            content = str(known.get(evidence_id, {}).get("content", ""))
            if (
                not isinstance(char_range_value, list)
                or len(char_range_value) != 2
                or any(
                    not isinstance(value, int) or isinstance(value, bool)
                    for value in char_range_value
                )
                or not 0 <= char_range_value[0] < char_range_value[1] <= len(content)
                or not content[char_range_value[0]:char_range_value[1]].strip()
            ):
                raise BehaviorError("Trace response_ref requires a valid char_range")
            identity_range = (char_range_value[0], char_range_value[1])
        identity = (evidence_id, identity_range)
        if identity in response_identities:
            raise BehaviorError("Trace Behavior must not repeat response refs")
        response_identities.add(identity)
    if not action_ids and not response_refs:
        raise BehaviorError("Trace Behavior requires an action or response")

    role_ids = [
        *demand_ids,
        *action_ids,
        *response_ids,
    ]
    if any(evidence_id not in known for evidence_id in role_ids):
        raise BehaviorError("Trace Behavior references unknown Evidence")
    items = [known[evidence_id] for evidence_id in role_ids]
    if any(item["source_type"] != "trace" for item in items):
        raise BehaviorError("Trace Behavior references non-trace Evidence")
    demands = [known[evidence_id] for evidence_id in demand_ids]
    if any(item["locator"]["event_type"] != "user_prompt" for item in demands):
        raise BehaviorError("Trace demand must reference user Evidence")
    demand_indices = {int(item["locator"]["event_index"]) for item in demands}
    if len(demand_indices) != 1:
        raise BehaviorError("Trace Behavior demands must come from one user event")
    trigger_index = next(iter(demand_indices))
    next_user_indices = [
        int(item["locator"]["event_index"])
        for item in known.values()
        if item["locator"]["event_type"] == "user_prompt"
        and int(item["locator"]["event_index"]) > trigger_index
    ]
    boundary = min(next_user_indices) if next_user_indices else None
    for evidence_id in action_ids:
        item = known[evidence_id]
        event_index = int(item["locator"]["event_index"])
        if event_index <= trigger_index or (
            boundary is not None and event_index >= boundary
        ):
            raise BehaviorError("Trace Behavior crosses an interaction boundary")
        if item["locator"]["event_type"] != "tool_exchange":
            raise BehaviorError("Trace action must reference Tool Evidence")
    for evidence_id in response_ids:
        item = known[evidence_id]
        event_index = int(item["locator"]["event_index"])
        if event_index <= trigger_index or (
            boundary is not None and event_index >= boundary
        ):
            raise BehaviorError("Trace Behavior crosses an interaction boundary")
        if item["locator"]["event_type"] != "assistant_response":
            raise BehaviorError("Trace response must reference Assistant Evidence")
    turns = [int(item["locator"]["turn"]) for item in items]
    if behavior["start_turn"] != min(turns) or behavior["end_turn"] != max(turns):
        raise BehaviorError("Trace Behavior turn range is inconsistent")


def _validate_trace_scope_order(behaviors: list[dict[str, Any]]) -> None:
    task_order: list[str] = []
    sequences: dict[str, list[int]] = defaultdict(list)
    previous_start_turn: int | None = None
    for behavior in behaviors:
        start_turn = int(behavior["start_turn"])
        if previous_start_turn is not None and start_turn < previous_start_turn:
            raise BehaviorError("Trace Behaviors must preserve original order")
        previous_start_turn = start_turn
        task_id = str(behavior["task_id"])
        if task_id not in task_order:
            task_order.append(task_id)
        sequences[task_id].append(int(behavior["sequence_index"]))
    expected_tasks = [f"K{index:04d}" for index in range(1, len(task_order) + 1)]
    if task_order != expected_tasks:
        raise BehaviorError("Trace task IDs must be contiguous by first appearance")
    for task_id, values in sequences.items():
        if values != list(range(1, len(values) + 1)):
            raise BehaviorError(
                f"Trace sequence_index must be contiguous within {task_id}"
            )


def _is_python(artifact: RepoArtifact) -> bool:
    suffix = PurePosixPath(artifact.path).suffix.casefold()
    first = artifact.content.splitlines()[0] if artifact.content.splitlines() else ""
    return suffix in {".py", ".pyw"} or (first.startswith("#!") and "python" in first.casefold())


# ---------------------------------------------------------------------------
# Python path and reaching-definition analysis
# ---------------------------------------------------------------------------

CONTROL_NODES = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.With,
    ast.AsyncWith,
    ast.Try,
    ast.TryStar,
    ast.Match,
)
OUTPUT_CALLS = {"print", "pprint", "warn", "warning", "error", "exception", "critical", "info", "debug", "log"}
EXTERNAL_ROOTS = {
    "aiohttp", "asyncio", "boto3", "click", "flask", "httpx", "os", "pathlib",
    "requests", "shutil", "socket", "sqlite3", "subprocess", "sys", "urllib",
}
EXTERNAL_METHODS = {
    "append_text", "commit", "delete", "dump", "emit", "execute", "flush",
    "mkdir", "open", "post", "publish", "put", "remove", "rename", "replace",
    "rmdir", "save", "send", "sendall", "unlink", "write", "write_bytes",
    "write_text", "writelines",
}
STATE_MUTATORS = {
    "add", "append", "clear", "discard", "extend", "insert", "pop",
    "remove", "setdefault", "sort", "update",
}


@dataclass(frozen=True, slots=True)
class PythonScope:
    path: str
    symbol: str
    node: ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
    body: tuple[ast.stmt, ...]


@dataclass(frozen=True, slots=True)
class _Anchor:
    node: ast.AST
    statement: ast.stmt
    kind: str
    line: int
    column: int


@dataclass(frozen=True, slots=True)
class _DefinitionVariant:
    statements: frozenset[ast.stmt]
    decisions: tuple[tuple[ast.AST, str, int | None], ...]


class EvidenceLineIndex:
    """Map exact source ranges back to module-one Evidence IDs."""

    def __init__(self, nodes: Iterable[dict[str, Any]]) -> None:
        self._order: dict[str, int] = {}
        self._by_scope: dict[tuple[str, str], list[tuple[int, int, str]]] = {}
        for order, item in enumerate(nodes):
            evidence_id = str(item["evidence_id"])
            locator = item["locator"]
            start = locator.get("line_start")
            end = locator.get("line_end")
            if item.get("source_type") != "code" or not isinstance(start, int) or not isinstance(end, int):
                continue
            key = (str(locator["path"]), str(locator["symbol"]))
            self._order[evidence_id] = order
            self._by_scope.setdefault(key, []).append((start, end, evidence_id))
        for values in self._by_scope.values():
            values.sort(key=lambda item: (item[0], item[1], self._order[item[2]]))

    def ids_for_ranges(
        self,
        path: str,
        symbol: str,
        ranges: Iterable[tuple[int, int]],
    ) -> tuple[str, ...]:
        selected: set[str] = set()
        normalized = [(min(a, b), max(a, b)) for a, b in ranges]
        for start, end, evidence_id in self._by_scope.get((path, symbol), []):
            if any(start <= wanted_end and end >= wanted_start for wanted_start, wanted_end in normalized):
                selected.add(evidence_id)
        return tuple(sorted(selected, key=self._order.__getitem__))

    def order_ids(self, ids: Iterable[str]) -> tuple[str, ...]:
        return tuple(sorted(set(ids), key=self._order.__getitem__))


def analyze_python_artifact(
    artifact: RepoArtifact,
    evidence_index: EvidenceLineIndex,
) -> list[BehaviorCandidate]:
    tree = parse_python(artifact.content, artifact.path)
    local_names = _module_local_names(tree)
    imported_names = _imported_names(tree)
    candidates: list[BehaviorCandidate] = []
    for scope in _python_scopes(artifact.path, tree):
        candidates.extend(
            _analyze_scope(scope, evidence_index, local_names, imported_names)
        )
    return candidates


def _analyze_scope(
    scope: PythonScope,
    evidence_index: EvidenceLineIndex,
    local_names: set[str],
    imported_names: set[str],
) -> list[BehaviorCandidate]:
    statements = list(_walk_statements(scope.body))
    if not statements:
        return []
    scope_imported_names = imported_names | {
        alias.asname or alias.name.split(".", 1)[0]
        for statement in statements
        if isinstance(statement, (ast.Import, ast.ImportFrom))
        for alias in statement.names
    }
    parents, parent_fields = _parent_maps(scope.body)
    global_names = {
        name
        for statement in statements
        if isinstance(statement, ast.Global)
        for name in statement.names
    }
    anchors: list[_Anchor] = []
    for statement in statements:
        if isinstance(statement, ast.Return):
            anchors.append(_anchor(statement, statement, "return"))
        elif isinstance(statement, ast.Raise):
            anchors.append(_anchor(statement, statement, "raise"))
        for value in _yield_nodes_in_statement(statement):
            anchors.append(_anchor(value, statement, "yield"))
        state_effect = _statement_effect(
            statement,
            is_module=isinstance(scope.node, ast.Module),
            global_names=global_names,
        )
        if state_effect is not None:
            anchors.append(_anchor(statement, statement, state_effect))
        for node in _call_nodes_in_statement(statement):
            call_effect = _call_effect(
                node,
                local_names,
                scope_imported_names,
                global_names,
                isinstance(scope.node, ast.Module),
            )
            if call_effect is not None:
                anchors.append(_anchor(node, statement, call_effect))

    anchors = _deduplicate_anchors(anchors)
    candidates: list[BehaviorCandidate] = []
    for anchor in anchors:
        controls = _control_ancestors(anchor.statement, parents)
        anchor_guard = _guard_signature(anchor.statement, parents, parent_fields)
        dependency_node = anchor.node if isinstance(anchor.node, ast.Call) else anchor.statement
        needed = set(_loaded_names(dependency_node))
        for control in controls:
            needed.update(_loaded_names(_control_expression(control)))
        definitions, _ = _backward_definitions(
            statements,
            anchor,
            needed,
            parents,
            parent_fields,
        )
        for variant in _definition_path_variants(
            definitions,
            anchor_guard,
            needed,
            parents,
            parent_fields,
        ):
            anchor_node = anchor.node if isinstance(anchor.node, ast.Call) else anchor.statement
            ranges: list[tuple[int, int]] = [_node_range(anchor_node)]
            for control in controls:
                ranges.extend(
                    _control_ranges(control, anchor.statement, parents, parent_fields)
                )
            for definition in variant.statements:
                ranges.append(_node_range(definition))
                for control in _control_ancestors(definition, parents):
                    ranges.extend(
                        _control_ranges(
                            control,
                            definition,
                            parents,
                            parent_fields,
                        )
                    )
            for control, field, index in variant.decisions:
                ranges.extend(
                    _control_decision_ranges(
                        control,
                        field,
                        index,
                        parents,
                        parent_fields,
                    )
                )
            evidence_ids = evidence_index.ids_for_ranges(scope.path, scope.symbol, ranges)
            if not evidence_ids:
                continue
            candidates.append(
                BehaviorCandidate(
                    path=scope.path,
                    symbol=scope.symbol,
                    result_kind=anchor.kind,
                    result_line=anchor.line,
                    evidence_ids=evidence_index.order_ids(evidence_ids),
                )
            )
    return candidates


def _definition_path_variants(
    definitions: set[ast.stmt],
    anchor_guard: tuple[tuple[int, str, int | None], ...],
    needed: set[str],
    parents: dict[ast.AST, ast.AST],
    parent_fields: dict[ast.AST, tuple[str, int | None]],
) -> list[_DefinitionVariant]:
    if not definitions:
        return [_DefinitionVariant(frozenset(), ())]
    guards = {
        statement: _guard_signature(statement, parents, parent_fields)
        for statement in definitions
    }
    fixed = {control: (field, index) for control, field, index in anchor_guard}
    options: dict[int, set[tuple[str, int | None]]] = defaultdict(set)
    for guard in guards.values():
        for control, field, index in guard:
            if control not in fixed:
                options[control].add((field, index))
    control_nodes = {
        id(node): node
        for node in parents.values()
        if isinstance(node, CONTROL_NODES)
    }
    for control, branches in options.items():
        node = control_nodes.get(control)
        if not isinstance(node, ast.If):
            continue
        branches.add(("body", None))
        branches.add(("orelse", None) if node.orelse else ("<inactive>", None))
    guard_depth = {
        control: min(
            index
            for guard in guards.values()
            for index, (guard_control, _, _) in enumerate(guard)
            if guard_control == control
        )
        for control in options
    }
    ordered_controls = sorted(
        options,
        key=lambda control: (
            guard_depth[control],
            _node_position(control_nodes[control]),
            control,
        ),
    )
    variants: dict[tuple[Any, ...], _DefinitionVariant] = {}
    for decisions in _reachable_decisions(
        ordered_controls,
        options,
        fixed,
        guards.values(),
    ):
        selected = {
            statement
            for statement, guard in guards.items()
            if all(
                control not in decisions or decisions[control] == (field, index)
                for control, field, index in guard
            )
        }
        selected = _live_definitions(selected, needed)
        relevant: list[tuple[ast.AST, str, int | None]] = []
        for control in ordered_controls:
            if control not in decisions:
                continue
            node = control_nodes.get(control)
            if node is None:
                continue
            field, index = decisions[control]
            relevant.append((node, field, index))
        identity = (
            tuple(sorted(id(statement) for statement in selected)),
            tuple((id(control), field, index) for control, field, index in relevant),
        )
        variants.setdefault(
            identity,
            _DefinitionVariant(frozenset(selected), tuple(relevant)),
        )
    return list(variants.values()) or [
        _DefinitionVariant(frozenset(_live_definitions(definitions, needed)), ())
    ]


def _reachable_decisions(
    controls: list[int],
    options: dict[int, set[tuple[str, int | None]]],
    fixed: dict[int, tuple[str, int | None]],
    guards: Iterable[tuple[tuple[int, str, int | None], ...]],
) -> Iterable[dict[int, tuple[str, int | None]]]:
    guard_list = tuple(guards)

    def visit(
        index: int,
        decisions: dict[int, tuple[str, int | None]],
    ) -> Iterable[dict[int, tuple[str, int | None]]]:
        if index == len(controls):
            yield decisions
            return
        control = controls[index]
        if not _decision_is_relevant(control, decisions, guard_list):
            yield from visit(index + 1, decisions)
            return
        for choice in sorted(options[control]):
            yield from visit(index + 1, {**decisions, control: choice})

    yield from visit(0, dict(fixed))


def _live_definitions(
    definitions: set[ast.stmt],
    needed: set[str],
) -> set[ast.stmt]:
    """Keep only definitions that reach a needed value on one chosen route."""

    live = set(needed)
    selected: set[ast.stmt] = set()
    for statement in sorted(definitions, key=_node_position, reverse=True):
        stored = _direct_stored_names(statement)
        if not stored & live:
            continue
        selected.add(statement)
        live.difference_update(stored)
        live.update(_definition_loaded_names(statement))
    return selected


def _decision_is_relevant(
    control: int,
    decisions: dict[int, tuple[str, int | None]],
    guards: Iterable[tuple[tuple[int, str, int | None], ...]],
) -> bool:
    for guard in guards:
        for index, (guard_control, _, _) in enumerate(guard):
            if guard_control != control:
                continue
            if all(
                decisions.get(parent_control) == (field, branch_index)
                for parent_control, field, branch_index in guard[:index]
            ):
                return True
    return False


def _python_scopes(path: str, tree: ast.Module) -> list[PythonScope]:
    scopes = [
        PythonScope(
            path=path,
            symbol="<module>",
            node=tree,
            body=tuple(_without_docstring(tree.body)),
        )
    ]

    def visit(body: list[ast.stmt], parent: str | None) -> None:
        for statement in body:
            if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                for nested_body in _statement_child_bodies(statement):
                    visit(nested_body, parent)
                continue
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_overload_definition(statement):
                continue
            symbol = f"{parent}.{statement.name}" if parent else statement.name
            scopes.append(
                PythonScope(
                    path=path,
                    symbol=symbol,
                    node=statement,
                    body=tuple(_without_docstring(statement.body)),
                )
            )
            visit(statement.body, symbol)

    visit(tree.body, None)
    return scopes


def _walk_statements(body: Iterable[ast.stmt]) -> Iterable[ast.stmt]:
    for statement in body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        yield statement
        for child_body in _statement_child_bodies(statement):
            yield from _walk_statements(child_body)


def _statement_child_bodies(statement: ast.stmt) -> list[list[ast.stmt]]:
    bodies: list[list[ast.stmt]] = []
    for _, value in ast.iter_fields(statement):
        if isinstance(value, list) and value and all(isinstance(item, ast.stmt) for item in value):
            bodies.append(value)
        elif isinstance(value, ast.ExceptHandler):
            bodies.append(value.body)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, ast.ExceptHandler):
                    bodies.append(item.body)
                elif isinstance(item, ast.match_case):
                    bodies.append(item.body)
    return bodies


def _yield_nodes_in_statement(statement: ast.stmt) -> Iterable[ast.Yield | ast.YieldFrom]:
    """Find yields owned by this statement without entering nested statements."""

    stack: list[ast.AST] = [
        child for child in ast.iter_child_nodes(statement) if not isinstance(child, ast.stmt)
    ]
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.Yield, ast.YieldFrom)):
            yield node
        stack.extend(
            child for child in ast.iter_child_nodes(node) if not isinstance(child, ast.stmt)
        )


def _call_nodes_in_statement(statement: ast.stmt) -> Iterable[ast.Call]:
    """Find calls owned by this statement without entering child statements."""

    stack: list[ast.AST] = [
        child for child in ast.iter_child_nodes(statement) if not isinstance(child, ast.stmt)
    ]
    while stack:
        node = stack.pop()
        if isinstance(node, ast.Call):
            yield node
        if isinstance(node, ast.Lambda):
            continue
        stack.extend(
            child for child in ast.iter_child_nodes(node) if not isinstance(child, ast.stmt)
        )


def _parent_maps(
    body: Iterable[ast.stmt],
) -> tuple[dict[ast.AST, ast.AST], dict[ast.AST, tuple[str, int | None]]]:
    parents: dict[ast.AST, ast.AST] = {}
    fields: dict[ast.AST, tuple[str, int | None]] = {}

    def visit(parent: ast.AST) -> None:
        for field_name, value in ast.iter_fields(parent):
            if isinstance(value, ast.AST):
                parents[value] = parent
                fields[value] = (field_name, None)
                if not isinstance(value, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    visit(value)
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    if not isinstance(child, ast.AST):
                        continue
                    parents[child] = parent
                    fields[child] = (field_name, index)
                    if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        visit(child)

    synthetic = ast.Module(body=list(body), type_ignores=[])
    visit(synthetic)
    return parents, fields


def _statement_effect(
    statement: ast.stmt,
    *,
    is_module: bool,
    global_names: set[str],
) -> str | None:
    if isinstance(statement, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Delete)):
        targets = _assignment_targets(statement)
        if any(
            isinstance(target, (ast.Attribute, ast.Subscript))
            or (isinstance(target, ast.Name) and (is_module or target.id in global_names))
            for target in targets
        ):
            return "state_write"
    return None


def _call_effect(
    call: ast.Call,
    local_names: set[str],
    imported_names: set[str],
    global_names: set[str],
    is_module: bool,
) -> str | None:
    label = _call_label(call.func)
    if not label:
        return None
    parts = label.split(".")
    root = parts[0]
    tail = parts[-1].casefold()
    if tail in STATE_MUTATORS and (
        root in {"self", "cls"} or root in global_names or is_module
    ):
        return "state_write"
    if tail in OUTPUT_CALLS or root == "logging" or root in {"logger", "log"}:
        return "output"
    if root in {"self", "cls"}:
        return "external_call" if tail in EXTERNAL_METHODS else None
    if root in imported_names or root in EXTERNAL_ROOTS or tail in EXTERNAL_METHODS:
        return "external_call"
    if root in local_names:
        return None
    return None


def _backward_definitions(
    statements: list[ast.stmt],
    anchor: _Anchor,
    needed: set[str],
    parents: dict[ast.AST, ast.AST],
    parent_fields: dict[ast.AST, tuple[str, int | None]],
) -> tuple[set[ast.stmt], list[tuple[int, int]]]:
    selected: set[ast.stmt] = set()
    control_ranges: list[tuple[int, int]] = []
    pending: list[tuple[str, ast.stmt]] = [
        (name, anchor.statement) for name in sorted(needed)
    ]
    visited: set[tuple[str, int]] = set()
    while pending:
        name, use_statement = pending.pop()
        visit_key = (name, id(use_statement))
        if visit_key in visited:
            continue
        visited.add(visit_key)
        use_guard = _guard_signature(use_statement, parents, parent_fields)
        use_position = _node_position(use_statement)
        for statement in reversed(statements):
            if _node_position(statement) >= use_position:
                continue
            if name not in _direct_stored_names(statement):
                continue
            definition_guard = _guard_signature(statement, parents, parent_fields)
            if not _guards_compatible(definition_guard, use_guard):
                continue
            selected.add(statement)
            for control in _control_ancestors(statement, parents):
                control_ranges.extend(
                    _control_ranges(control, statement, parents, parent_fields)
                )
                for dependency in sorted(
                    _loaded_names(_control_expression(control))
                ):
                    pending.append((dependency, control))
            stored = _direct_stored_names(statement)
            for dependency in sorted(_definition_loaded_names(statement) - stored):
                pending.append((dependency, statement))
            # A definition on the same unconditional path dominates older
            # ones. Branch-local definitions do not dominate a use after the
            # join, so all compatible branches remain visible.
            if _guard_is_subset(definition_guard, use_guard):
                break
    return selected, sorted(set(control_ranges))


def _guards_compatible(
    left: tuple[tuple[int, str, int | None], ...],
    right: tuple[tuple[int, str, int | None], ...],
) -> bool:
    left_branches = {control: (field, index) for control, field, index in left}
    right_branches = {control: (field, index) for control, field, index in right}
    return all(
        left_branches[control] == right_branches[control]
        for control in left_branches.keys() & right_branches.keys()
    )


def _node_position(node: ast.AST) -> tuple[int, int]:
    return (
        int(getattr(node, "lineno", 1)),
        int(getattr(node, "col_offset", 0)),
    )


def _direct_stored_names(statement: ast.stmt) -> set[str]:
    targets: list[ast.AST] = []
    if isinstance(statement, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Delete)):
        targets.extend(_assignment_targets(statement))
    elif isinstance(statement, (ast.For, ast.AsyncFor)):
        targets.append(statement.target)
    elif isinstance(statement, (ast.With, ast.AsyncWith)):
        targets.extend(
            item.optional_vars for item in statement.items if item.optional_vars is not None
        )
    elif isinstance(statement, (ast.Import, ast.ImportFrom)):
        return {
            alias.asname or alias.name.split(".", 1)[0]
            for alias in statement.names
        }
    return {
        node.id
        for target in targets
        for node in ast.walk(target)
        if isinstance(node, ast.Name)
    }


def _definition_loaded_names(statement: ast.stmt) -> set[str]:
    values: list[ast.AST] = []
    if isinstance(statement, ast.Assign):
        values.append(statement.value)
    elif isinstance(statement, ast.AnnAssign):
        if statement.value is not None:
            values.append(statement.value)
    elif isinstance(statement, ast.AugAssign):
        values.extend((statement.target, statement.value))
    elif isinstance(statement, (ast.For, ast.AsyncFor)):
        values.append(statement.iter)
    elif isinstance(statement, (ast.With, ast.AsyncWith)):
        values.extend(item.context_expr for item in statement.items)
    return {name for value in values for name in _loaded_names(value)}


def _control_ancestors(
    node: ast.AST, parents: dict[ast.AST, ast.AST]
) -> list[ast.AST]:
    controls: list[ast.AST] = []
    current = node
    while current in parents:
        current = parents[current]
        if isinstance(current, CONTROL_NODES):
            controls.append(current)
    controls.reverse()
    return controls


def _guard_signature(
    node: ast.AST,
    parents: dict[ast.AST, ast.AST],
    parent_fields: dict[ast.AST, tuple[str, int | None]],
) -> tuple[tuple[int, str, int | None], ...]:
    result: list[tuple[int, str, int | None]] = []
    current = node
    while current in parents:
        parent = parents[current]
        if isinstance(parent, CONTROL_NODES):
            field, index = parent_fields.get(current, ("", None))
            branch_index = index if field in {"handlers", "cases"} else None
            result.append((id(parent), field, branch_index))
        current = parent
    result.reverse()
    return tuple(result)


def _guard_is_subset(
    possible_prefix: tuple[tuple[int, str, int | None], ...],
    full_guard: tuple[tuple[int, str, int | None], ...],
) -> bool:
    return len(possible_prefix) <= len(full_guard) and possible_prefix == full_guard[: len(possible_prefix)]


def _control_ranges(
    control: ast.AST,
    anchor: ast.AST,
    parents: dict[ast.AST, ast.AST],
    parent_fields: dict[ast.AST, tuple[str, int | None]],
) -> list[tuple[int, int]]:
    body = getattr(control, "body", [])
    start = int(getattr(control, "lineno", 1))
    first_body = int(body[0].lineno) if body else start
    ranges = [(start, max(start, first_body - 1))]
    child = anchor
    while child in parents and parents[child] is not control:
        child = parents[child]
    if child in parents and parents[child] is control:
        field, index = parent_fields.get(child, ("", None))
        if field == "orelse":
            orelse = getattr(control, "orelse", [])
            if body and orelse:
                gap_start = int(getattr(body[-1], "end_lineno", body[-1].lineno)) + 1
                gap_end = int(orelse[0].lineno) - 1
                if gap_start <= gap_end:
                    ranges.append((gap_start, gap_end))
        elif field == "handlers" and isinstance(child, ast.ExceptHandler):
            if child.body:
                ranges.append(
                    (int(child.lineno), max(int(child.lineno), int(child.body[0].lineno) - 1))
                )
        elif field == "finalbody":
            finalbody = getattr(control, "finalbody", [])
            predecessor = getattr(control, "orelse", []) or getattr(control, "handlers", []) or body
            if finalbody and predecessor:
                prior = predecessor[-1]
                if isinstance(prior, ast.ExceptHandler):
                    prior_end = getattr(prior.body[-1], "end_lineno", prior.body[-1].lineno) if prior.body else prior.lineno
                else:
                    prior_end = getattr(prior, "end_lineno", prior.lineno)
                gap_start = int(prior_end) + 1
                gap_end = int(finalbody[0].lineno) - 1
                if gap_start <= gap_end:
                    ranges.append((gap_start, gap_end))
        elif field == "cases" and isinstance(child, ast.match_case) and child.body:
            pattern_line = int(getattr(child.pattern, "lineno", child.body[0].lineno))
            ranges.append((pattern_line, max(pattern_line, int(child.body[0].lineno) - 1)))
    return ranges


def _control_decision_ranges(
    control: ast.AST,
    field: str,
    index: int | None,
    parents: dict[ast.AST, ast.AST],
    parent_fields: dict[ast.AST, tuple[str, int | None]],
) -> list[tuple[int, int]]:
    """Return only the source headers needed to identify a chosen branch."""

    body = getattr(control, "body", [])
    start = int(getattr(control, "lineno", 1))
    first_body = int(body[0].lineno) if body else start
    header = [(start, max(start, first_body - 1))]
    if field == "<inactive>":
        return header
    branch = getattr(control, field, None)
    if not isinstance(branch, list) or not branch:
        return header
    if field in {"handlers", "cases"}:
        if index is None or not 0 <= index < len(branch):
            return header
        anchor = branch[index]
    else:
        anchor = branch[0]
    if not isinstance(anchor, ast.AST):
        return header
    return _control_ranges(control, anchor, parents, parent_fields)


def _control_expression(node: ast.AST) -> ast.AST:
    if isinstance(node, ast.If | ast.While):
        return node.test
    if isinstance(node, (ast.For, ast.AsyncFor)):
        return node.iter
    if isinstance(node, (ast.With, ast.AsyncWith)):
        return ast.Tuple(elts=[item.context_expr for item in node.items], ctx=ast.Load())
    if isinstance(node, ast.Match):
        return node.subject
    return ast.Tuple(elts=[], ctx=ast.Load())


def _assignment_targets(statement: ast.stmt) -> list[ast.AST]:
    if isinstance(statement, ast.Assign):
        return list(statement.targets)
    if isinstance(statement, ast.AnnAssign):
        return [statement.target]
    if isinstance(statement, ast.AugAssign):
        return [statement.target]
    if isinstance(statement, ast.Delete):
        return list(statement.targets)
    return []


def _loaded_names(node: ast.AST) -> set[str]:
    return {
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
    }


def _module_local_names(tree: ast.Module) -> set[str]:
    return {
        statement.name
        for statement in tree.body
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def _imported_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for statement in tree.body:
        if isinstance(statement, ast.Import):
            names.update(alias.asname or alias.name.split(".")[0] for alias in statement.names)
        elif isinstance(statement, ast.ImportFrom):
            names.update(alias.asname or alias.name for alias in statement.names)
    return names


def _call_label(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_label(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _anchor(node: ast.AST, statement: ast.stmt, kind: str) -> _Anchor:
    return _Anchor(
        node=node,
        statement=statement,
        kind=kind,
        line=int(getattr(node, "lineno", statement.lineno)),
        column=int(getattr(node, "col_offset", statement.col_offset)),
    )


def _deduplicate_anchors(anchors: Iterable[_Anchor]) -> list[_Anchor]:
    result: dict[tuple[str, int, int], _Anchor] = {}
    for anchor in anchors:
        result.setdefault((anchor.kind, anchor.line, anchor.column), anchor)
    return sorted(result.values(), key=lambda item: (item.line, item.column, item.kind))


def _node_range(node: ast.AST) -> tuple[int, int]:
    start = int(getattr(node, "lineno", 1))
    return start, int(getattr(node, "end_lineno", start))


def _without_docstring(body: list[ast.stmt]) -> list[ast.stmt]:
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        return body[1:]
    return body


def _is_overload_definition(
    statement: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    return any(
        _call_label(decorator.func if isinstance(decorator, ast.Call) else decorator)
        .rsplit(".", 1)[-1]
        == "overload"
        for decorator in statement.decorator_list
    )
