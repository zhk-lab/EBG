"""Lossless trace rendering and one-shot TraceReview evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from agentloop.backend import content_identity
from agentloop.context import FastTokenCounter
from agentloop.errors import AgentLoopError, ProviderContextError, RetryableModelError
from agentloop.provider import ModelClient, ModelCompletion, validated_model_profile
from agentloop.storage import RunStore
from evaluation_core.contracts import validate_trace_prediction

from .config import DEFAULT_CONFIG, TraceReviewConfig


_EVENT_TITLES = {
    "system": "SYSTEM CONTEXT",
    "user_prompt": "USER PROMPT",
    "assistant_response": "ASSISTANT RESPONSE",
    "tool_exchange": "TOOL EXCHANGE",
}
TRACE_STATE_VERSION = 9


@dataclass(frozen=True, slots=True)
class TraceView:
    input_id: str
    text: str
    evidence_ids: frozenset[str]


def prepare_trace_request(
    messages: list[dict[str, str]],
    *,
    config: TraceReviewConfig = DEFAULT_CONFIG,
) -> tuple[int, dict[str, Any]]:
    """Validate a lossless one-shot request before a model client is created."""

    if not messages or any(
        not isinstance(message, dict)
        or not isinstance(message.get("role"), str)
        or not isinstance(message.get("content"), str)
        for message in messages
    ):
        raise AgentLoopError("TraceReview messages must be non-empty chat messages")
    token_count = FastTokenCounter(config.token_estimator).count_messages(messages)
    if token_count > config.physical_hard_limit:
        raise AgentLoopError("lossless TraceReview input exceeds physical context")
    return token_count, {
        "mode": "lossless_one_shot",
        "tokens_before": token_count,
        "tokens_after": token_count,
    }


def render_raw_trace_payload(payload: dict[str, Any]) -> TraceView:
    """Validate and canonically render the complete Raw FeedbackTrace input."""

    required = {"input_id", "track", "selectable_evidence_ids", "events"}
    allowed = required | {"early_history_summary"}
    input_id = payload.get("input_id")
    events = payload.get("events")
    selectable = payload.get("selectable_evidence_ids")
    if (
        set(payload) - allowed
        or not required <= set(payload)
        or not isinstance(input_id, str)
        or not input_id
        or payload.get("track") != "long"
        or not isinstance(events, list)
        or not events
        or not isinstance(selectable, list)
        or any(not isinstance(item, str) or not item for item in selectable)
        or len(selectable) != len(set(selectable))
    ):
        raise AgentLoopError("Raw FeedbackTrace payload is invalid")
    actual_ids: list[str] = []
    previous_turn = -1
    for event in events:
        if not isinstance(event, dict):
            raise AgentLoopError("Raw FeedbackTrace event is invalid")
        event_type = event.get("event_type")
        turn = event.get("turn_number")
        content = event.get("content")
        if (
            event_type not in _EVENT_TITLES
            or not isinstance(turn, int)
            or isinstance(turn, bool)
            or turn < previous_turn
            or not isinstance(content, str)
        ):
            raise AgentLoopError("Raw FeedbackTrace event is invalid")
        previous_turn = turn
        evidence_id = event.get("evidence_id")
        if evidence_id is not None:
            if not isinstance(evidence_id, str) or not evidence_id:
                raise AgentLoopError("Raw FeedbackTrace Evidence ID is invalid")
            actual_ids.append(evidence_id)
    if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != set(selectable):
        raise AgentLoopError("Raw FeedbackTrace selectable Evidence IDs changed")
    summary = payload.get("early_history_summary")
    if summary is not None and (not isinstance(summary, str) or not summary.strip()):
        raise AgentLoopError("Raw FeedbackTrace history summary is invalid")
    text = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return TraceView(input_id, text, frozenset(actual_ids))


def render_trace_view(graph: dict[str, Any]) -> TraceView:
    """Render the complete backend graph as a lossless Task Scope view."""

    if graph.get("benchmark") != "feedbacktrace":
        raise AgentLoopError("trace view requires a FeedbackTrace graph")
    input_id = str(graph.get("input_id") or "")
    evidence = graph.get("evidence")
    behaviors = graph.get("behaviors")
    edges = graph.get("edges")
    if (
        set(graph) != {"input_id", "benchmark", "evidence", "behaviors", "edges"}
        or not input_id
        or not isinstance(evidence, list)
        or not isinstance(behaviors, list)
        or not behaviors
        or not isinstance(edges, list)
    ):
        raise AgentLoopError("FeedbackTrace graph is invalid")

    evidence_by_id, original_ids = _validate_trace_evidence(evidence)
    task_order = _validate_trace_behaviors(behaviors, evidence_by_id)
    outgoing = _validate_trace_edges(edges, evidence_by_id, task_order, behaviors)
    current_task_id = str(behaviors[-1]["task_id"])
    task_scopes = [
        _task_scope(
            task_id,
            [item for item in behaviors if item["task_id"] == task_id],
            outgoing.get(task_id, []),
            evidence_by_id,
            current=task_id == current_task_id,
        )
        for task_id in task_order
    ]
    rendered_original_ids = {
        event["evidence_id"]
        for scope in task_scopes
        for behavior in scope["behaviors"]
        for role in ("demand", "action", "response")
        for event in behavior[role]
        if event["evidence_id"] is not None
    }
    if rendered_original_ids != original_ids:
        missing = original_ids - rendered_original_ids
        raise AgentLoopError(
            f"FeedbackTrace selectable Evidence was omitted: {sorted(missing)}"
        )

    scope_graph = {
        "input_id": input_id,
        "benchmark": "feedbacktrace",
        "current_task_id": current_task_id,
        "task_scopes": task_scopes,
    }
    text = json.dumps(
        scope_graph,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    return TraceView(input_id, text + "\n", frozenset(original_ids))


def _validate_trace_evidence(
    evidence: list[Any],
) -> tuple[dict[str, dict[str, Any]], set[str]]:
    evidence_by_id: dict[str, dict[str, Any]] = {}
    original_ids: set[str] = set()
    previous_key = (-1, -1)
    for item in evidence:
        if (
            not isinstance(item, dict)
            or set(item) != {"evidence_id", "source_type", "locator", "content"}
            or item.get("source_type") != "trace"
            or not isinstance(item.get("content"), str)
            or not isinstance(item.get("locator"), dict)
        ):
            raise AgentLoopError("FeedbackTrace Evidence is invalid")
        evidence_id = str(item.get("evidence_id") or "")
        locator = item["locator"]
        if (
            not evidence_id
            or evidence_id in evidence_by_id
            or set(locator)
            != {"turn", "event_index", "event_type", "tool_name", "original_evidence_id"}
            or locator.get("event_type") not in _EVENT_TITLES
        ):
            raise AgentLoopError("FeedbackTrace Evidence locator is invalid")
        key = (locator.get("turn"), locator.get("event_index"))
        if (
            not isinstance(key[0], int)
            or isinstance(key[0], bool)
            or key[0] < 0
            or not isinstance(key[1], int)
            or isinstance(key[1], bool)
            or key[1] < 0
            or key < previous_key
        ):
            raise AgentLoopError("FeedbackTrace Evidence changed original order")
        previous_key = key
        original_id = locator.get("original_evidence_id")
        if original_id is not None:
            if (
                not isinstance(original_id, str)
                or not original_id
                or original_id in original_ids
            ):
                raise AgentLoopError("FeedbackTrace original Evidence ID is invalid")
            original_ids.add(original_id)
        evidence_by_id[evidence_id] = item
    return evidence_by_id, original_ids


def _validate_trace_behaviors(
    behaviors: list[Any],
    evidence_by_id: dict[str, dict[str, Any]],
) -> list[str]:
    behavior_ids: set[str] = set()
    task_order: list[str] = []
    sequences: dict[str, list[int]] = {}
    chronological = sorted(behaviors, key=_behavior_sort_key)
    if behaviors != chronological:
        raise AgentLoopError("FeedbackTrace Behaviors changed chronological order")
    for behavior in behaviors:
        if (
            not isinstance(behavior, dict)
            or set(behavior)
            != {
                "behavior_id", "task_id", "sequence_index", "start_turn",
                "end_turn", "demand_refs", "action_evidence_ids",
                "response_refs",
            }
        ):
            raise AgentLoopError("FeedbackTrace Behavior is invalid")
        behavior_id = str(behavior.get("behavior_id") or "")
        task_id = str(behavior.get("task_id") or "")
        sequence_index = behavior.get("sequence_index")
        demand_refs = behavior["demand_refs"]
        action_ids = behavior["action_evidence_ids"]
        response_refs = behavior["response_refs"]
        if any(
            not isinstance(values, list)
            for values in (demand_refs, action_ids, response_refs)
        ):
            raise AgentLoopError("FeedbackTrace Behavior roles are invalid")
        demand_ids: list[str] = []
        for demand_ref in demand_refs:
            if (
                not isinstance(demand_ref, dict)
                or set(demand_ref) not in (
                    {"evidence_id"},
                    {"evidence_id", "char_range"},
                )
                or not isinstance(demand_ref.get("evidence_id"), str)
            ):
                raise AgentLoopError("FeedbackTrace demand reference is invalid")
            evidence_id = demand_ref["evidence_id"]
            demand_ids.append(evidence_id)
            if "char_range" in demand_ref:
                char_range = demand_ref["char_range"]
                content = str(evidence_by_id.get(evidence_id, {}).get("content", ""))
                if (
                    not isinstance(char_range, list)
                    or len(char_range) != 2
                    or any(
                        not isinstance(value, int) or isinstance(value, bool)
                        for value in char_range
                    )
                    or not 0 <= char_range[0] < char_range[1] <= len(content)
                ):
                    raise AgentLoopError("FeedbackTrace demand char_range is invalid")
        response_ids: list[str] = []
        response_identities: set[tuple[str, tuple[int, int] | None]] = set()
        for response_ref in response_refs:
            if (
                not isinstance(response_ref, dict)
                or set(response_ref) not in (
                    {"evidence_id"},
                    {"evidence_id", "char_range"},
                )
                or not isinstance(response_ref.get("evidence_id"), str)
            ):
                raise AgentLoopError("FeedbackTrace response reference is invalid")
            evidence_id = response_ref["evidence_id"]
            response_ids.append(evidence_id)
            identity_range: tuple[int, int] | None = None
            if "char_range" in response_ref:
                char_range = response_ref["char_range"]
                content = str(evidence_by_id.get(evidence_id, {}).get("content", ""))
                if (
                    not isinstance(char_range, list)
                    or len(char_range) != 2
                    or any(
                        not isinstance(value, int) or isinstance(value, bool)
                        for value in char_range
                    )
                    or not 0 <= char_range[0] < char_range[1] <= len(content)
                ):
                    raise AgentLoopError("FeedbackTrace response char_range is invalid")
                identity_range = (char_range[0], char_range[1])
            identity = (evidence_id, identity_range)
            if identity in response_identities:
                raise AgentLoopError("FeedbackTrace response reference is repeated")
            response_identities.add(identity)
        role_ids = [*demand_ids, *action_ids, *response_ids]
        if (
            not behavior_id
            or behavior_id in behavior_ids
            or not task_id
            or not isinstance(sequence_index, int)
            or isinstance(sequence_index, bool)
            or sequence_index < 1
            or not demand_ids
            or (not action_ids and not response_ids)
            or any(
                len(values) != len(set(values))
                for values in (demand_ids, action_ids)
            )
            or any(item not in evidence_by_id for item in role_ids)
        ):
            raise AgentLoopError("FeedbackTrace Behavior references invalid Evidence")
        if any(
            evidence_by_id[evidence_id]["locator"]["event_type"] != "user_prompt"
            for evidence_id in demand_ids
        ) or any(
            evidence_by_id[evidence_id]["locator"]["event_type"] != "tool_exchange"
            for evidence_id in action_ids
        ) or any(
            evidence_by_id[evidence_id]["locator"]["event_type"] != "assistant_response"
            for evidence_id in response_ids
        ):
            raise AgentLoopError("FeedbackTrace Behavior role has the wrong event type")
        behavior_ids.add(behavior_id)
        if task_id not in task_order:
            task_order.append(task_id)
            sequences[task_id] = []
        sequences[task_id].append(sequence_index)
    for task_id, values in sequences.items():
        if values != list(range(1, len(values) + 1)):
            raise AgentLoopError(
                f"FeedbackTrace sequence_index changed within {task_id}"
            )
    return task_order


def _validate_trace_edges(
    edges: list[Any],
    evidence_by_id: dict[str, dict[str, Any]],
    task_order: list[str],
    behaviors: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    task_ids = set(task_order)
    task_evidence: dict[str, set[str]] = {task_id: set() for task_id in task_order}
    for behavior in behaviors:
        task_evidence[behavior["task_id"]].update(
            demand_ref["evidence_id"] for demand_ref in behavior["demand_refs"]
        )
        task_evidence[behavior["task_id"]].update(behavior["action_evidence_ids"])
        task_evidence[behavior["task_id"]].update(
            item["evidence_id"] for item in behavior["response_refs"]
        )
    outgoing: dict[str, list[dict[str, Any]]] = {task_id: [] for task_id in task_order}
    edge_ids: set[str] = set()
    identities: set[tuple[str, str, str]] = set()
    for edge in edges:
        if (
            not isinstance(edge, dict)
            or set(edge)
            != {
                "edge_id", "source_task_id", "type", "target_task_id",
                "evidence_ids",
            }
            or edge.get("type") not in {"informs", "supersedes"}
            or not isinstance(edge.get("evidence_ids"), list)
            or any(item not in evidence_by_id for item in edge["evidence_ids"])
        ):
            raise AgentLoopError("FeedbackTrace edge is invalid")
        edge_id = str(edge.get("edge_id") or "")
        source = str(edge.get("source_task_id") or "")
        target = str(edge.get("target_task_id") or "")
        identity = (source, str(edge["type"]), target)
        if (
            not edge_id
            or edge_id in edge_ids
            or source not in task_ids
            or target not in task_ids
            or source == target
            or identity in identities
        ):
            raise AgentLoopError("FeedbackTrace edge references an invalid Task Scope")
        if not set(edge["evidence_ids"]) <= (
            task_evidence[source] | task_evidence[target]
        ):
            raise AgentLoopError("FeedbackTrace edge support is outside its Task Scopes")
        edge_ids.add(edge_id)
        identities.add(identity)
        outgoing[source].append(edge)
    for task_id in outgoing:
        outgoing[task_id].sort(
            key=lambda item: (
                str(item["type"]), str(item["target_task_id"]), str(item["edge_id"])
            )
        )
    return outgoing


def _behavior_sort_key(item: Any) -> tuple[int, str, int]:
    if not isinstance(item, dict):
        raise AgentLoopError("FeedbackTrace Behavior is invalid")
    start = item.get("start_turn")
    end = item.get("end_turn")
    if (
        not isinstance(start, int)
        or isinstance(start, bool)
        or not isinstance(end, int)
        or isinstance(end, bool)
        or not 0 <= start <= end
    ):
        raise AgentLoopError("FeedbackTrace Behavior range is invalid")
    # Behaviors split from one user turn share start_turn.  Their end turns
    # may interleave because each demand has different Tool/response events;
    # the generated Behavior ID preserves their demand order.
    return start, str(item.get("behavior_id") or ""), end


def _task_scope(
    task_id: str,
    behaviors: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    evidence_by_id: dict[str, dict[str, Any]],
    *,
    current: bool,
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "status": "current" if current else "history",
        "turn_range": [
            min(int(item["start_turn"]) for item in behaviors),
            max(int(item["end_turn"]) for item in behaviors),
        ],
        "behaviors": [_scope_behavior(item, evidence_by_id) for item in behaviors],
        "relations": [_scope_edge(item, evidence_by_id) for item in edges],
    }


def _display_event(
    internal_id: str,
    evidence_by_id: dict[str, dict[str, Any]],
    *,
    char_range: list[int] | None = None,
) -> dict[str, Any]:
    item = evidence_by_id[internal_id]
    locator = item["locator"]
    content = str(item["content"])
    if char_range is not None:
        content = content[char_range[0]:char_range[1]]
    result: dict[str, Any] = {
        "evidence_id": locator["original_evidence_id"],
        "content": content,
    }
    if locator["tool_name"] is not None:
        result["tool_name"] = str(locator["tool_name"])
    return result


def _scope_behavior(
    behavior: dict[str, Any],
    evidence_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "behavior_id": str(behavior["behavior_id"]),
        "sequence_index": int(behavior["sequence_index"]),
        "turn_range": [int(behavior["start_turn"]), int(behavior["end_turn"])],
        "demand": [
            _display_event(
                demand_ref["evidence_id"],
                evidence_by_id,
                char_range=demand_ref.get("char_range"),
            )
            for demand_ref in behavior["demand_refs"]
        ],
        "action": [
            _display_event(evidence_id, evidence_by_id)
            for evidence_id in behavior["action_evidence_ids"]
        ],
        "response": [
            _display_event(
                response_ref["evidence_id"],
                evidence_by_id,
                char_range=response_ref.get("char_range"),
            )
            for response_ref in behavior["response_refs"]
        ],
    }


def _scope_edge(
    edge: dict[str, Any],
    evidence_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "edge_id": str(edge["edge_id"]),
        "type": str(edge["type"]),
        "target_task_id": str(edge["target_task_id"]),
        "support": [
            _display_event(evidence_id, evidence_by_id)
            for evidence_id in edge["evidence_ids"]
        ],
    }


class TraceReview:
    """Run one lossless trace review without repository browsing or receipts."""

    def __init__(
        self,
        *,
        view: TraceView,
        messages: list[dict[str, str]],
        prediction_schema: dict[str, Any],
        client: ModelClient,
        store: RunStore,
        config: TraceReviewConfig = DEFAULT_CONFIG,
    ) -> None:
        if not messages or any(
            not isinstance(message, dict)
            or not isinstance(message.get("role"), str)
            or not isinstance(message.get("content"), str)
            for message in messages
        ):
            raise AgentLoopError("TraceReview messages must be non-empty chat messages")
        self.view = view
        self.messages = [dict(message) for message in messages]
        self.schema = prediction_schema
        self.client = client
        self.model_profile = validated_model_profile(client)
        self.store = store
        self.config = config
        self.counter = FastTokenCounter(config.token_estimator)

    def run(self) -> dict[str, Any]:
        messages = [dict(message) for message in self.messages]
        token_count = self.counter.count_messages(messages)
        expected = {
            "schema_version": TRACE_STATE_VERSION,
            "input_id": self.view.input_id,
            "benchmark": "feedbacktrace",
            "config": self.config.public_dict(),
            "prediction_schema": self.schema,
            "model_profile": self.model_profile,
            "view_identity": content_identity([self.view.text]),
            "message_identity": content_identity(
                [
                    json.dumps(
                        messages,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                ]
            ),
        }
        state = self.store.load_state()
        if state is None:
            state = {
                **expected,
                "status": "running",
                "turns": 0,
                "input_tokens": token_count,
                "terminal_record": None,
                "prediction": None,
                "failure": None,
            }
            self.store.save_state(state)
        else:
            for key, value in expected.items():
                if state.get(key) != value:
                    raise AgentLoopError(
                        f"resume state does not match current {key}"
                    )
        self.store.save_trace_view(self.view.text)
        if state.get("status") == "complete":
            prediction = state.get("prediction")
            if not isinstance(prediction, dict):
                raise AgentLoopError("completed TraceReview state lacks prediction")
            return prediction
        if state.get("status") == "failed":
            raise AgentLoopError(
                f"TraceReview run already failed: {state.get('failure')}"
            )
        if token_count > self.config.physical_hard_limit:
            self._save_failure(state, "context_unfit", token_count)
            raise AgentLoopError("lossless TraceReview input exceeds physical context")
        try:
            persisted = self.store.load_response(1)
            if persisted is None:
                completion, provider_retry = self._request(messages, token_count)
            else:
                provider_retry = int(persisted.get("provider_retry", 0))
                request = self.store.load_request(1, provider_retry)
                if request is None or (
                    request.get("messages") != messages
                    or request.get("token_count") != token_count
                ):
                    raise AgentLoopError(
                        "persisted TraceReview response has no matching request"
                    )
                completion = ModelCompletion(
                    content=str(persisted["content"]),
                    raw_response=persisted.get("raw_response", {}),
                    usage=persisted.get("usage", {}),
                )
        except RetryableModelError as error:
            self._save_failure(state, "network_retries_exhausted", token_count)
            raise AgentLoopError("TraceReview network retries exhausted") from error
        except ProviderContextError as error:
            self._save_failure(
                state,
                "context_unfit/provider_limit_mismatch",
                token_count,
            )
            raise AgentLoopError(
                "lossless TraceReview request exceeded provider context"
            ) from error
        except AgentLoopError:
            self._save_failure(state, "provider_response_invalid", token_count)
            raise
        try:
            body = json.loads(
                completion.content.strip(),
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
            if not isinstance(body, dict):
                raise AgentLoopError("TraceReview response must be an object")
            if "verdict" in body:
                raise ValueError("TraceReview verdict is runner-owned")
            prediction = validate_trace_prediction(
                {"verdict": "KEY", **body},
                input_id=self.view.input_id,
                schema=self.schema,
                selectable_evidence_ids=set(self.view.evidence_ids),
            )
        except (json.JSONDecodeError, ValueError) as error:
            self._save_failure(
                state,
                "invalid_prediction",
                token_count,
                completion=completion,
                provider_retry=provider_retry,
            )
            raise AgentLoopError(
                "TraceReview response is not valid for the formal contract"
            ) from error
        self.store.save_prediction_response(body)
        self.store.save_prediction(prediction)
        self.store.save_state(
            {
                **expected,
                "status": "complete",
                "turns": 1,
                "input_tokens": token_count,
                "terminal_record": {
                    "turn": 1,
                    "raw_response": completion.content,
                    "provider_retry": provider_retry,
                    "usage": completion.usage,
                },
                "prediction": prediction,
                "failure": None,
            }
        )
        return prediction

    def _save_failure(
        self,
        state: dict[str, Any],
        failure: str,
        token_count: int,
        *,
        completion: ModelCompletion | None = None,
        provider_retry: int | None = None,
    ) -> None:
        terminal: dict[str, Any] = {"turn": 1, "failure": failure}
        if completion is not None:
            terminal.update(
                {
                    "raw_response": completion.content,
                    "provider_retry": provider_retry,
                    "usage": completion.usage,
                }
            )
        self.store.save_state(
            {
                **state,
                "status": "failed",
                "turns": 1,
                "input_tokens": token_count,
                "terminal_record": terminal,
                "prediction": None,
                "failure": failure,
            }
        )

    def _request(
        self, messages: list[dict[str, str]], token_count: int
    ) -> tuple[ModelCompletion, int]:
        _, request_policy = prepare_trace_request(
            messages,
            config=self.config,
        )
        for retry in range(self.config.network_retries + 1):
            self.store.save_request_if_unchanged(
                1,
                messages=messages,
                token_count=token_count,
                compression=request_policy,
                provider_retry=retry,
            )
            try:
                completion = self.client.complete(
                    messages,
                    max_output_tokens=self.config.max_output_tokens,
                )
            except RetryableModelError as error:
                self.store.save_provider_failure(
                    1,
                    provider_retry=retry,
                    error=str(error),
                    raw_response=error.raw_response,
                    usage=error.usage,
                )
                if retry == self.config.network_retries:
                    raise
                continue
            except ProviderContextError:
                raise
            self.store.save_response(
                1,
                content=completion.content,
                raw_response=completion.raw_response,
                usage=completion.usage,
                provider_retry=retry,
            )
            return completion, retry
        raise AssertionError("unreachable")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")
