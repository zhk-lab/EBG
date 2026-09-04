"""Single deterministic state machine shared by interactive evaluations."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from typing import Any

from .backend import EvidenceBackend
from .config import DEFAULT_CONFIG, AgentLoopConfig
from .context import (
    FastTokenCounter,
    HistoryEntry,
    PreparedRequest,
    prepare_messages,
)
from .errors import (
    ActionParameterError,
    AgentLoopError,
    ContextUnfitError,
    PredictionError,
    PredictionFormatError,
    ProtocolError,
    ProviderContextError,
    RetryableModelError,
)
from .evidence import EvidenceSpan, ToolResult
from .finish_contract import FinishContract, validated_finish_contract_identity
from .prompts import system_prompt
from .protocol import FinishAction, ReadAction, SearchAction, parse_action
from .provider import ModelClient, ModelCompletion, validated_model_profile
from .storage import RunStore


STATE_VERSION = 6


@dataclass(frozen=True, slots=True)
class RunOutcome:
    status: str
    turns: int
    prediction: dict[str, Any] | None
    failure: str | None


def _response_attempts_finish(content: str) -> bool:
    return bool(re.search(r'"action"\s*:\s*"finish"', content))


def _finish_format_example(benchmark: str) -> str:
    if benchmark == "silentswap":
        swaps = []
        for slot in range(1, 6):
            swaps.append(
                {
                    "target": {
                        "file": f"<read path for swap {slot}>",
                        "symbol": {
                            "kind": "function",
                            "qualified_name": [
                                f"<source identifier for swap {slot}>"
                            ],
                        },
                        "line_ranges": [{"start": slot, "end": slot}],
                    },
                    "swap_type": "input_validation_boundary",
                    "code_change": "<original operation -> current operation>",
                    "trigger_condition": "<distinguishing condition>",
                    "behavioral_effect": {
                        "before": "<documented original outcome>",
                        "after": "<implemented current outcome>",
                    },
                }
            )
        prediction: dict[str, Any] = {"swaps": swaps}
    elif benchmark == "specgap":
        prediction = {
            "findings": [
                {
                    "finding_id": "F001",
                    "claim": "<missing implemented contract>",
                    "verification_question": "<question resolving the contract>",
                    "why_important": "<practical importance>",
                    "downstream_impact": "<changed behavior or decision>",
                    "code_evidence": [
                        {
                            "path": "<read repository path>",
                            "start_line": 1,
                            "end_line": 1,
                            "symbol": "<enclosing symbol>",
                            "explanation": "<what the lines establish>",
                        }
                    ],
                }
            ]
        }
    else:
        prediction = {}
    return json.dumps(
        {"action": "finish", "prediction": prediction},
        ensure_ascii=False,
        indent=2,
    )


def prepare_initial_request(
    initial_user_prompt: str,
    max_rounds: int,
    *,
    config: AgentLoopConfig = DEFAULT_CONFIG,
    priority_groups: dict[str, tuple[str, ...]] | None = None,
) -> PreparedRequest:
    """Build and validate turn one without constructing a model client."""

    if not isinstance(initial_user_prompt, str) or not initial_user_prompt.strip():
        raise AgentLoopError("initial user prompt must be non-empty")
    if type(max_rounds) is not int or max_rounds <= 0:
        raise AgentLoopError("max_rounds must be a positive integer")
    return prepare_messages(
        system=system_prompt(config),
        initial_user=initial_user_prompt,
        history=(),
        max_rounds=max_rounds,
        config=config,
        counter=FastTokenCounter(config.token_estimator),
        priority_groups=priority_groups,
    )


class AgentLoop:
    """Run or resume one interactive evidence audit."""

    def __init__(
        self,
        *,
        backend: EvidenceBackend,
        initial_user_prompt: str,
        max_rounds: int,
        finish_contract: FinishContract,
        client: ModelClient,
        store: RunStore,
        config: AgentLoopConfig = DEFAULT_CONFIG,
    ) -> None:
        if not isinstance(initial_user_prompt, str) or not initial_user_prompt.strip():
            raise AgentLoopError("initial user prompt must be non-empty")
        if type(max_rounds) is not int or max_rounds <= 0:
            raise AgentLoopError("max_rounds must be a positive integer")
        self.backend = backend
        self.client = client
        self.model_profile = validated_model_profile(client)
        self.store = store
        self.config = config
        self.max_rounds = max_rounds
        self.finish_contract = finish_contract
        self.finish_contract_identity = validated_finish_contract_identity(
            finish_contract
        )
        self.counter = FastTokenCounter(config.token_estimator)
        if backend.initial_index_tokens > config.index_budget:
            raise AgentLoopError("initial repository index exceeds index_budget")
        self.system = system_prompt(config)
        self.initial_user = initial_user_prompt

    def run(self) -> RunOutcome:
        state = self._load_or_initialize()
        if state["status"] == "complete":
            return RunOutcome(
                status="complete",
                turns=int(state["turns"]),
                prediction=state.get("prediction"),
                failure=None,
            )
        if state["status"] == "failed":
            return RunOutcome(
                status="failed",
                turns=int(state.get("turns", 0)),
                prediction=None,
                failure=str(state.get("failure") or "unknown_failure"),
            )

        records = list(state["records"])
        while len(records) < self.max_rounds:
            history = self._history(records)
            turn = len(records) + 1
            prior_format_repairs = sum(
                int(record.get("format_retry", 0)) for record in records
            )
            try:
                base_prepared = prepare_messages(
                    system=self.system,
                    initial_user=self.initial_user,
                    history=history,
                    max_rounds=self.max_rounds,
                    config=self.config,
                    counter=self.counter,
                    priority_groups=self.backend.priority_groups,
                )
            except ContextUnfitError:
                return self._fail(state, records, turn, "context_unfit")

            format_errors: list[dict[str, Any]] = []
            format_retry = 0
            repair_context: tuple[str, str, str] | None = None
            while True:
                try:
                    completion, prepared, provider_retry = self._completion(
                        turn,
                        history,
                        base_prepared,
                        format_retry=format_retry,
                        repair_context=repair_context,
                    )
                except ContextUnfitError:
                    return self._fail(
                        state,
                        records,
                        turn,
                        "context_unfit",
                        terminal_record={
                            "turn": turn,
                            "action": None,
                            "format_errors": format_errors,
                        },
                    )
                except ProviderContextError:
                    return self._fail(
                        state,
                        records,
                        turn,
                        "context_unfit/provider_limit_mismatch",
                        terminal_record={
                            "turn": turn,
                            "action": None,
                            "format_errors": format_errors,
                        },
                    )
                except RetryableModelError:
                    return self._fail(
                        state,
                        records,
                        turn,
                        "network_retries_exhausted",
                        terminal_record={
                            "turn": turn,
                            "action": None,
                            "format_errors": format_errors,
                        },
                    )
                except AgentLoopError:
                    return self._fail(
                        state,
                        records,
                        turn,
                        "provider_response_invalid",
                        terminal_record={
                            "turn": turn,
                            "action": None,
                            "format_errors": format_errors,
                        },
                    )

                terminal = self._terminal_record(
                    turn,
                    completion,
                    prepared,
                    provider_retry,
                    format_retry,
                    format_errors,
                )
                issue_category: str | None = None
                issue_error: str | None = None
                try:
                    action = parse_action(
                        completion.content,
                        max_read_ids=self.config.max_read_ids,
                        max_search_characters=self.config.max_search_characters,
                    )
                except (ActionParameterError, ProtocolError) as error:
                    issue_category = (
                        "finish_schema"
                        if (
                            isinstance(error, ActionParameterError)
                            and error.action == "finish"
                        )
                        or _response_attempts_finish(completion.content)
                        else "action_protocol"
                    )
                    issue_error = str(error)
                else:
                    correcting_finish = (
                        repair_context is not None
                        and repair_context[1] == "finish_schema"
                    )
                    if correcting_finish and not isinstance(action, FinishAction):
                        issue_category = "finish_schema"
                        issue_error = (
                            "a finish schema correction must remain a finish action"
                        )
                    elif isinstance(action, FinishAction):
                        terminal = {**terminal, "action": action.as_dict()}
                        try:
                            prediction = self.finish_contract.validate(
                                action.prediction,
                                input_id=self.backend.input_id,
                                observed_spans=self._observed_spans(records),
                            )
                            if not isinstance(prediction, dict):
                                raise PredictionFormatError(
                                    "finish contract must return a prediction object"
                                )
                        except PredictionFormatError as error:
                            issue_category = "finish_schema"
                            issue_error = str(error)
                        except PredictionError as error:
                            return self._fail(
                                state,
                                records,
                                turn,
                                "invalid_finish_grounding_or_content",
                                terminal_record={
                                    **terminal,
                                    "validation_error": str(error),
                                },
                            )
                        else:
                            self.store.save_prediction_response(action.as_dict())
                            self.store.save_prediction(prediction)
                            completed = {
                                **state,
                                "status": "complete",
                                "turns": turn,
                                "records": records,
                                "terminal_record": terminal,
                                "prediction": prediction,
                                "failure": None,
                            }
                            self.store.save_state(completed)
                            return RunOutcome("complete", turn, prediction, None)
                    else:
                        break

                assert issue_category is not None and issue_error is not None
                format_errors.append(
                    self._format_error_record(
                        category=issue_category,
                        error=issue_error,
                        completion=completion,
                        prepared=prepared,
                        provider_retry=provider_retry,
                        format_retry=format_retry,
                    )
                )
                terminal = {
                    **terminal,
                    "format_errors": list(format_errors),
                    "format_error_category": issue_category,
                    "format_error": issue_error,
                }
                if (
                    prior_format_repairs + format_retry
                    >= self.config.format_repair_attempts
                ):
                    failure = (
                        "finish_format_retries_exhausted"
                        if issue_category == "finish_schema"
                        else "action_format_retries_exhausted"
                    )
                    return self._fail(
                        state,
                        records,
                        turn,
                        failure,
                        terminal_record=terminal,
                    )
                repair_context = (
                    completion.content,
                    issue_category,
                    issue_error,
                )
                format_retry += 1

            if turn == self.max_rounds:
                return self._fail(
                    state,
                    records,
                    turn,
                    "last_round_requires_finish",
                    terminal_record={**terminal, "action": action.as_dict()},
                )

            known_ids = self._known_ids(records)
            if isinstance(action, SearchAction):
                result = self.backend.search(
                    action.text,
                    limit=self.config.max_search_hits,
                    exact_path_limit=self.config.max_exact_path_hits,
                )
            else:
                assert isinstance(action, ReadAction)
                if any(
                    unit_id not in known_ids or not self.backend.has_id(unit_id)
                    for unit_id in action.ids
                ):
                    result = ToolResult(
                        action="read",
                        status="error",
                        requested_ids=action.ids,
                        error_code="unknown_or_unexposed_id",
                    )
                else:
                    try:
                        result = self.backend.read(
                            action.ids,
                            token_budget=self.config.tool_result_budget,
                            max_atomic_unit_tokens=self.config.max_atomic_unit_tokens,
                            count_tokens=self.counter.count_text,
                        )
                    except ContextUnfitError:
                        return self._fail(
                            state,
                            records,
                            turn,
                            "atomic_unit_exceeds_limit",
                            terminal_record={
                                **terminal,
                                "action": action.as_dict(),
                            },
                        )
            records.append(
                self._record(
                    turn,
                    completion,
                    action.as_dict(),
                    result,
                    prepared,
                    provider_retry,
                    format_retry,
                    format_errors,
                )
            )
            self._save_running(state, records)

        return self._fail(state, records, self.max_rounds, "round_limit_exhausted")

    def _completion(
        self,
        turn: int,
        history: list[HistoryEntry],
        base_prepared: PreparedRequest,
        *,
        format_retry: int,
        repair_context: tuple[str, str, str] | None,
    ) -> tuple[ModelCompletion, PreparedRequest, int]:
        prepared = self._apply_repair_context(
            base_prepared,
            format_retry=format_retry,
            repair_context=repair_context,
        )
        persisted = self.store.load_response(
            turn,
            format_retry=format_retry,
        )
        if persisted is not None:
            provider_retry = int(persisted.get("provider_retry", 0))
            if provider_retry >= 100:
                forced = prepare_messages(
                    system=self.system,
                    initial_user=self.initial_user,
                    history=history,
                    max_rounds=self.max_rounds,
                    config=self.config,
                    counter=self.counter,
                    force_compression=True,
                    priority_groups=self.backend.priority_groups,
                )
                prepared = self._apply_repair_context(
                    forced,
                    format_retry=format_retry,
                    repair_context=repair_context,
                )
            request = self.store.load_request(
                turn,
                provider_retry,
                format_retry=format_retry,
            )
            if request is None or (
                request.get("messages") != list(prepared.messages)
                or request.get("token_count") != prepared.token_count
            ):
                raise AgentLoopError(
                    f"persisted response has no matching request for turn {turn}"
                )
            return (
                ModelCompletion(
                    content=str(persisted["content"]),
                    raw_response=persisted.get("raw_response", {}),
                    usage=persisted.get("usage", {}),
                ),
                prepared,
                provider_retry,
            )
        try:
            return self._request(
                turn,
                prepared,
                retry_offset=0,
                format_retry=format_retry,
            )
        except ProviderContextError:
            forced_base = prepare_messages(
                system=self.system,
                initial_user=self.initial_user,
                history=history,
                max_rounds=self.max_rounds,
                config=self.config,
                counter=self.counter,
                force_compression=True,
                priority_groups=self.backend.priority_groups,
            )
            forced = self._apply_repair_context(
                forced_base,
                format_retry=format_retry,
                repair_context=repair_context,
            )
            if (
                forced.manifest.search_receipt_turns
                == prepared.manifest.search_receipt_turns
                and forced.manifest.read_receipt_units
                == prepared.manifest.read_receipt_units
                and forced.manifest.duplicate_receipt_units
                == prepared.manifest.duplicate_receipt_units
            ) or forced.token_count >= prepared.token_count:
                raise
            return self._request(
                turn,
                forced,
                retry_offset=100,
                format_retry=format_retry,
            )

    def _apply_repair_context(
        self,
        prepared: PreparedRequest,
        *,
        format_retry: int,
        repair_context: tuple[str, str, str] | None,
    ) -> PreparedRequest:
        if format_retry == 0:
            if repair_context is not None:
                raise AgentLoopError("initial completion cannot have repair context")
            return prepared
        if repair_context is None:
            raise AgentLoopError("format repair lacks the rejected response")
        rejected, category, error = repair_context
        if category == "finish_schema":
            correction = "\n".join(
                (
                    "Your finish JSON format is invalid. Please strictly correct "
                    "it to the format below:",
                    _finish_format_example(self.backend.benchmark),
                )
            )
        else:
            allowed = (
                '- {"action":"search","text":"..."}\n'
                '- {"action":"read","ids":["..."]}\n'
                '- {"action":"finish","prediction":{...}}'
            )
            correction_lines = [
                "[[ACTION FORMAT CORRECTION]]",
                "This correction uses the sample-wide format-repair budget "
                f"of {self.config.format_repair_attempts} calls.",
                f"Rejected category: {category}",
                f"Validation error: {error}",
                "Pick exactly one corrected template below. Output only that one "
                "bare JSON object, then stop:",
                allowed,
                "Do not repeat the rejected JSON unchanged. Apply every validation "
                "error while preserving the same semantic answer.",
                "If the rejected response contained multiple actions, keep only its "
                "first immediate evidence action. Never keep a finish that depended "
                "on an unexecuted search or read.",
                "Correct only the action or prediction format. Do not request, "
                "read, or assume any new evidence in this correction.",
                "[[END ACTION FORMAT CORRECTION]]",
            ]
            correction = "\n".join(correction_lines)
        messages = (
            *prepared.messages,
            {"role": "assistant", "content": rejected},
            {"role": "user", "content": correction},
        )
        token_count = self.counter.count_messages(messages)
        if token_count > self.config.physical_hard_limit:
            raise ContextUnfitError(
                "format correction does not fit the provider context"
            )
        manifest = replace(
            prepared.manifest,
            tokens_after=token_count,
            protected_oversize=(
                prepared.manifest.protected_oversize
                or token_count > self.config.compression_target
            ),
        )
        return PreparedRequest(messages, token_count, manifest)

    def _request(
        self,
        turn: int,
        prepared: PreparedRequest,
        *,
        retry_offset: int,
        format_retry: int,
    ) -> tuple[ModelCompletion, PreparedRequest, int]:
        messages = list(prepared.messages)
        for retry in range(self.config.network_retries + 1):
            retry_number = retry_offset + retry
            self.store.save_request_if_unchanged(
                turn,
                messages=messages,
                token_count=prepared.token_count,
                compression=prepared.manifest.to_dict(),
                provider_retry=retry_number,
                format_retry=format_retry,
            )
            try:
                completion = self.client.complete(
                    messages,
                    max_output_tokens=self.config.max_output_tokens,
                )
            except RetryableModelError as error:
                self.store.save_provider_failure(
                    turn,
                    provider_retry=retry_number,
                    error=str(error),
                    raw_response=error.raw_response,
                    usage=error.usage,
                    format_retry=format_retry,
                )
                if retry == self.config.network_retries:
                    raise
                continue
            self.store.save_response(
                turn,
                content=completion.content,
                raw_response=completion.raw_response,
                usage=completion.usage,
                provider_retry=retry_number,
                format_retry=format_retry,
            )
            return completion, prepared, retry_number
        raise AssertionError("unreachable")

    def _load_or_initialize(self) -> dict[str, Any]:
        expected = {
            "schema_version": STATE_VERSION,
            "input_id": self.backend.input_id,
            "benchmark": self.backend.benchmark,
            "backend": self.backend.kind,
            "config": self.config.public_dict(),
            "max_rounds": self.max_rounds,
            "finish_contract": self.finish_contract_identity,
            "system": self.system,
            "initial_user": self.initial_user,
            "model_profile": self.model_profile,
            "backend_identity": self.backend.resume_identity,
        }
        state = self.store.load_state()
        if state is None:
            state = {
                **expected,
                "status": "running",
                "turns": 0,
                "records": [],
                "terminal_record": None,
                "prediction": None,
                "failure": None,
            }
            self.store.save_state(state)
            return state
        for key, value in expected.items():
            if state.get(key) != value:
                raise AgentLoopError(f"resume state does not match current {key}")
        records = state.get("records")
        if not isinstance(records, list):
            raise AgentLoopError("resume state records must be an array")
        for index, record in enumerate(records, start=1):
            if not isinstance(record, dict) or record.get("turn") != index:
                raise AgentLoopError("resume state has a non-contiguous action ledger")
            ToolResult.from_dict(record["tool_result"])
        status = state.get("status")
        turns = state.get("turns")
        if not isinstance(turns, int) or turns < 0:
            raise AgentLoopError("resume state has an invalid turn count")
        if status == "running":
            if turns != len(records) or state.get("terminal_record") is not None:
                raise AgentLoopError("running state has an invalid action ledger")
        elif status in {"complete", "failed"}:
            terminal = state.get("terminal_record")
            if not isinstance(terminal, dict) or terminal.get("turn") != turns:
                raise AgentLoopError("terminal state lacks its final turn record")
        else:
            raise AgentLoopError("resume state has an invalid status")
        return state

    def _save_running(
        self, base: dict[str, Any], records: list[dict[str, Any]]
    ) -> None:
        self.store.save_state(
            {
                **base,
                "status": "running",
                "turns": len(records),
                "records": records,
                "terminal_record": None,
                "prediction": None,
                "failure": None,
            }
        )

    def _fail(
        self,
        base: dict[str, Any],
        records: list[dict[str, Any]],
        turn: int,
        failure: str,
        terminal_record: dict[str, Any] | None = None,
    ) -> RunOutcome:
        self.store.save_state(
            {
                **base,
                "status": "failed",
                "turns": turn,
                "records": records,
                "terminal_record": {
                    **(terminal_record or {"turn": turn}),
                    "failure": failure,
                },
                "prediction": None,
                "failure": failure,
            }
        )
        return RunOutcome("failed", turn, None, failure)

    @staticmethod
    def _history(records: list[dict[str, Any]]) -> list[HistoryEntry]:
        return [
            HistoryEntry(
                turn=int(record["turn"]),
                raw_response=json.dumps(
                    record["action"],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                action=record["action"],
                tool_result=ToolResult.from_dict(record["tool_result"]),
            )
            for record in records
        ]

    def _known_ids(self, records: list[dict[str, Any]]) -> set[str]:
        result = set(self.backend.initial_ids)
        for entry in self._history(records):
            tool = entry.tool_result
            result.update(tool.returned_ids)
            for unit in tool.units:
                result.update(item.unit_id for item in unit.related)
        return result

    @staticmethod
    def _observed_spans(records: list[dict[str, Any]]) -> tuple[EvidenceSpan, ...]:
        result: list[EvidenceSpan] = []
        for entry in AgentLoop._history(records):
            for span in entry.tool_result.displayed_spans:
                if span not in result:
                    result.append(span)
        return tuple(result)

    @staticmethod
    def _record(
        turn: int,
        completion: ModelCompletion,
        action: dict[str, Any],
        result: ToolResult,
        prepared: PreparedRequest,
        provider_retry: int,
        format_retry: int,
        format_errors: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "turn": turn,
            "raw_response": completion.content,
            "action": action,
            "tool_result": result.to_dict(),
            "request_tokens": prepared.token_count,
            "compression": prepared.manifest.to_dict(),
            "provider_retry": provider_retry,
            "format_retry": format_retry,
            "format_errors": list(format_errors),
            "usage": completion.usage,
        }

    @staticmethod
    def _terminal_record(
        turn: int,
        completion: ModelCompletion,
        prepared: PreparedRequest,
        provider_retry: int,
        format_retry: int,
        format_errors: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "turn": turn,
            "raw_response": completion.content,
            "action": None,
            "request_tokens": prepared.token_count,
            "compression": prepared.manifest.to_dict(),
            "provider_retry": provider_retry,
            "format_retry": format_retry,
            "format_errors": list(format_errors),
            "usage": completion.usage,
        }

    @staticmethod
    def _format_error_record(
        *,
        category: str,
        error: str,
        completion: ModelCompletion,
        prepared: PreparedRequest,
        provider_retry: int,
        format_retry: int,
    ) -> dict[str, Any]:
        return {
            "format_retry": format_retry,
            "category": category,
            "error": error,
            "raw_response": completion.content,
            "request_tokens": prepared.token_count,
            "compression": prepared.manifest.to_dict(),
            "provider_retry": provider_retry,
            "usage": completion.usage,
        }
