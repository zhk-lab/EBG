"""Adapt captured events to BEG while retaining every original event separately."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..beg.behavior_atomization import build_behaviors
from ..beg.core.model import TraceEvent, VisibleBundle
from ..beg.evidence_intake import build_evidence
from ..beg.graph_assembly import build_graph
from ..beg.relation_linking import build_edges

from .matching import literal_match, trace_signal


def build_trace(events: list[dict[str, Any]]) -> dict[str, Any]:
    signals = {event["id"]: trace_signal(event["content"], tool_name=event.get("tool_name"))
               for event in events if event["kind"] in {"tool_call", "assistant"}}
    results = {event["call_id"]: event for event in events if event["kind"] == "tool_result"}
    converted = []
    pairs = {}
    for event in events:
        kind = event["kind"]
        if kind == "tool_result":
            continue
        text = event["content"]
        event_type = {"user": "user_prompt", "assistant": "assistant_response", "tool_call": "tool_exchange"}[kind]
        if kind == "tool_call":
            result = results.get(event["call_id"])
            # This wrapper is only an internal BEG adapter, never returned as raw evidence.
            text = f"Tool invocation:\n{text}"
            if result:
                text += f"\n\nTool result:\n{result['content']}"
            pairs[event["id"]] = [event["id"], *([result["id"]] if result else [])]
        converted.append(TraceEvent(event_type, event["turn"], text, event["id"], event.get("tool_name"), event["seq"]))
    if not converted:
        return {"evidence": [], "behaviors": [], "edges": [], "pairs": {}, "signals": signals}
    bundle = VisibleBundle(Path("."), "harness", "feedbacktrace", None, (), tuple(converted), max(e.turn_number for e in converted), 0, 0)
    evidence = build_evidence(bundle)
    behaviors = build_behaviors(bundle, evidence)
    edges = build_edges(bundle, evidence, behaviors)
    return {**build_graph(bundle, evidence, behaviors, edges), "pairs": pairs, "signals": signals}


def matched_events(requirement: dict[str, Any], events: list[dict[str, Any]], paths: list[str],
                   signals: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Use BEG's operation/object signals; require explicit operation compatibility.

Never promote a read of a test file to evidence that tests ran, or join two
fully named files merely because BEG's basename aliases overlap.
"""
    text = "\n".join(ref["content"] for ref in requirement["refs"])
    demand = trace_signal(text)
    demand_paths = {path for path in paths if literal_match(text, path, path=True)}
    results = {event["call_id"]: event for event in events if event["kind"] == "tool_result"}
    selected = {}
    for event in events:
        if event["kind"] not in {"tool_call", "assistant"}:
            continue
        signal = signals[event["id"]] if signals and event["id"] in signals else trace_signal(event["content"], tool_name=event.get("tool_name"))
        overlap = set(demand["objects"]) & set(signal["objects"])
        event_paths = {path for path in paths if literal_match(event["content"], path, path=True) or path.casefold() in signal["objects"]}
        if demand_paths and event_paths and not demand_paths & event_paths:
            continue
        if not overlap:
            continue
        if demand_paths and not event_paths:
            ambiguous = {Path(path).name.casefold() for path in demand_paths
                         if sum(Path(other).name.casefold() == Path(path).name.casefold() for other in paths) > 1}
            if overlap and overlap <= ambiguous:
                continue
        if demand["operations"] and event["kind"] == "tool_call":
            if not set(demand["operations"]) & set(signal["operations"]):
                continue
        selected[event["id"]] = event
        if event["kind"] == "tool_call" and event["call_id"] in results:
            result = results[event["call_id"]]
            selected[result["id"]] = result
    return sorted(selected.values(), key=lambda event: event["seq"])
