"""Module 3 ablation: flat Behaviors without Scope grouping or graph traversal."""

import json
from dataclasses import replace

from agentloop.graph_backend import GraphBackend, _numbered_source_regions
from ebg.behavior_directory import build_ranked_directory
from tracereview import TraceView, render_trace_view


class BehaviorListBackend(GraphBackend):
    def __init__(self, bundle, graph, directory, *, count_tokens=None):
        # Rebuild the entry list too: graph-derived neighboring entries must disappear.
        without_edges = {**graph, "edges": []}
        directory = build_ranked_directory(bundle, without_edges, count_tokens=count_tokens)
        super().__init__(bundle, without_edges, directory, count_tokens=count_tokens,
                         expand_neighbors=False)

    def _unit_from_local_graph(self, read_id, path, local_graph):
        base = super()._unit_from_local_graph(read_id, path, local_graph)
        evidence_by_id = self._retriever.evidence_by_id
        behaviors_by_id = {b["behavior_id"]: b for b in self._retriever.graph["behaviors"]}
        records, seen = [], set()
        for group in local_graph["graphs"]:
            node = group["root"]
            if "ref" in node:
                continue
            covered = set()
            for compact in node["behaviors"]:
                b = behaviors_by_id[compact["behavior_id"]]
                record = {"behavior_id": b["behavior_id"], "path": b["path"],
                          "result_type": b["result_type"]}
                if node.get("document_section"):
                    record["document_section"] = node["document_section"]
                for role, key in (("trigger", "trigger_evidence_ids"),
                                  ("operation", "operation_evidence_ids"),
                                  ("result", "result_evidence_ids")):
                    values = []
                    for eid in b[key]:
                        item = evidence_by_id[eid]
                        loc = item["locator"]
                        covered.update(range(loc["line_start"], loc["line_end"] + 1))
                        values.append({"id": eid, "lines": [loc["line_start"], loc["line_end"]],
                                       "content": item["content"]})
                    record[role] = values
                if b["behavior_id"] not in seen:
                    records.append(record)
                    seen.add(b["behavior_id"])
            # Keep source lines not represented by a Behavior (e.g. declarations/comments).
            remaining = []
            for start, _, source in _numbered_source_regions(node["source"]):
                remaining.extend(line for n, line in enumerate(source.splitlines(), start) if n not in covered)
            if remaining:
                records.append({"path": node["path"], "source": "\n".join(remaining)})
        return replace(base, unit_kind="source", name=f"{path} evidence",
                       source=json.dumps({"read_id": read_id, "items": records},
                                         ensure_ascii=False, separators=(",", ":")))


def render_trace(graph):
    original = render_trace_view(graph)
    payload = json.loads(original.text)
    by_id = {b["behavior_id"]: b for scope in payload["task_scopes"] for b in scope["behaviors"]}
    behaviors = [by_id[b["behavior_id"]] for b in graph["behaviors"]]
    for b in behaviors:
        b.pop("sequence_index")  # This index is local to a Task, not a global event order.
    result = {"input_id": original.input_id, "benchmark": "feedbacktrace", "behaviors": behaviors}
    return TraceView(original.input_id, json.dumps(result, ensure_ascii=False,
                     separators=(",", ":")) + "\n", original.evidence_ids)
