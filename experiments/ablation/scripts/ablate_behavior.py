"""Module 2 ablation: show existing Evidence individually inside unchanged scopes."""

import json
from dataclasses import replace

from agentloop.evidence import SourceRegion
from agentloop.graph_backend import GraphBackend, _numbered_source_regions
from tracereview import TraceView, render_trace_view
from tracereview.runner import _display_event


class EvidenceBackend(GraphBackend):
    def _unit_from_local_graph(self, read_id, path, local_graph):
        base = super()._unit_from_local_graph(read_id, path, local_graph)
        regions = []

        def evidence_node(node):
            if "ref" in node:
                return dict(node)
            ranges = [(a, b) for a, b, _ in _numbered_source_regions(node["source"])]
            evidence = []
            items = self._retriever.evidence_by_path[node["path"]]
            for item in sorted(items, key=lambda x: (x["locator"]["line_start"], x["evidence_id"])):
                loc = item["locator"]
                if not any(loc["line_start"] <= b and loc["line_end"] >= a for a, b in ranges):
                    continue
                evidence.append({"id": item["evidence_id"],
                                 "lines": [loc["line_start"], loc["line_end"]],
                                 "content": item["content"]})
                numbered = "\n".join(f"{n} | {line}" for n, line in
                                     enumerate(item["content"].splitlines(), loc["line_start"]))
                regions.append(SourceRegion((loc["symbol"],), loc["line_start"],
                                            loc["line_end"], numbered, loc["path"]))
            result = {"path": node["path"], "scope": node["symbol"],
                      "lines": node["lines"], "evidence": evidence}
            if node.get("document_section"):
                result["document_section"] = node["document_section"]
            return result

        graphs = [{"root": evidence_node(g["root"]),
                   "paths": [{"steps": [{"edge": step["edge"], "node": evidence_node(step["node"])}
                                         for step in p["steps"]]} for p in g["paths"]]}
                  for g in local_graph["graphs"]]
        source = json.dumps({"read_id": read_id, "graphs": graphs}, ensure_ascii=False, separators=(",", ":"))
        return replace(base, source=source, source_regions=tuple(regions),
                       groundable=bool(regions), behavior_total=0)


def render_trace(graph):
    original = render_trace_view(graph)  # Reuse validation and the unchanged Scope/edge view.
    payload = json.loads(original.text)
    evidence = {e["evidence_id"]: e for e in graph["evidence"]}
    order = {e["evidence_id"]: i for i, e in enumerate(graph["evidence"])}
    for scope in payload["task_scopes"]:
        refs = {}
        for behavior in graph["behaviors"]:
            if behavior["task_id"] != scope["task_id"]:
                continue
            for ref in [*behavior["demand_refs"], *behavior["response_refs"],
                        *({"evidence_id": i} for i in behavior["action_evidence_ids"])]:
                eid = ref["evidence_id"]
                span = tuple(ref.get("char_range", (0, len(evidence[eid]["content"]))))
                refs.setdefault(eid, []).append(span)
        events = []
        for eid in sorted(refs, key=order.get):
            merged = []
            for a, b in sorted(refs[eid]):
                if merged and a <= merged[-1][1]:
                    merged[-1][1] = max(b, merged[-1][1])
                else:
                    merged.append([a, b])
            for span in merged:
                locator = evidence[eid]["locator"]
                events.append({"id": eid, "char_range": span,
                               "turn": locator["turn"], "event_index": locator["event_index"],
                               "event_type": locator["event_type"],
                               **_display_event(eid, evidence, char_range=span)})
        scope.pop("behaviors")
        scope["evidence"] = events
    return TraceView(original.input_id, json.dumps(payload, ensure_ascii=False,
                     separators=(",", ":")) + "\n", original.evidence_ids)
