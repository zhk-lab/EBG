"""Module 4 ablation (Repo only): fixed entry order and no document anchors."""

from copy import deepcopy

from agentloop.graph_backend import GraphBackend


class FixedEntryBackend(GraphBackend):
    def __init__(self, bundle, graph, directory, *, count_tokens=None):
        super().__init__(bundle, graph, directory, count_tokens=count_tokens)
        retriever = self._retriever
        retriever.document_section_by_endpoint = {}
        retriever.module_root_evidence_ids = {}
        retriever.direct_endpoints = set()
        retriever.root_symbols_by_path = {
            path: (endpoints[0][1],) for path, endpoints in retriever.endpoints_by_path.items() if endpoints
        }
        # Retain the common root-budget algorithm; replace only its task-based candidates.
        selected = []
        lines = ["[FILES]", "read_id | path"]
        counter = retriever.count_tokens
        for item in sorted(directory["query_index"], key=lambda x: (x["path"].casefold(), x["path"])):
            line = f"{item['read_id']} | {item['path']}"
            if counter("\n".join([*lines, line])) > directory["token_budget"]:
                break
            lines.append(line)
            selected.append({"read_id": item["read_id"], "path": item["path"],
                             "related_document_sections": [], "code_hints": ""})
        if not selected:
            raise ValueError("fixed entry index cannot fit even one file")
        self.initial_index = "\n".join(lines)
        self.initial_index_tokens = counter(self.initial_index)
        self._initial_ids = frozenset(item["read_id"] for item in selected)
        self._priority_groups = {}
        # Stage 2 uses the same fixed top files, without reintroducing task-ranked entries.
        self.effective_directory = deepcopy(directory)
        self.effective_directory.update(entries=selected, token_count=self.initial_index_tokens)

    def _unit_from_local_graph(self, read_id, path, local_graph):
        clean = deepcopy(local_graph)
        for group in clean["graphs"]:
            group["root"].pop("document_section", None)
        return super()._unit_from_local_graph(read_id, path, clean)
