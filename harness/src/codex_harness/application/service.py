"""Checkpoint evidence service shared by MCP and lifecycle hooks."""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import Any

from .matching import direct_repo_roots, explicit_symbol_match, literal_match, section_navigation, source_line_paths
from .checks import Checks
from .paging import OutputPages
from .repository import GRAPH_VERSION, construct_graph, module_statement_span
from .artifacts import linked_artifacts
from .requirements import normalize_requirements
from .research import research_context
from .storage import HarnessError, Store
from .sessions import SessionLog
from .selection import compact_contexts, enclosing_contexts, prioritize_repo_entries, share_repo_excerpts, value_contexts
from .trace import build_trace, matched_events
from .source_refs import SourceReader
from .components import components, node_id, relations_by_node


class Harness:
    def __init__(self, state_dir: str | Path, *, token_budget: int = 12_000, max_file_bytes: int = 2_000_000,
                 review_call_threshold: int = 10, review_seconds_threshold: float = 300) -> None:
        if token_budget < 256 or max_file_bytes < 1:
            raise HarnessError("token_budget must be >=256 and max_file_bytes positive.")
        self.store = Store(state_dir)
        self.token_budget = token_budget
        self.max_file_bytes = max_file_bytes
        if review_call_threshold < 1 or review_seconds_threshold <= 0:
            raise HarnessError('Review thresholds must be positive.')
        self.review_call_threshold = review_call_threshold
        self.review_seconds_threshold = review_seconds_threshold
        self.sessions = SessionLog(self.store, max_file_bytes)
        self.checks = Checks(self)

    @property
    def pages(self) -> OutputPages:
        return OutputPages(self.store, self.token_budget)

    def build_evidence_groups(self, task_id: str, requirements: list[dict[str, Any]] | None = None,
                              *, read_ref: str | None = None, offset: int = 0) -> str:
        task = self.store.task(task_id)
        if task['scope'].get('mode') != 'checkpoint':
            raise HarnessError('Evidence requires a checkpoint; create one with ebg_review.')
        self.sessions.current(task['scope']['session_id'])
        owner = f"evidence:{task_id}"
        if read_ref:
            if requirements is not None:
                raise HarnessError("Read fixed evidence or submit a checklist, not both.")
            if read_ref.startswith('V'):
                view_id, separator, material_id = read_ref.partition(':')
                view = self.store.view(view_id)
                if not separator or view['task_id'] != task_id:
                    raise HarnessError("Evidence reference belongs to a different task.")
                if material_id == 'changes':
                    value = [{'path': p, 'change': c, 'read_ref': f'{view_id}:diff:{p}'}
                             for p, c in view['changes'].items()]
                else:
                    value = self.material(view_id, material_id)
                read_ref = self.pages.save(owner, value)
            return self.pages.read(owner, read_ref, offset)
        if offset:
            raise HarnessError("Use the read_ref returned with next to continue evidence reads.")
        if requirements is not None:
            normalized = normalize_requirements(requirements, task['sources'])
            if normalized != task['requirements']:
                task['requirements'] = normalized
                self.store.put_task(task)
        if not task['requirements']:
            raise HarnessError("Supply a normalized requirements checklist with original source references.")
        refreshed = self.refresh_task(task_id)
        view = self.store.view(refreshed['view_id'])
        scope = view['scope']
        prompt = ('Check the specific concern against referenced sources. A concern is neither a user requirement nor a proven issue. Code comes from the trigger-time snapshot; Trace ends at the checkpoint. A match indicates relevance only. Disclose issues that affect task conclusions or subsequent decisions. Uncollected material may still exist, and static code does not prove execution.')
        notes = view['collection_notes'] + view['graph_notes']
        if notes:
            prompt += '\nCollection limits: ' + '；'.join(notes)
        payload = {
            'ebg_disclose_prompt': prompt,
            'requirements': {r['id']: r['check'] for r in view['requirements']},
            'evidence_groups': {r['id']: self.group(view, r) for r in view['requirements']},
            'repository': {
                'count': len(view['files']) + len(view.get('uncollected_files', {})),
                'content_count': len(view['files']), 'read_ref': f"{view['view_id']}:files:index",
                'note': 'The directory lists files observed within the collection scope. Uncollected files may still exist. Binary entries record existence only and do not prove executability. Excluded directories, Git ignores, and collection errors can leave gaps. changes compares collected text only.',
            },
        }
        share_repo_excerpts(payload['evidence_groups'])
        payload['check_id'] = scope['check_id']
        payload['questions'] = payload.pop('requirements')
        for group in payload['evidence_groups'].values():
            group['cited_sources'] = group.pop('requirement')
        check = self.checks.get(scope['check_id'])
        selected = set(check['selected_plans'])
        seeds = [(s.get('path', ''), s['content']) for s in view['sources'] if s['id'] in selected]
        anchors = set(check['anchors'])
        calls = {e.get('call_id') for e in check['events'] if e['id'] in anchors}
        seeds = [('', e['content']) for e in check['events'] if e['kind'] == 'tool_result'
                 and (e.get('call_id') in calls if anchors else e['turn_id'] == check['turn_id'])] + seeds
        seeds.extend(('', ref['content']) for r in view['requirements'] for ref in r['refs'])
        for group in payload['evidence_groups'].values():
            for entry in group.get('actual', {}).get('repo', []):
                path = entry['source'].split(':', 1)[-1].split('::', 1)[0]
                seeds.append((path if path in view['files'] else '', entry.get('content', '')))
        linked = linked_artifacts(self.store, view, seeds)
        if linked:
            payload['linked_artifacts'] = linked
        research = research_context(self.store, view)
        if research:
            payload['research_context'] = research
        if view.get('history_snapshots'):
            payload['history_context'] = {
                'note': 'Find intermediate code versions by turn. Snapshots establish turn-boundary state only; verify the executed version against calls and edits within that turn.',
                'read_ref': f"{view['view_id']}:history:index",
            }
        matched_trace = {
            event['source']
            for group in payload['evidence_groups'].values()
            for role in ('action', 'response')
            for event in group.get('actual', {}).get('trace', {}).get(role, [])
        }
        unmatched_trace = {}
        for event in view['events']:
            if event['kind'] not in {'assistant', 'tool_call', 'tool_result'}:
                continue
            source = f"{event['id']} {event['kind']} {event.get('tool_name', '')}".strip()
            if source not in matched_trace:
                role = 'response' if event['kind'] == 'assistant' else 'action'
                unmatched_trace.setdefault(role, []).append({
                    'source': source, 'turn': event['turn'], 'content': event['content'],
                    'read_ref': f"{view['view_id']}:trace:{event['id']}",
                })
        if unmatched_trace:
            payload['trace_context'] = {
                'note': 'These calls, results, and Agent responses belong to the selected task but have no direct requirement link. They provide context without asserting a requirement match.',
                **unmatched_trace,
            }
        if view['changes']:
            payload['changes'] = {'count': len(view['changes']), 'read_ref': f"{view['view_id']}:changes"}
        ref = self.pages.save(owner, payload)
        return self.pages.read(owner, ref)

    def refresh_task(self, task_id: str) -> dict[str, Any]:
        """Build from frozen material; persisted file fragments resume partial work."""
        task = self.store.task(task_id)
        scope = task['scope']
        self.sessions.current(scope['session_id'])
        root = Path(task['repo_path'])
        if scope.get('mode') != 'checkpoint':
            raise HarnessError('Evidence requires a checkpoint; create one with ebg_review.')
        files = task['current_files']
        selected_ids = set(scope['event_ids'])
        events = [e for e in self.sessions.events(scope['session_id']) if e['id'] in selected_ids]
        signature = {'graph_version': GRAPH_VERSION, 'history_version': 1,
                     'scope': scope, 'requirements': task['requirements']}
        previous = self.store.latest_view(task_id)
        if previous and previous['signature'] == signature:
            return {'view_id': previous['view_id'], 'files_built': 0, 'reused': True}
        graph, contexts, graph_notes, built = construct_graph(self.store, root, files)
        changes = {
            path: 'added' if path not in task['baseline'] else 'deleted' if path not in files else 'modified'
            for path in sorted(set(files) | set(task['baseline']))
            if files.get(path) != task['baseline'].get(path)
            and path not in task.get('uncollected_files', {})
            and path not in task.get('baseline_uncollected_files', {})
        }
        turns = {e['turn_id']: e['turn'] for e in events if e['kind'] == 'user'}
        snapshots = [
            {**{key: snap[key] for key in ('snapshot_id', 'turn_id', 'phase', 'files', 'notes')},
             'turn': turns[snap['turn_id']], 'uncollected_files': snap.get('uncollected_files', {})}
            for snap in task['history_snapshots']
            if snap['turn_id'] in turns
        ]
        data = {'signature': signature, 'scope': scope, 'files': files, 'baseline': task['baseline'],
                'history_snapshots': snapshots,
                'repo_graph': graph, 'contexts': contexts, 'trace_graph': build_trace(events), 'events': events,
                'requirements': task['requirements'], 'sources': task['sources'], 'changes': changes,
                'collection_notes': task['collection_notes'], 'graph_notes': graph_notes,
                'uncollected_files': task.get('uncollected_files', {})}
        return {'view_id': self.store.save_view(task_id, data), 'files_built': built, 'reused': False}

    def _requirement_originals(self, requirement: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, list] = {}
        for ref in requirement["refs"]:
            key = {'user': 'demand', 'trace': 'trace', 'repo': 'repo'}.get(ref['kind'], 'plan')
            result.setdefault(key, []).append({"source": ref["source"], "content": ref["content"]})
        return result

    def _repo_evidence(self, view: dict[str, Any], requirement: dict[str, Any]) -> list[dict[str, Any]]:
        reader = SourceReader(self.store)
        text = "\n".join(ref["content"] for ref in requirement["refs"])
        text += '\n' + requirement['check']  # Retrieval hint, never a new requirement.
        files = view["files"]
        paths = {path for path in files if literal_match(text, path, path=True)}
        anchors = source_line_paths(requirement, view['sources'], list(files))
        paths.update(anchors)
        graph = view["repo_graph"]
        match_graph = {**graph, "behaviors": [b for b in graph["behaviors"] if not paths or b["path"] in paths]}
        roots = direct_repo_roots(match_graph, text)
        selected = []
        for context in view["contexts"]:
            path, symbol = context["path"], context["symbol"]
            if symbol in roots.get(path, ()):
                selected.append({**context, "match": "path + symbol" if path in paths else "symbol / explicit code term"})
        # Concrete definitions without observable results (e.g. pass stubs) still matter.
        for context in view["contexts"]:
            if context["symbol"] == "<module>" or (paths and context["path"] not in paths):
                continue
            if (explicit_symbol_match(text, context["symbol"])
                    or (context['path'] in paths and literal_match(text, context['symbol']))):
                candidates = [c for c in view["contexts"] if c["symbol"] == context["symbol"] and (not paths or c["path"] in paths)]
                if len({c["path"] for c in candidates}) == 1:
                    selected.append({**context, "match": "path + symbol" if context["path"] in paths else "symbol"})
        selected.extend(value_contexts(view['contexts'], text, paths, read_source=reader.context))
        for path in sorted(paths):
            if not any(item["path"] == path for item in selected):
                content = self.store.file(files[path])["content"]
                lines = [1, max(1, len(content.splitlines()))]
                selected.append({"path": path, "symbol": "<file>", "lines": lines,
                                 "source_ref": {'file_id': files[path], 'lines': lines}, "match": "path (file context)"})
        direct = {(item["path"], item["symbol"]) for item in selected}
        evidence = {e["evidence_id"]: e for e in graph["evidence"]}
        for edge in graph["edges"]:
            left = edge["from"]["path"], edge["from"]["symbol"]
            right = edge["to"]["path"], edge["to"]["symbol"]
            if not (left in direct or right in direct):
                continue
            neighbor = right if left in direct else left
            positions = [f"{evidence[e]['locator']['path']}@{evidence[e]['locator']['line_start']}" for e in edge["evidence_ids"]]
            relation = f"{left[0]}::{left[1]} {edge['type']} {right[0]}::{right[1]} via {', '.join(positions)}"
            if neighbor[1] == "<module>":
                # A module endpoint can span an entire file. Keep the actual
                # edge-supported statements, not every unrelated definition.
                content = self.store.file(files[neighbor[0]])["content"]
                for evidence_id in edge["evidence_ids"]:
                    locator = evidence[evidence_id]["locator"]
                    if (locator["path"], locator["symbol"]) == neighbor:
                        first, last = module_statement_span(neighbor[0], content, locator["line_start"], locator["line_end"])
                        selected.append({"path": neighbor[0], "symbol": neighbor[1], "lines": [first, last],
                                         "source_ref": {'file_id': files[neighbor[0]], 'lines': [first, last]}, "context": relation})
                continue
            for context in view["contexts"]:
                if (context["path"], context["symbol"]) == neighbor:
                    selected.append({**context, "context": relation})
        selected.extend(enclosing_contexts(selected, view['contexts']))
        context_refs = {(c['path'], c['symbol'], tuple(c['lines'])): f"{view['view_id']}:context:{index}"
                        for index, c in enumerate(view['contexts'])}
        # Keep a containing excerpt once, while retaining every graph node's identity.
        excerpts = compact_contexts(selected)
        source_ids = {ref['source_id'] for ref in requirement['refs']}
        document = '\n'.join(source['content'] for source in view['sources'] if source['id'] in source_ids)
        document_refs = {}
        for ref in requirement['refs']:
            matches = direct_repo_roots(match_graph, ref['content'])
            values = value_contexts(view['contexts'], ref['content'], paths, read_source=reader.context)
            matched_values = {(c['path'], c['symbol']) for c in values}
            for node in direct:
                if (node[1] in matches.get(node[0], ()) or node in matched_values
                        or explicit_symbol_match(ref['content'], node[1])
                        or (node[1] == '<file>' and literal_match(ref['content'], node[0], path=True))):
                    document_refs.setdefault(node, []).append(ref['source'])
        component_list = components(direct, graph['edges'], document)
        result = []
        for component_number, component in enumerate(component_list, 1):
            relations = relations_by_node(component['edges'], evidence)
            for node in component['nodes']:
                contexts = [c for c in selected if (c['path'], c['symbol']) == node]
                containers = [c for c in excerpts if any(c['path'] == item['path']
                              and c['lines'][0] <= item['lines'][0] <= item['lines'][1] <= c['lines'][1]
                              for item in contexts)]
                for context in containers:
                    entry = self._repo_entry(view, context, context_refs, reader, anchors)
                    entry = {'component': f'C{component_number}', 'node': node_id(node),
                             'role': 'seed' if node in direct else 'neighbor',
                             'root': node_id(component['root']), **entry}
                    # The original source may include an enclosing scope. Name it
                    # through `source`, keeping the actual graph endpoint in `node`.
                    if node in document_refs:
                        entry['document_refs'] = document_refs[node]
                    elif node in direct:
                        entry['seed_basis'] = 'query hint / source-line path; not a direct document Symbol match'
                    if relations[node]:
                        entry['relations'] = relations[node]
                    result.append(entry)
        # Keep changes (including deletions) beside requirements naming that path.
        for path in view['changes']:
            if literal_match(text, path, path=True):
                diff = self.material(view['view_id'], f'diff:{path}')
                result.append({**diff, 'match': 'path / before-after diff',
                               'read_ref': f"{view['view_id']}:diff:{path}"})
        result = prioritize_repo_entries(result, requirement['check'], view['events'])
        shown = {}
        for entry in result:
            if 'node' not in entry:
                continue
            key = entry['read_ref']
            if key in shown:
                entry.pop('content')
                entry.pop('_source_ref', None)
                entry['content_ref'] = entry['read_ref']
                entry['included_in'] = shown[key]
            else:
                shown[key] = entry['node']
        return result

    def _repo_entry(self, view, context, context_refs, reader, anchors):
        first, last = context["lines"]
        key = context['path'], context['symbol'], tuple(context['lines'])
        entry = {
            "source": f"{view['scope']['repo_after']}:{context['path']}::{context['symbol']}@{first}-{last}",
            "content": reader.context(context),
            "_source_ref": context.get('source_ref', {'file_id': view['files'][context['path']], 'lines': [first, last]}),
            "read_ref": context_refs.get(key, f"{view['view_id']}:slice:{first}:{last}:{context['path']}"),
        }
        entry.update({key: context[key] for key in ("match", "context") if key in context})
        if context['path'] in anchors:
            entry['anchor'] = anchors[context['path']]
        if context["path"] in view["changes"]:
            entry["change"] = view["changes"][context['path']]
        return entry

    def group(self, view: dict[str, Any], requirement: dict[str, Any]) -> dict[str, Any]:
        group: dict[str, Any] = {"requirement": self._requirement_originals(requirement)}
        repo = self._repo_evidence(view, requirement)
        events = matched_events(requirement, view["events"], list(set(view["files"]) | set(view["baseline"])),
                                view["trace_graph"].get("signals"))
        actual = {}
        if repo:
            actual["repo"] = repo
        else:
            candidates = section_navigation(requirement, view['sources'], list(view['files']))
            group['navigation'] = {
                'note': 'This is a lexical match between a filename and a referenced plan section, not implementation evidence. Read the references to verify it.',
                'candidates': [{**item, 'read_ref': f"{view['view_id']}:outline:{item['path']}"}
                               for item in candidates[:8]],
                'repository_ref': f"{view['view_id']}:files:index",
            }
            if len(candidates) > 8:
                group['navigation']['total_candidates'] = len(candidates)
        trace: dict[str, Any] = {}
        for event in events:
            key = "response" if event["kind"] == "assistant" else "action"
            trace.setdefault(key, []).append({
                "source": f"{event['id']} {event['kind']} {event.get('tool_name', '')}".strip(),
                "content": event["content"],
            })
        if trace:
            trace["match"] = "demand/action: operation + object; response: object"
            actual["trace"] = trace
        if actual:
            group["actual"] = actual
        notes = []
        if not actual:
            notes.append("No relevant evidence matched the registered Repo/Trace for this view. This does not prove missing implementation or execution.")
        completed = {event["call_id"] for event in events if event["kind"] == "tool_result"}
        pending = [event["id"] for event in events if event["kind"] == "tool_call" and event["call_id"] not in completed]
        if pending:
            notes.append(f"Tool call {', '.join(pending)} has no result in this view; completion is unconfirmed.")
        text = "\n".join(ref["content"] for ref in requirement["refs"])
        for path, change in view["changes"].items():
            if change == "deleted" and literal_match(text, path, path=True):
                notes.append(f"File {path} exists in the starting snapshot but was deleted by the ending snapshot; see Repo evidence for the diff.")
        if notes:
            group["note"] = " ".join(notes)
        return group

    def material(self, view_id: str, read_id: str) -> dict[str, Any] | list[dict[str, Any]]:
        view = self.store.view(view_id)
        kind, separator, identifier = read_id.partition(":")
        if not separator:
            raise HarnessError("Use a returned material reference.")
        if kind == 'history' and identifier == 'index':
            snapshots = view.get('history_snapshots', [])
            if not snapshots:
                raise HarnessError('This view has no recorded historical snapshots.')
            turns = {}
            for snap in snapshots:
                entry = turns.setdefault(snap['turn'], {'prompt': f"P{snap['turn']}",
                    'events': [f"{view_id}:trace:{e['id']}" for e in view['events']
                               if e['turn'] == snap['turn'] and e['kind'] != 'user']})
                entry[snap['phase']] = {'snapshot': snap['snapshot_id'],
                    'read_ref': f"{view_id}:snapshot:{snap['snapshot_id']}"}
            return list(turns.values())
        if kind in {'snapshot', 'snapshot_file'}:
            snapshot_id, _, path = identifier.partition(':')
            snap = next((s for s in view.get('history_snapshots', [])
                         if s['snapshot_id'] == snapshot_id), None)
            if snap is None:
                raise HarnessError('Snapshot is outside this frozen task; use a returned reference.')
            uncollected = snap.get('uncollected_files', {})
            if kind == 'snapshot':
                return {'source': snapshot_id, 'prompt': f"P{snap['turn']}", 'phase': snap['phase'],
                        'notes': snap['notes'], 'files': [
                            {'path': name, **uncollected[name]} if name in uncollected else
                            {'path': name, 'read_ref': f'{view_id}:snapshot_file:{snapshot_id}:{name}'}
                            for name in sorted(set(snap['files']) | set(uncollected))]}
            if path not in snap['files']:
                raise HarnessError('File content was not collected in this snapshot.')
            content = self.store.file(snap['files'][path])['content']
            return {'source': f'{snapshot_id}:{path}',
                    'content': content,
                    '_source_ref': {'file_id': snap['files'][path], 'lines': [1, max(1, len(content.splitlines()))]}}
        if kind == 'files' and identifier == 'index':
            uncollected = view.get('uncollected_files', {})
            return [
                {'path': path, **uncollected[path]} if path in uncollected else
                {'path': path, 'exists': True, 'content_status': 'collected', 'read_ref': f'{view_id}:outline:{path}'}
                for path in sorted(set(view['files']) | set(uncollected))
            ]
        if kind == 'outline':
            if identifier not in view['files']:
                raise HarnessError(f'File not collected in {view_id}: {identifier}')
            return {'source': f'{view_id}:repo:{identifier}',
                    'file_ref': f'{view_id}:repo:{identifier}',
                    'symbols': [{'symbol': context['symbol'], 'lines': context['lines'],
                                 'read_ref': f'{view_id}:context:{index}'}
                                for index, context in enumerate(view['contexts']) if context['path'] == identifier]}
        if kind == 'context':
            if not identifier.isdecimal() or int(identifier) >= len(view['contexts']):
                raise HarnessError('Unknown source context; use a returned reference.')
            context = view['contexts'][int(identifier)]
            first, last = context['lines']
            return {'source': f"{view['scope']['repo_after']}:{context['path']}::{context['symbol']}@{first}-{last}",
                    'content': SourceReader(self.store).context(context),
                    '_source_ref': context.get('source_ref', {'file_id': view['files'][context['path']], 'lines': [first, last]}),
                    'file_ref': f"{view_id}:repo:{context['path']}"}
        if kind == 'slice':
            parts = identifier.split(':', 2)
            if len(parts) != 3:
                raise HarnessError('Unknown source slice; use a returned reference.')
            first, last, path = parts
            if path not in view['files'] or not first.isdecimal() or not last.isdecimal() or not 1 <= int(first) <= int(last):
                raise HarnessError('Unknown source slice; use a returned reference.')
            ref = {'file_id': view['files'][path], 'lines': [int(first), int(last)]}
            return {'source': f"{view['scope']['repo_after']}:{path}@{first}-{last}",
                    'content': SourceReader(self.store).read(ref), '_source_ref': ref}
        if kind in {"repo", "diff"}:
            file_id = view["files"].get(identifier)
            old_id = view["baseline"].get(identifier)
            if file_id is None and (kind != "diff" or old_id is None):
                raise HarnessError(f"File not collected in {view_id}: {identifier}")
            content = self.store.file(file_id)["content"] if file_id else ""
            if kind == "diff":
                before = self.store.file(old_id)["content"] if old_id else ""
                diff = difflib.unified_diff(before.splitlines(keepends=True), content.splitlines(keepends=True),
                                            fromfile=f"{view['scope']['repo_before']}/{identifier}",
                                            tofile=f"{view['scope']['repo_after']}/{identifier}")
                content = ''.join(line if line.endswith('\n') else line + '\n\\ No newline at end of file\n'
                                  for line in diff)
        elif kind == "trace":
            event = next((e for e in view["events"] if e["id"] == identifier), None)
            if not event:
                raise HarnessError(f"Unknown Trace event in {view_id}: {identifier}")
            content = event["content"]
        elif kind == "source":
            source = next((s for s in view["sources"] if s["id"] == identifier), None)
            if not source:
                raise HarnessError(f"Unknown requirement source: {identifier}")
            content = source["content"]
        else:
            raise HarnessError(f"Unknown read kind: {kind}")
        result = {"source": f"{view_id}:{read_id}", "content": content}
        if kind == 'repo':
            result['_source_ref'] = {'file_id': file_id, 'lines': [1, max(1, len(content.splitlines()))]}
        return result
