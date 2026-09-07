"""Application service shared by MCP, lifecycle hooks and offline use."""

from __future__ import annotations

import difflib
import json
from pathlib import Path
from typing import Any

from .matching import direct_repo_roots, explicit_symbol_match, literal_match, section_navigation, source_line_paths
from .checks import Checks
from .render import PROMPT
from .paging import OutputPages
from .repository import GRAPH_VERSION, capture, construct_graph, module_statement_span
from .artifacts import linked_artifacts
from .requirements import normalize_requirements
from .research import research_context
from .storage import HarnessError, Store
from .sessions import SessionLog
from .scoping import select_plan_task, select_task as make_scoped_task
from .selection import compact_contexts, enclosing_contexts, share_repo_excerpts, value_contexts
from .trace import build_trace, matched_events


class Harness:
    def __init__(self, state_dir: str | Path, *, token_budget: int = 12_000, max_file_bytes: int = 2_000_000) -> None:
        if token_budget < 256 or max_file_bytes < 1:
            raise HarnessError("token_budget must be >=256 and max_file_bytes positive.")
        self.store = Store(state_dir)
        self.token_budget = token_budget
        self.max_file_bytes = max_file_bytes
        self.sessions = SessionLog(self.store, max_file_bytes)
        self.checks = Checks(self)

    @property
    def pages(self) -> OutputPages:
        return OutputPages(self.store, self.token_budget)

    def list_task_sources(self, session_id: str | None = None, *, read_ref: str | None = None,
                          offset: int = 0, repo_path: str | None = None) -> str:
        if read_ref and repo_path is not None:
            raise HarnessError('Read a saved listing or collect current Plans, not both.')
        session_id = (self.sessions.inspection_context(repo_path, session_id) if repo_path is not None
                      else self.sessions.current(session_id))
        owner = f"session:{session_id}"
        if not read_ref:
            if offset:
                raise HarnessError("Use the read_ref returned with next to continue a listing.")
            plans = [{k: v for k, v in p.items() if k not in {'file_id', 'snapshot_id'}}
                     for p in self.sessions.plans(session_id)]
            notes = []
            if repo_path is not None:
                files, notes, _ = capture(self.store, self.sessions.root(session_id),
                                       max_file_bytes=self.max_file_bytes, session_id=session_id)
                current = [{'id': f'L{file_id}', 'path': path, 'version': 'current'}
                           for path, file_id in files.items() if Path(path).suffix.lower() in {'.md', '.markdown'}]
                ids = {p['id'] for p in current}
                plans = [p for p in plans if p['id'] not in ids] + current
            read_ref = self.pages.save(owner, {
                'session_id': session_id, 'prompts': self.sessions.prompts(session_id), 'plans': plans,
                'note': 'Plan 列表为已保存的 Markdown 候选版本，请选择相关项；来源身份由原文确认。',
                **({'collection_notes': notes} if notes else {}),
            })
        return self.pages.read(owner, read_ref, offset)

    def select_task(self, start_prompt: str | None = None, end_prompt: str | None = None,
                    plan_ids: list[str] | None = None, *, session_id: str | None = None,
                    read_ref: str | None = None, offset: int = 0, repo_path: str | None = None) -> str:
        if not read_ref:
            if offset or bool(start_prompt) != bool(end_prompt):
                raise HarnessError('Supply both start_prompt/end_prompt, or omit both for Plan-only inspection.')
            if not start_prompt and not plan_ids:
                raise HarnessError('Prompt and Plan cannot both be empty; select a Prompt range or at least one Plan.')
        if repo_path is not None and (read_ref or start_prompt or end_prompt):
            raise HarnessError('repo_path is only for current-code inspection, not historical selection or saved reads.')
        session_id = (self.sessions.inspection_context(repo_path, session_id) if not read_ref and not start_prompt
                      else self.sessions.current(session_id))
        owner = f"selection:{session_id}"
        if read_ref:
            if start_prompt or end_prompt or plan_ids:
                raise HarnessError("Read a saved selection or select a new interval, not both.")
            return self.pages.read(owner, read_ref, offset)
        task = (make_scoped_task(self.sessions, session_id, start_prompt, end_prompt, plan_ids or [])
                if start_prompt else select_plan_task(self.sessions, session_id, plan_ids or []))
        if not any(s['content'].strip() for s in task['sources']):
            raise HarnessError('Prompt and Plan cannot both be empty.')
        # Historical retries reuse their task/checklist; current-code selections capture anew.
        with self.store.connect() as db:
            existing = [json.loads(r[0]) for r in db.execute("SELECT data FROM tasks")]
        previous = next((t for t in existing if t.get('scope') == task['scope']), None)
        if previous:
            task = previous
        else:
            self.store.put_task(task, new=True)
        result = {'task_id': task['id'],
                  'scope': {k: v for k, v in task['scope'].items() if k not in {'event_ids', 'session_id'}},
                  'sources': task['sources']}
        ref = self.pages.save(owner, result)
        return self.pages.read(owner, ref)

    def build_evidence_groups(self, task_id: str, requirements: list[dict[str, Any]] | None = None,
                              *, read_ref: str | None = None, offset: int = 0) -> str:
        task = self.store.task(task_id)
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
                    if view['scope'].get('mode') == 'current':
                        raise HarnessError('Current-code inspection has no historical baseline or change list.')
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
        if scope.get('mode') == 'current':
            prompt = PROMPT + '\n本次按选定 Plan 核对选择时保存的当前 Repo，不自动关联历史 Trace。已有初始材料、实验记录和汇报可按其来源身份核对；历史材料缺失不自动构成任务违规，不逐条重复未运行或未完成的疑问。'
        elif scope.get('mode') == 'checkpoint':
            prompt = ('根据引用原文核对具体疑问；疑问不是用户要求，也不是已证实的问题。'
                      '代码来自触发时保存的快照，Trace 截至该检查点；匹配仅表示相关。'
                      '只有影响任务结论或后续决策的问题才需要披露。未采集不等于不存在，静态代码不证明执行。')
        else:
            prompt = (PROMPT + f"\n任务范围：{scope['start_prompt']}～{scope['end_prompt']}；"
                      f"Repo：{scope['repo_before']} → {scope['repo_after']}。")
        notes = view['collection_notes'] + view['graph_notes']
        if notes:
            prompt += '\n采集限制：' + '；'.join(notes)
        payload = {
            'beg_disclose_prompt': prompt,
            'requirements': {r['id']: r['check'] for r in view['requirements']},
            'evidence_groups': {r['id']: self.group(view, r) for r in view['requirements']},
            'repository': {
                'count': len(view['files']) + len(view.get('uncollected_files', {})),
                'content_count': len(view['files']), 'read_ref': f"{view['view_id']}:files:index",
                'note': '目录记录采集范围内观察到的文件；未采集不等于不存在。二进制等仅保留存在信息，不能据此确认可运行。忽略目录、Git 忽略项和采集错误可能使目录不完整；changes 只比较已采集文本，不包含未读取内容的文件。',
            },
        }
        share_repo_excerpts(payload['evidence_groups'])
        if scope.get('mode') == 'checkpoint':
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
                'note': '按回合查找中间代码版本；快照只证明回合边界状态，具体执行版本须结合该回合调用与改动核对。',
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
                'note': '以下是所选任务中未直接关联到要求的调用、结果和 Agent 回复；保留上下文供核对，不表示已对应某条要求。',
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
        current = scope.get('mode') == 'current'
        checkpoint = scope.get('mode') == 'checkpoint'
        files = task['current_files'] if current or checkpoint else self.sessions.snapshot(scope['session_id'], scope['repo_after'])['files']
        selected_ids = set(scope['event_ids'])
        events = [e for e in self.sessions.events(scope['session_id']) if e['id'] in selected_ids]
        signature = {'graph_version': GRAPH_VERSION, 'history_version': 1,
                     'scope': scope, 'requirements': task['requirements']}
        previous = self.store.latest_view(task_id)
        if previous and previous['signature'] == signature:
            return {'view_id': previous['view_id'], 'files_built': 0, 'reused': True}
        graph, contexts, graph_notes, built = construct_graph(self.store, root, files)
        changes = {} if current else {
            path: 'added' if path not in task['baseline'] else 'deleted' if path not in files else 'modified'
            for path in sorted(set(files) | set(task['baseline']))
            if files.get(path) != task['baseline'].get(path)
            and path not in task.get('uncollected_files', {})
            and path not in task.get('baseline_uncollected_files', {})
        }
        turns = {e['turn_id']: e['turn'] for e in events if e['kind'] == 'user'}
        snapshots = [] if current else [
            {**{key: snap[key] for key in ('snapshot_id', 'turn_id', 'phase', 'files', 'notes')},
             'turn': turns[snap['turn_id']], 'uncollected_files': snap.get('uncollected_files', {})}
            for snap in (task['history_snapshots'] if checkpoint else self.sessions.snapshots(scope['session_id']))
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
        text = "\n".join(ref["content"] for ref in requirement["refs"])
        if view['scope'].get('mode') == 'checkpoint':
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
        selected.extend(value_contexts(view['contexts'], text, paths))
        for path in sorted(paths):
            if not any(item["path"] == path for item in selected):
                content = self.store.file(files[path])["content"]
                selected.append({"path": path, "symbol": "<file>", "lines": [1, max(1, len(content.splitlines()))], "source": content, "match": "path (file context)"})
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
                lines = content.splitlines(keepends=True)
                for evidence_id in edge["evidence_ids"]:
                    locator = evidence[evidence_id]["locator"]
                    if (locator["path"], locator["symbol"]) == neighbor:
                        first, last = module_statement_span(neighbor[0], content, locator["line_start"], locator["line_end"])
                        selected.append({"path": neighbor[0], "symbol": neighbor[1], "lines": [first, last],
                                         "source": "".join(lines[first - 1:last]), "context": relation})
                continue
            for context in view["contexts"]:
                if (context["path"], context["symbol"]) == neighbor:
                    selected.append({**context, "context": relation})
        selected.extend(enclosing_contexts(selected, view['contexts']))
        context_refs = {(c['path'], c['symbol'], tuple(c['lines'])): f"{view['view_id']}:context:{index}"
                        for index, c in enumerate(view['contexts'])}
        result = []
        for context in compact_contexts(selected):
            first, last = context["lines"]
            key = context['path'], context['symbol'], tuple(context['lines'])
            entry = {
                "source": f"{view['scope']['repo_after']}:{context['path']}::{context['symbol']}@{first}-{last}",
                "content": context["source"],
                "read_ref": context_refs.get(key, f"{view['view_id']}:repo:{context['path']}"),
            }
            entry.update({key: context[key] for key in ("match", "context") if key in context})
            if context['path'] in anchors:
                entry['anchor'] = anchors[context['path']]
            if context["path"] in view["changes"]:
                entry["change"] = view["changes"][context["path"]]
            result.append(entry)
        # Keep changes (including deletions) beside requirements naming that path.
        for path in view['changes']:
            if literal_match(text, path, path=True):
                diff = self.material(view['view_id'], f'diff:{path}')
                result.append({**diff, 'match': 'path / before-after diff',
                               'read_ref': f"{view['view_id']}:diff:{path}"})
        return result

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
                'note': '仅为文件名与所引 Plan 章节的词面匹配，不是实现证据；按引用读取核对。',
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
            notes.append("本视图已登记的 Repo/Trace 中未匹配到相关证据；不等于未实现或未执行。")
        completed = {event["call_id"] for event in events if event["kind"] == "tool_result"}
        pending = [event["id"] for event in events if event["kind"] == "tool_call" and event["call_id"] not in completed]
        if pending:
            notes.append(f"工具调用 {', '.join(pending)} 在本视图尚无结果，不能确认执行完成。")
        text = "\n".join(ref["content"] for ref in requirement["refs"])
        for path, change in view["changes"].items():
            if change == "deleted" and literal_match(text, path, path=True):
                notes.append(f"起点快照中的文件 {path} 在终点快照中被删除；相关差异见 Repo 证据。")
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
            return {'source': f'{snapshot_id}:{path}',
                    'content': self.store.file(snap['files'][path])['content']}
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
                    'content': context['source'], 'file_ref': f"{view_id}:repo:{context['path']}"}
        if kind in {"repo", "diff"}:
            file_id = view["files"].get(identifier)
            old_id = view["baseline"].get(identifier)
            if file_id is None and (kind != "diff" or old_id is None):
                raise HarnessError(f"File not collected in {view_id}: {identifier}")
            content = self.store.file(file_id)["content"] if file_id else ""
            if kind == "diff":
                if view['scope'].get('mode') == 'current':
                    raise HarnessError('Current-code inspection has no historical baseline or diff.')
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
        return {"source": f"{view_id}:{read_id}", "content": content}
