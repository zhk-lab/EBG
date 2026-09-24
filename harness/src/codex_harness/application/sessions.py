"""Session recording and historical boundaries, independent of disclosure tasks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .repository import capture
from .storage import HarnessError, Store, dumps


class SessionLog:
    def __init__(self, store: Store, max_file_bytes: int) -> None:
        self.store = store
        self.max_file_bytes = max_file_bytes

    def current(self, session_id: str | None = None) -> str:
        active = self.store.current_session()
        if not active or (session_id is not None and session_id != active):
            raise HarnessError("Only the current recorded session is available. Enable hooks before the task.")
        self.root(active)
        return active

    def snapshots(self, session_id: str) -> list[dict[str, Any]]:
        with self.store.connect() as db:
            rows = db.execute("SELECT * FROM snapshots WHERE session_id=? ORDER BY id", (session_id,)).fetchall()
        return [{**dict(r), **json.loads(r['data']), 'snapshot_id': f"S{r['id']}"} for r in rows]

    def prompts(self, session_id: str) -> list[dict[str, Any]]:
        after_turns = {s['turn_id'] for s in self.snapshots(session_id) if s['phase'] == 'after'}
        return [{'id': f"P{e['turn']}", 'content': e['content'], 'complete': e['turn_id'] in after_turns}
                for e in self.events(session_id) if e['kind'] == 'user' and not e.get('internal')]

    def root(self, session_id: str) -> Path:
        with self.store.connect() as db:
            row = db.execute("SELECT repo_path FROM recordings WHERE session_id=?", (session_id,)).fetchone()
        if row is None:
            raise HarnessError("Session not recorded; enable hooks before executing the task.")
        return Path(row[0])

    def turn(self, session_id: str, turn_id: str | None = None) -> dict[str, Any] | None:
        with self.store.connect() as db:
            if turn_id is None:
                row = db.execute("SELECT * FROM recorded_turns WHERE session_id=? ORDER BY number DESC LIMIT 1", (session_id,)).fetchone()
            else:
                row = db.execute("SELECT * FROM recorded_turns WHERE session_id=? AND turn_id=?", (session_id, turn_id)).fetchone()
        return dict(row) if row else None

    def events(self, session_id: str) -> list[dict[str, Any]]:
        with self.store.connect() as db:
            rows = db.execute("SELECT * FROM recorded_events WHERE session_id=? ORDER BY seq", (session_id,)).fetchall()
        return [{**json.loads(r['data']), 'id': f"s{r['seq']}", 'seq': r['seq'], 'turn_id': r['turn_id']} for r in rows]

    def snapshot(self, session_id: str, snapshot_id: str) -> dict[str, Any]:
        if not snapshot_id.startswith("S") or not snapshot_id[1:].isdigit():
            raise HarnessError("Use an existing snapshot ID such as S1.")
        with self.store.connect() as db:
            row = db.execute("SELECT * FROM snapshots WHERE id=? AND session_id=?", (int(snapshot_id[1:]), session_id)).fetchone()
        if row is None:
            raise HarnessError(f"Snapshot {snapshot_id} does not belong to this session.")
        return {**dict(row), **json.loads(row['data']), 'snapshot_id': snapshot_id}

    def _snapshot_for(self, session_id: str, turn_id: str, phase: str) -> str | None:
        with self.store.connect() as db:
            row = db.execute("SELECT id FROM snapshots WHERE session_id=? AND turn_id=? AND phase=?", (session_id, turn_id, phase)).fetchone()
        return f"S{row[0]}" if row else None

    def _cutoff(self, session_id: str) -> int:
        with self.store.connect() as db:
            return db.execute("SELECT COALESCE(MAX(seq),0) FROM recorded_events WHERE session_id=?", (session_id,)).fetchone()[0]

    def start(self, session_id: str, turn_id: str, repo_path: str, prompt: str, *, internal: bool = False) -> dict[str, Any]:
        self.store.activate_session(session_id)
        root = Path(repo_path).resolve()
        existing = self.turn(session_id, turn_id)
        if existing:
            event = next(e for e in self.events(session_id) if e['seq'] == existing['prompt_seq'])
            if event['content'] != prompt or root != self.root(session_id):
                raise HarnessError("Prompt replay differs from its recorded turn.")
            return existing
        observed_cutoff = self._cutoff(session_id)
        files, notes, uncollected = capture(self.store, root, max_file_bytes=self.max_file_bytes, session_id=session_id)
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            active = db.execute("SELECT id FROM active_session WHERE singleton=1").fetchone()
            if not active or active[0] != session_id:
                raise HarnessError("Session switched during capture; discard this old turn.")
            known = db.execute("SELECT repo_path FROM recordings WHERE session_id=?", (session_id,)).fetchone()
            if known and Path(known[0]) != root:
                raise HarnessError("Repository changed within a recorded session; use a separate session.")
            if db.execute("SELECT 1 FROM recorded_turns WHERE session_id=? AND turn_id=?", (session_id, turn_id)).fetchone():
                raise HarnessError("Turn was recorded concurrently; retry the hook.")
            db.execute("INSERT OR IGNORE INTO recordings VALUES (?,?)", (session_id, str(root)))
            number = db.execute("SELECT COALESCE(MAX(number),0)+1 FROM recorded_turns WHERE session_id=?", (session_id,)).fetchone()[0]
            cutoff = db.execute("SELECT COALESCE(MAX(seq),0) FROM recorded_events WHERE session_id=?", (session_id,)).fetchone()[0]
            if cutoff != observed_cutoff:
                raise HarnessError("Events changed during before capture; retry the prompt hook.")
            seq = db.execute("INSERT INTO recorded_events(session_id,turn_id,event_key,data) VALUES (?,?,?,?)",
                             (session_id, turn_id, 'prompt', dumps({'kind': 'user', 'turn': number, 'content': prompt,
                                                                  **({'internal': True} if internal else {})}))).lastrowid
            db.execute("INSERT INTO recorded_turns VALUES (?,?,?,?)", (session_id, turn_id, number, seq))
            db.execute("INSERT INTO snapshots(session_id,turn_id,phase,event_cutoff,data) VALUES (?,?,?,?,?)",
                       (session_id, turn_id, 'before', cutoff,
                        dumps({'files': files, 'notes': notes, 'uncollected_files': uncollected})))
        return self.turn(session_id, turn_id)

    def append(self, session_id: str, turn_id: str, event: dict[str, Any], key: str) -> None:
        turn = self.turn(session_id, turn_id)
        if turn is None:
            raise HarnessError("Tool event has no recorded prompt turn.")
        event = {**event, 'turn': turn['number']}
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT data FROM recorded_events WHERE session_id=? AND turn_id=? AND event_key=?", (session_id, turn_id, key)).fetchone()
            if row:
                if json.loads(row[0]) != event:
                    raise HarnessError("Event replay has different content.")
                return
            if db.execute("SELECT 1 FROM snapshots WHERE session_id=? AND turn_id=? AND phase='after'", (session_id, turn_id)).fetchone():
                raise HarnessError("Late event arrived after the turn snapshot; its old cutoff cannot be extended.")
            db.execute("INSERT INTO recorded_events(session_id,turn_id,event_key,data) VALUES (?,?,?,?)", (session_id, turn_id, key, dumps(event)))

    def stop(self, session_id: str, turn_id: str, response: str | None) -> None:
        if self._snapshot_for(session_id, turn_id, 'after'):
            recorded = [e['content'] for e in self.events(session_id) if e['turn_id'] == turn_id and e['kind'] == 'assistant']
            if response and recorded != [response]:
                raise HarnessError("Stop replay has different response content.")
            return
        latest = self.turn(session_id)
        if not latest or latest['turn_id'] != turn_id:
            raise HarnessError("Cannot capture an old turn after a newer prompt; historical after snapshot is missing.")
        observed_cutoff = self._cutoff(session_id)
        files, notes, uncollected = capture(self.store, self.root(session_id), max_file_bytes=self.max_file_bytes, session_id=session_id)
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            newest = db.execute("SELECT turn_id FROM recorded_turns WHERE session_id=? ORDER BY number DESC LIMIT 1", (session_id,)).fetchone()[0]
            if newest != turn_id:
                raise HarnessError("A newer prompt arrived during capture; cannot label this as the old turn's snapshot.")
            cutoff = db.execute("SELECT MAX(seq) FROM recorded_events WHERE session_id=?", (session_id,)).fetchone()[0]
            if cutoff != observed_cutoff:
                raise HarnessError("Events changed during after capture; retry the Stop hook.")
            if db.execute("SELECT 1 FROM snapshots WHERE session_id=? AND turn_id=? AND phase='after'", (session_id, turn_id)).fetchone():
                raise HarnessError("Turn stopped concurrently; retry the hook.")
            if response:
                db.execute("INSERT INTO recorded_events(session_id,turn_id,event_key,data) VALUES (?,?,?,?)",
                           (session_id, turn_id, 'response', dumps({'kind': 'assistant', 'turn': latest['number'], 'content': response})))
            cutoff = db.execute("SELECT MAX(seq) FROM recorded_events WHERE session_id=?", (session_id,)).fetchone()[0]
            db.execute("INSERT INTO snapshots(session_id,turn_id,phase,event_cutoff,data) VALUES (?,?,?,?,?)",
                       (session_id, turn_id, 'after', cutoff,
                        dumps({'files': files, 'notes': notes, 'uncollected_files': uncollected})))

    def timeline(self, session_id: str, offset: int = 0, limit: int = 20,
                 prompt_id: str | None = None) -> dict[str, Any]:
        self.root(session_id)
        if offset < 0 or not 1 <= limit <= 50:
            raise HarnessError("Use offset >= 0 and limit between 1 and 50.")
        events = self.events(session_id)
        if prompt_id:
            prompt = next((e for e in events if e['id'] == prompt_id and e['kind'] == 'user'), None)
            if prompt is None:
                raise HarnessError("Unknown prompt ID in this session.")
            selected = [e for e in events if e['turn_id'] == prompt['turn_id']]
            items = [{**{k: v for k, v in e.items() if k != 'content'}, 'preview': e['content'][:400]}
                     for e in selected[offset:offset + limit]]
            result = {'session_id': session_id, 'prompt_id': prompt_id, 'events': items, 'total': len(selected)}
            if offset + limit < len(selected):
                result['next_offset'] = offset + limit
            return result
        with self.store.connect() as db:
            turns = db.execute("SELECT * FROM recorded_turns WHERE session_id=? ORDER BY number", (session_id,)).fetchall()
        items = []
        for turn in turns[offset:offset + limit]:
            local = [e for e in events if e['turn_id'] == turn['turn_id']]
            item = {'prompt_id': f"s{turn['prompt_seq']}", 'turn': turn['number'],
                    'prompt_preview': local[0]['content'][:400], 'last_event_id': local[-1]['id'],
                    'event_count': len(local), 'before': self._snapshot_for(session_id, turn['turn_id'], 'before')}
            after = self._snapshot_for(session_id, turn['turn_id'], 'after')
            if after:
                item['after'] = after
            else:
                item['note'] = 'No ending snapshot was recorded for this turn; do not assume a final task state.'
            items.append(item)
        result = {'session_id': session_id, 'turns': items, 'total': len(turns)}
        if offset + limit < len(turns):
            result['next_offset'] = offset + limit
        return result

    def material(self, session_id: str, event_id: str | None = None, snapshot_id: str | None = None,
                 path: str | None = None, offset: int = 0, limit: int = 4000) -> dict[str, Any]:
        if offset < 0 or not 1 <= limit <= 8000:
            raise HarnessError("Use offset >= 0 and limit between 1 and 8000.")
        self.root(session_id)
        if event_id:
            if snapshot_id or path:
                raise HarnessError("Read an event or a snapshot, not both.")
            event = next((e for e in self.events(session_id) if e['id'] == event_id), None)
            if event is None:
                raise HarnessError("Event not found in this session.")
            content = event['content']
            result = {k: v for k, v in event.items() if k != 'content'}
        else:
            if not snapshot_id:
                raise HarnessError("Provide event_id or snapshot_id.")
            snap = self.snapshot(session_id, snapshot_id)
            uncollected = snap.get('uncollected_files', {})
            if path is None:
                names = sorted(set(snap['files']) | set(uncollected))
                page = names[offset:offset + min(limit, 100)]
                result = {'snapshot_id': snapshot_id, 'files': page, 'total': len(names), 'notes': snap['notes'],
                          'uncollected_files': {name: uncollected[name] for name in page if name in uncollected}}
                if offset + min(limit, 100) < len(names):
                    result['next_offset'] = offset + min(limit, 100)
                return result
            if path in uncollected:
                return {'snapshot_id': snapshot_id, 'path': path, **uncollected[path]}
            if path not in snap['files']:
                raise HarnessError("File was not collected in this snapshot.")
            content = self.store.file(snap['files'][path])['content']
            result = {'snapshot_id': snapshot_id, 'path': path}
        result.update(content=content[offset:offset + limit], total_chars=len(content), offset=offset)
        if offset + limit < len(content):
            result['next_offset'] = offset + limit
        return result
