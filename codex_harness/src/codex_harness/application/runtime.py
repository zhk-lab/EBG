"""Small persistent session state shared by hooks and review tools."""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from typing import Any, Iterator

from .storage import Store, dumps


@contextmanager
def runtime(store: Store, session: str) -> Iterator[dict[str, Any]]:
    with store.connect() as db:
        db.execute('BEGIN IMMEDIATE')
        store.require_session(db, session)
        row = db.execute('SELECT data FROM review_runtime WHERE session_id=?', (session,)).fetchone()
        state = json.loads(row[0]) if row else {
            'calls': [], 'elapsed': 0.0, 'active_since': None, 'failed': False,
            'plan': None, 'plan_seen': False, 'plan_checks': {}, 'waiting': None,
            'batch': None, 'prompts': [], 'stops': [],
            'observations': [],
        }
        yield state
        db.execute('INSERT OR REPLACE INTO review_runtime VALUES (?, ?)', (session, dumps(state)))


def pause(state: dict[str, Any]) -> None:
    if state['active_since'] is not None:
        state['elapsed'] += max(0, time.time() - state['active_since'])
        state['active_since'] = None


def elapsed(state: dict[str, Any]) -> float:
    return state['elapsed'] + (max(0, time.time() - state['active_since'])
                               if state['active_since'] is not None else 0)
