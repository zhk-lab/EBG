"""Transactional task state and reusable, content-compared file checkpoints."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


class HarnessError(ValueError):
    """An actionable input or evidence-availability error."""


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


class Store:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "harness.sqlite3"
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY, data TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS checkpoints (
                    id TEXT PRIMARY KEY, data TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS active_session (
                    singleton INTEGER PRIMARY KEY CHECK(singleton=1), id TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS outputs (
                    id TEXT PRIMARY KEY, owner TEXT NOT NULL, data TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    root TEXT NOT NULL, path TEXT NOT NULL,
                    content TEXT NOT NULL, fragment TEXT
                );
                CREATE INDEX IF NOT EXISTS files_lookup ON files(root, path);
                CREATE TABLE IF NOT EXISTS views (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL, data TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS recordings (
                    session_id TEXT PRIMARY KEY, repo_path TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS recorded_turns (
                    session_id TEXT NOT NULL, turn_id TEXT NOT NULL,
                    number INTEGER NOT NULL, prompt_seq INTEGER NOT NULL,
                    PRIMARY KEY(session_id, turn_id)
                );
                CREATE TABLE IF NOT EXISTS recorded_events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL, turn_id TEXT NOT NULL,
                    event_key TEXT NOT NULL, data TEXT NOT NULL,
                    UNIQUE(session_id, turn_id, event_key)
                );
                CREATE INDEX IF NOT EXISTS recorded_events_session ON recorded_events(session_id, seq);
                CREATE TABLE IF NOT EXISTS snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL, turn_id TEXT NOT NULL,
                    phase TEXT NOT NULL, event_cutoff INTEGER NOT NULL,
                    data TEXT NOT NULL, UNIQUE(session_id, turn_id, phase)
                );
                CREATE INDEX IF NOT EXISTS views_task ON views(task_id, id);
            """)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def task(self, task_id: str) -> dict[str, Any]:
        with self.connect() as db:
            row = db.execute("SELECT data FROM tasks WHERE id=?", (task_id,)).fetchone()
        if row is None:
            raise HarnessError(f"Unknown task_id: {task_id}. Select a task in the current session first.")
        return json.loads(row[0])

    def put_task(self, task: dict[str, Any], *, new: bool = False) -> None:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self.require_session(db, task['scope']['session_id'])
            if new:
                if db.execute("SELECT 1 FROM tasks WHERE id=?", (task["id"],)).fetchone():
                    raise HarnessError(f"Task already selected: {task['id']}")
                task["_revision"] = 1
                db.execute("INSERT INTO tasks VALUES (?, ?)", (task["id"], dumps(task)))
            else:
                row = db.execute("SELECT data FROM tasks WHERE id=?", (task["id"],)).fetchone()
                if row is None or json.loads(row[0]).get("_revision", 0) != task.get("_revision", 0):
                    raise HarnessError("Task state changed concurrently; reload and retry the operation.")
                task["_revision"] = task.get("_revision", 0) + 1
                db.execute("UPDATE tasks SET data=? WHERE id=?", (dumps(task), task["id"]))

    @staticmethod
    def require_session(db: sqlite3.Connection, session_id: str) -> None:
        row = db.execute("SELECT id FROM active_session WHERE singleton=1").fetchone()
        if not row or row[0] != session_id:
            raise HarnessError("Session switched; this operation belongs to discarded history.")

    def cache_file(self, root: str, path: str, content: str, *, session_id: str | None = None) -> int:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if session_id is not None:
                self.require_session(db, session_id)
            row = db.execute(
                "SELECT id FROM files WHERE root=? AND path=? AND content=? ORDER BY id DESC LIMIT 1",
                (root, path, content),
            ).fetchone()
            if row:
                return row[0]
            return db.execute(
                "INSERT INTO files(root,path,content) VALUES (?,?,?)", (root, path, content)
            ).lastrowid

    def current_session(self) -> str | None:
        with self.connect() as db:
            row = db.execute("SELECT id FROM active_session WHERE singleton=1").fetchone()
        return row[0] if row else None

    def activate_session(self, session_id: str) -> bool:
        """Switch atomically; retain the same session and reclaim old local data."""
        if not session_id:
            raise HarnessError("session_id cannot be empty.")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT id FROM active_session WHERE singleton=1").fetchone()
            if row and row[0] == session_id:
                return False
            # Also discard tables from the former manual-registration workflow.
            tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            for table in ("outputs", "views", "tasks", "checkpoints", "snapshots", "recorded_events",
                          "recorded_turns", "recordings", "files", "events", "sessions"):
                if table in tables:
                    db.execute(f"DELETE FROM {table}")
            db.execute("INSERT OR REPLACE INTO active_session VALUES (1, ?)", (session_id,))
        # DELETE frees reusable pages; VACUUM also returns space to the filesystem.
        with self.connect() as db:
            db.execute("VACUUM")
        return True

    def file(self, file_id: int) -> dict[str, Any]:
        with self.connect() as db:
            row = db.execute("SELECT * FROM files WHERE id=?", (file_id,)).fetchone()
        if row is None:
            raise HarnessError(f"File checkpoint is unavailable: {file_id}")
        result = dict(row)
        result["fragment"] = json.loads(result["fragment"]) if result["fragment"] else None
        return result

    def save_fragment(self, file_id: int, fragment: dict[str, Any]) -> None:
        with self.connect() as db:
            db.execute("UPDATE files SET fragment=? WHERE id=?", (dumps(fragment), file_id))

    def save_view(self, task_id: str, data: dict[str, Any]) -> str:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if not db.execute("SELECT 1 FROM tasks WHERE id=?", (task_id,)).fetchone():
                raise HarnessError("Task no longer exists in the active session.")
            number = db.execute(
                "INSERT INTO views(task_id,data) VALUES (?,?)", (task_id, dumps(data))
            ).lastrowid
        return f"V{number}"

    def view(self, view_id: str) -> dict[str, Any]:
        if not view_id.startswith("V") or not view_id[1:].isdigit():
            raise HarnessError("Use an existing evidence view ID.")
        with self.connect() as db:
            row = db.execute("SELECT task_id,data FROM views WHERE id=?", (int(view_id[1:]),)).fetchone()
        if row is None:
            raise HarnessError(f"Unknown evidence view: {view_id}")
        return {**json.loads(row["data"]), "task_id": row["task_id"], "view_id": view_id}

    def latest_view(self, task_id: str) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute("SELECT id FROM views WHERE task_id=? ORDER BY id DESC LIMIT 1", (task_id,)).fetchone()
        return self.view(f"V{row[0]}") if row else None
