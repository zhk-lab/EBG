"""Read repository snapshots and cache BEG's per-file construction stages."""

from __future__ import annotations

import ast
import copy
import os
import subprocess
from pathlib import Path
from typing import Any

from ..beg.behavior_atomization import build_behaviors
from ..beg.core.errors import BEGError
from ..beg.core.model import RepoArtifact, TaskDocument, VisibleBundle
from ..beg.evidence_intake import (
    BINARY_EXTENSIONS, CONFIG_EXTENSIONS, EXECUTABLE_EXTENSIONS, SOURCE_EXTENSIONS,
    build_evidence, classify_repository_file, source_symbol_spans,
)
from ..beg.graph_assembly import build_graph
from ..beg.relation_linking import build_edges

from .storage import HarnessError, Store


SKIP_DIRS = {".git", ".venv", ".venv-harness", "venv", "node_modules", "__pycache__", ".beg-harness", ".state", ".tmp", ".tmp-tests", "dist", "build"}
GRAPH_VERSION = 1


def bundle(root: Path, artifacts: list[RepoArtifact], document: str = "") -> VisibleBundle:
    return VisibleBundle(root, "harness", "specgap", TaskDocument("requirements", document), tuple(artifacts), (), None, len(artifacts), 0)


def capture(store: Store, root: Path, *, max_file_bytes: int = 2_000_000,
            session_id: str | None = None) -> tuple[dict[str, int], list[str], dict[str, dict[str, Any]]]:
    if not root.is_dir():
        raise HarnessError(f"Repository directory is unavailable: {root}")
    names = None
    if (root / ".git").exists():
        try:
            result = subprocess.run(
                ["git", "-C", str(root), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                stdin=subprocess.DEVNULL, capture_output=True, check=True, timeout=30,
            )
            names = sorted(set(result.stdout.decode("utf-8").split("\0")) - {""})
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass
    if names is None:
        names = []
        for directory, children, filenames in os.walk(root):
            children[:] = [name for name in children if name not in SKIP_DIRS and not name.endswith(".egg-info")
                           and not (Path(directory) / name).resolve().is_relative_to(store.directory)]
            names.extend((Path(directory) / name).relative_to(root).as_posix() for name in filenames)
        names.sort()
    files: dict[str, int] = {}
    notes: list[str] = []
    uncollected: dict[str, dict[str, Any]] = {}
    observed = {}
    for name in names:
        path = root / name
        if any(part in SKIP_DIRS or part.endswith(".egg-info") for part in Path(name).parts):
            continue
        resolved = path.resolve()
        if resolved.is_relative_to(store.directory):
            continue
        if not resolved.is_relative_to(root):
            notes.append(f"{name}: symlink points outside the registered repository")
            continue
        if not path.is_file():
            continue
        try:
            before = path.stat()
            observed[path] = (before.st_mtime_ns, before.st_size)
            uncollected[name] = {'exists': True, 'size_bytes': before.st_size,
                                 'content_status': 'not_collected', 'reason': 'read_error'}
            if path.suffix.lower() in BINARY_EXTENSIONS:
                uncollected[name]['reason'] = 'binary_extension'
                continue
            if before.st_size > max_file_bytes:
                uncollected[name]['reason'] = 'size_limit'
                notes.append(f"{name}: exceeds file collection limit ({max_file_bytes} bytes)")
                continue
            raw = path.read_bytes()
            after = path.stat()
            if (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size):
                raise HarnessError(f"{name} changed during collection; retry refresh after the writer finishes.")
            if b"\0" in raw:
                uncollected[name]['reason'] = 'binary_content'
                continue
            content = raw.decode("utf-8-sig")
        except (OSError, UnicodeDecodeError) as error:
            if isinstance(error, UnicodeDecodeError):
                uncollected[name]['reason'] = 'non_utf8'
            notes.append(f"{name}: {type(error).__name__}; not collected")
            continue
        files[name] = store.cache_file(str(root), name, content, session_id=session_id)
        del uncollected[name]
        observed[path] = (after.st_mtime_ns, after.st_size)
    for path, expected in observed.items():
        try:
            stat = path.stat()
        except OSError as error:
            raise HarnessError(f"{path.name} changed during collection; retry refresh.") from error
        if (stat.st_mtime_ns, stat.st_size) != expected:
            raise HarnessError(f"{path.name} changed during collection; retry refresh after the writer finishes.")
    return files, notes, uncollected


def artifact(root: Path, path: str, content: str) -> RepoArtifact | None:
    kind = classify_repository_file(path, content[:4096].encode("utf-8"))
    # Harness admission includes verification artifacts; benchmark intake stays unchanged.
    if kind is None:
        suffix = Path(path).suffix.lower()
        if suffix in SOURCE_EXTENSIONS:
            kind = "source"
        elif suffix in CONFIG_EXTENSIONS:
            kind = "configuration"
        elif suffix in EXECUTABLE_EXTENSIONS:
            kind = "executable"
    return RepoArtifact(path, root / path, kind, content) if kind else None


def module_statement_span(path: str, content: str, first: int, last: int) -> tuple[int, int]:
    """Keep enclosing module-level conditions when selecting a definition."""
    if Path(path).suffix.lower() in {".py", ".pyw"}:
        try:
            statements = ast.parse(content).body
        except SyntaxError:
            return first, last
        for statement in statements:
            if statement.lineno <= first <= last <= statement.end_lineno:
                return statement.lineno, statement.end_lineno
    return first, last


def construct_graph(store: Store, root: Path, files: dict[str, int]) -> tuple[dict[str, Any], list[dict[str, Any]], list[str], int]:
    artifacts = []
    evidence = []
    behaviors = []
    contexts = []
    notes = []
    built = 0
    for path, file_id in sorted(files.items()):
        record = store.file(file_id)
        item = artifact(root, path, record["content"])
        fragment = record["fragment"]
        if fragment is None or fragment.get("version") != GRAPH_VERSION:
            fragment = {"version": GRAPH_VERSION, "evidence": [], "behaviors": [], "contexts": [], "notes": []}
            if item and item.content.strip():
                local = bundle(root, [item])
                spans = source_symbol_spans(local)
                lines = item.content.splitlines(keepends=True)
                fragment["contexts"] = [
                    {"path": path, "symbol": span.symbol, "lines": [span.line_start, span.line_end],
                     "source": "".join(lines[span.line_start - 1:span.line_end])}
                    for values in spans.values() for span in values
                ]
                try:
                    fragment["evidence"] = build_evidence(local)
                    fragment["behaviors"] = build_behaviors(local, fragment["evidence"])
                except BEGError as error:
                    fragment["evidence"] = []
                    fragment["behaviors"] = []
                    fragment["notes"].append(f"{path}: BEG atomization unavailable ({error}); raw source retained")
            store.save_fragment(file_id, fragment)
            built += 1
        contexts.extend(fragment["contexts"])
        notes.extend(fragment["notes"])
        if not fragment["evidence"] or item is None:
            continue
        artifacts.append(item)
        mapping = {}
        for original in fragment["evidence"]:
            node = copy.deepcopy(original)
            node["evidence_id"] = f"E{len(evidence) + 1:06d}"
            mapping[original["evidence_id"]] = node["evidence_id"]
            evidence.append(node)
        for original in fragment["behaviors"]:
            behavior = copy.deepcopy(original)
            behavior["behavior_id"] = f"B{len(behaviors) + 1:04d}"
            for key in ("trigger_evidence_ids", "operation_evidence_ids", "result_evidence_ids"):
                behavior[key] = [mapping[value] for value in behavior[key]]
            behaviors.append(behavior)
    if not evidence:
        return {"evidence": [], "behaviors": [], "source_contexts": [], "edges": []}, contexts, notes, built
    combined = bundle(root, artifacts)
    edges = build_edges(combined, evidence, behaviors)
    return build_graph(combined, evidence, behaviors, edges), contexts, notes, built
