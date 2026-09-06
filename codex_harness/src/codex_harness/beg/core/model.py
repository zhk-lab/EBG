"""Small immutable objects shared by deterministic BEG modules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


Benchmark = Literal["specgap", "silentswap", "feedbacktrace"]
ArtifactKind = Literal[
    "source", "runtime_template", "executable", "configuration"
]


@dataclass(frozen=True, slots=True)
class TaskDocument:
    path: str
    content: str


@dataclass(frozen=True, slots=True)
class RepoArtifact:
    path: str
    absolute_path: Path
    kind: ArtifactKind
    content: str


@dataclass(frozen=True, slots=True)
class TraceEvent:
    event_type: str
    turn_number: int
    content: str
    evidence_id: str | None
    tool_name: str | None
    input_order: int


@dataclass(frozen=True, slots=True)
class VisibleBundle:
    root: Path
    input_id: str
    benchmark: Benchmark
    task_document: TaskDocument | None
    repo_artifacts: tuple[RepoArtifact, ...]
    trace_events: tuple[TraceEvent, ...]
    cutoff_turn: int | None
    visible_repository_files: int
    excluded_repository_files: int


@dataclass(frozen=True, slots=True)
class SourceSpan:
    path: str
    symbol: str
    line_start: int
    line_end: int


@dataclass(frozen=True, slots=True)
class BehaviorCandidate:
    path: str
    symbol: str
    result_kind: str
    result_line: int
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SymbolSpan:
    path: str
    symbol: str
    line_start: int
    line_end: int
