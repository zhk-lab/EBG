"""Raw production-artifact backend for the paired repository baseline."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from itertools import zip_longest
from pathlib import PurePosixPath

from beg.core.model import VisibleBundle

from .backend import TokenCounter, content_identity, pack_read_result
from .errors import BackendError
from .evidence import EvidenceSpan, EvidenceUnit, SearchHit, ToolResult


_INDEX_LANES = 16
_RAW_CHUNK_LINES = 400


class RawBackend:
    """Expose each retained production artifact as one stable source unit."""

    kind = "raw"

    def __init__(
        self,
        bundle: VisibleBundle,
        *,
        index_budget: int,
        count_tokens: TokenCounter,
    ) -> None:
        if bundle.benchmark not in {"specgap", "silentswap"}:
            raise BackendError("RawBackend supports only Repo benchmarks")
        artifacts = sorted(bundle.repo_artifacts, key=lambda item: item.path)
        units: dict[str, EvidenceUnit] = {}
        unit_number = 0
        for artifact in artifacts:
            path = _normalize_path(artifact.path)
            source_lines = artifact.content.splitlines()
            if not source_lines:
                continue
            for offset in range(0, len(source_lines), _RAW_CHUNK_LINES):
                start = offset + 1
                end = min(offset + _RAW_CHUNK_LINES, len(source_lines))
                unit_number += 1
                unit_id = f"R{unit_number:04d}"
                source = "\n".join(
                    f"{number} | {source_lines[number - 1]}"
                    for number in range(start, end + 1)
                )
                units[unit_id] = EvidenceUnit(
                    unit_id=unit_id,
                    unit_kind="source",
                    name=f"{path} lines {start}-{end}",
                    span=EvidenceSpan(
                        path=path,
                        symbol="<file>",
                        start=start,
                        end=end,
                    ),
                    source=source,
                )
        if not units:
            raise BackendError("RawBackend received no non-empty production artifact")
        self.input_id = bundle.input_id
        self.benchmark = bundle.benchmark
        self._units = units
        self.initial_index, selected = _build_index(
            tuple(units.values()), index_budget, count_tokens
        )
        self.initial_index_tokens = count_tokens(self.initial_index)
        self._initial_ids = frozenset(selected)
        self._resume_identity = content_identity(
            [
                f"{artifact.path}\0{artifact.kind}\0{artifact.content}"
                for artifact in artifacts
            ]
            + [self.initial_index]
        )

    @property
    def initial_ids(self) -> frozenset[str]:
        return self._initial_ids

    @property
    def priority_groups(self) -> dict[str, tuple[str, ...]]:
        return {"Repository source units": tuple(sorted(self._initial_ids))}

    @property
    def resume_identity(self) -> dict[str, object]:
        return dict(self._resume_identity)

    def has_id(self, unit_id: str) -> bool:
        return unit_id in self._units

    def search(
        self,
        text: str,
        *,
        limit: int,
        exact_path_limit: int | None = None,
    ) -> ToolResult:
        keywords = tuple(part.casefold() for part in text.split() if part)
        folded_query = text.casefold().strip()
        ranked: list[tuple[int, str, SearchHit]] = []
        partial_fallback = False
        for unit in self._units.values():
            path = unit.span.path.casefold()
            if folded_query in {unit.unit_id.casefold(), path}:
                rank, field = 0, "unit_id" if folded_query == unit.unit_id.casefold() else "path"
            elif keywords and all(keyword in path for keyword in keywords):
                rank, field = 1, "path"
            elif keywords and all(
                keyword in f"{path}\n{unit.source.casefold()}"
                for keyword in keywords
            ):
                rank, field = 2, "source"
            else:
                continue
            ranked.append(
                (
                    rank,
                    unit.unit_id,
                    SearchHit(
                        unit_id=unit.unit_id,
                        name=unit.name,
                        path=unit.span.path,
                        symbol=unit.span.symbol,
                        matched_field=field,
                    ),
                )
            )
        partial_strength = 0
        if not ranked and len(keywords) >= 2:
            partial_fallback = True
            for unit in self._units.values():
                searchable = f"{unit.span.path}\n{unit.source}".casefold()
                matched = sum(keyword in searchable for keyword in keywords)
                if not matched:
                    continue
                ranked.append(
                    (
                        -matched,
                        unit.unit_id,
                        SearchHit(
                            unit_id=unit.unit_id,
                            name=unit.name,
                            path=unit.span.path,
                            symbol=unit.span.symbol,
                            matched_field=f"partial {matched}/{len(keywords)}",
                        ),
                    )
                )
            if ranked:
                best_rank = min(item[0] for item in ranked)
                partial_strength = -best_rank
                ranked = [item for item in ranked if item[0] == best_rank]
        ranked.sort(key=lambda item: (item[0], item[1]))
        exact_path = any(
            folded_query == unit.span.path.casefold()
            for unit in self._units.values()
        )
        result_limit = exact_path_limit if exact_path and exact_path_limit else limit
        broad_partial = partial_fallback and partial_strength == 1
        if (
            len(ranked) > result_limit
            and not exact_path
            and (not partial_fallback or broad_partial)
        ):
            return ToolResult(
                action="search",
                status="ok",
                search_text=text,
                total_matches=len(ranked),
                group_hints=_group_hints(
                    [item[2] for item in ranked], result_limit
                ),
            )
        return ToolResult(
            action="search",
            status="ok",
            search_text=text,
            total_matches=len(ranked),
            hits=tuple(item[2] for item in ranked[:result_limit]),
        )

    def read(
        self,
        ids: Sequence[str],
        *,
        token_budget: int,
        max_atomic_unit_tokens: int,
        count_tokens: TokenCounter,
    ) -> ToolResult:
        return pack_read_result(
            ids,
            self._units,
            token_budget=token_budget,
            max_atomic_unit_tokens=max_atomic_unit_tokens,
            count_tokens=count_tokens,
        )


def _build_index(
    units: tuple[EvidenceUnit, ...],
    token_budget: int,
    count_tokens: Callable[[str], int],
) -> tuple[str, tuple[str, ...]]:
    header = [
        "This is a neutral partial repository index; listed units are not "
        "necessarily problematic.",
        "Use search to inspect the complete index and read to inspect original source.",
        "",
        "[REPOSITORY SOURCE UNITS]",
        "",
    ]
    base = "\n".join(header).rstrip() + "\n"
    if count_tokens(base) > token_budget:
        raise BackendError("index budget cannot hold the fixed Raw index header")
    ordered = _coverage_order(units)
    selected: list[EvidenceUnit] = []
    for unit in ordered:
        proposed = [*selected, unit]
        text = _render_index(header, proposed)
        if count_tokens(text) <= token_budget:
            selected = proposed
    return _render_index(header, selected), tuple(unit.unit_id for unit in selected)


def _render_index(header: list[str], units: list[EvidenceUnit]) -> str:
    lines = list(header)
    if not units:
        lines.append("No source unit fits; use search to inspect the complete index.")
    else:
        for unit in units:
            lines.append(
                f"{unit.unit_id}  {unit.span.path}  "
                f"(lines {unit.span.start}-{unit.span.end})"
            )
    return "\n".join(lines).rstrip() + "\n"


def _coverage_order(units: tuple[EvidenceUnit, ...]) -> list[EvidenceUnit]:
    lane_count = min(_INDEX_LANES, len(units))
    lane_size = (len(units) + lane_count - 1) // lane_count
    lanes = [
        units[start : start + lane_size]
        for start in range(0, len(units), lane_size)
    ]
    return [item for row in zip_longest(*lanes) for item in row if item is not None]


def _normalize_path(path: str) -> str:
    normalized = path.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    pure = PurePosixPath(normalized)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise BackendError(f"invalid repository path: {path!r}")
    return pure.as_posix()


def _group_hints(hits: list[SearchHit], limit: int) -> tuple[str, ...]:
    counts: Counter[str] = Counter()
    for hit in hits:
        parts = [part for part in hit.path.split("/") if part]
        group = "/".join(parts[:2]) if len(parts) > 1 else hit.path
        counts[group] += 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return tuple(f"{group} ({count} matches)" for group, count in ordered[:limit])
