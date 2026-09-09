"""Prioritize SilentSwap files by distinct documented behavior coverage."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .behavior_directory import _RepoFile


def rank_files(files: list[_RepoFile]) -> list[_RepoFile]:
    """Rank production source before auxiliary artifacts, retaining every file.

    Configuration keys often match generic document terms. Their apparent
    coverage must not displace production source from the first read batch.
    Within each artifact tier, prefer distinct documented behavior coverage.
    """
    from .behavior_directory import _ARTIFACT_ORDER

    return [
        item
        for kind in sorted(_ARTIFACT_ORDER, key=_ARTIFACT_ORDER.get)
        for item in _rank_coverage([item for item in files if item.artifact_kind == kind])
    ]


def _rank_coverage(files: list[_RepoFile]) -> list[_RepoFile]:
    """Cover specific document sections before expanding overlapping files.

    Shared overview sections contribute less than file-specific contracts.
    Previously covered sections retain diminishing weight: several files can
    implement different parts of the same contract.
    """
    from .behavior_directory import _file_rank

    frequency = Counter(order for item in files for order in set(item.section_orders))
    covered: Counter[int] = Counter()
    remaining = list(files)
    ordered: list[_RepoFile] = []
    while remaining:
        def priority(item: _RepoFile) -> tuple:
            coverage = sum(
                1 / (frequency[order] * (1 + covered[order]))
                for order in sorted(set(item.section_orders))
            )
            return (-coverage, *_file_rank(item))

        chosen = min(remaining, key=priority)
        ordered.append(chosen)
        remaining.remove(chosen)
        covered.update(set(chosen.section_orders))
    return ordered
