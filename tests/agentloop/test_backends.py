from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agentloop.backend import pack_read_result
from agentloop.evidence import EvidenceUnit, render_tool_result
from agentloop.errors import ContextUnfitError
from agentloop.raw_backend import RawBackend
from ebg.core.model import RepoArtifact, TaskDocument, VisibleBundle
from evaluation_core.contracts import EvidenceSpan


def bundle(*, source: str | None = None) -> VisibleBundle:
    content = source or (
        "def public(value):\n"
        "    if value is None:\n"
        "        raise TypeError('value')\n"
        "    return value\n"
    )
    return VisibleBundle(
        root=Path("."),
        input_id="sg_backend",
        benchmark="specgap",
        task_document=TaskDocument("3_document_after.md", "Use public."),
        repo_artifacts=(
            RepoArtifact(
                path="pkg/api.py",
                absolute_path=Path("pkg/api.py"),
                kind="source",
                content=content,
            ),
        ),
        trace_events=(),
        cutoff_turn=None,
        visible_repository_files=1,
        excluded_repository_files=0,
    )


class BackendTests(unittest.TestCase):
    def test_complete_unit_between_soft_and_hard_limits_is_kept(self) -> None:
        unit = EvidenceUnit(
            unit_id="R0001",
            unit_kind="source",
            name="large.py",
            span=EvidenceSpan("large.py", "<module>", 1, 1),
            source="x" * 40_000,
        )
        result = pack_read_result(
            ["R0001"],
            {"R0001": unit},
            token_budget=32_768,
            max_atomic_unit_tokens=65_536,
            count_tokens=len,
        )
        self.assertEqual(result.returned_ids, ("R0001",))
        self.assertTrue(result.oversize_unit)

    def test_complete_unit_above_hard_limit_fails_explicitly(self) -> None:
        unit = EvidenceUnit(
            unit_id="R0001",
            unit_kind="source",
            name="too_large.py",
            span=EvidenceSpan("too_large.py", "<module>", 1, 1),
            source="x" * 70_000,
        )
        with self.assertRaises(ContextUnfitError):
            pack_read_result(
                ["R0001"],
                {"R0001": unit},
                token_budget=32_768,
                max_atomic_unit_tokens=65_536,
                count_tokens=len,
            )

    def test_raw_backend_queries_and_reads_complete_source(self) -> None:
        backend = RawBackend(bundle(), index_budget=3_072, count_tokens=len)
        search = backend.search("public", limit=12)
        result = backend.read(
            [search.hits[0].unit_id],
            token_budget=10_000,
            max_atomic_unit_tokens=20_000,
            count_tokens=len,
        )
        rendered = render_tool_result(result)
        self.assertIn("1 | def public(value):", rendered)
        self.assertIn("4 |     return value", rendered)

    def test_raw_backend_splits_only_at_fixed_source_unit_boundaries(self) -> None:
        source = "".join(f"line_{number} = {number}\n" for number in range(1, 802))
        backend = RawBackend(
            bundle(source=source), index_budget=3_072, count_tokens=len
        )
        matches = backend.search("pkg/api.py", limit=12, exact_path_limit=12)
        self.assertEqual(len(matches.hits), 3)
        self.assertEqual(
            [(backend._units[item.unit_id].span.start, backend._units[item.unit_id].span.end) for item in matches.hits],
            [(1, 400), (401, 800), (801, 801)],
        )

    def test_resume_identity_changes_with_source(self) -> None:
        first = RawBackend(bundle(), index_budget=3_072, count_tokens=len)
        second = RawBackend(
            bundle(source="def public():\n    return 2\n"),
            index_budget=3_072,
            count_tokens=len,
        )
        self.assertNotEqual(first.resume_identity, second.resume_identity)


if __name__ == "__main__":
    unittest.main()
