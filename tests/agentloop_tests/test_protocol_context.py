from __future__ import annotations

import json
import sys
import unittest
from dataclasses import replace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agentloop.config import AgentLoopConfig
from agentloop.context import FastTokenCounter, HistoryEntry, prepare_messages
from agentloop.errors import ActionParameterError, AgentLoopError, ProtocolError
from agentloop.evidence import (
    BehaviorEvidence,
    EvidenceSpan,
    EvidenceUnit,
    SearchHit,
    ToolResult,
)
from agentloop.protocol import FinishAction, ReadAction, SearchAction, parse_action


class ProtocolTests(unittest.TestCase):
    def test_default_context_compression_threshold_and_target_are_frozen(self) -> None:
        config = AgentLoopConfig()

        self.assertEqual(config.working_prompt_trigger, 131_072)
        self.assertEqual(config.compression_target, 98_304)

    def test_exact_path_limit_cannot_be_below_normal_search_limit(self) -> None:
        with self.assertRaises(AgentLoopError):
            AgentLoopConfig(max_search_hits=12, max_exact_path_hits=11)

    def test_exact_three_actions(self) -> None:
        search = parse_action(
            '{"action":"search","text":"  Foo   bar "}',
            max_read_ids=6,
            max_search_characters=100,
        )
        read = parse_action(
            '{"action":"read","ids":["S0001","S0002"]}',
            max_read_ids=6,
            max_search_characters=100,
        )
        finish = parse_action(
            '{"action":"finish","prediction":{"findings":[]}}',
            max_read_ids=6,
            max_search_characters=100,
        )

        self.assertEqual(search, SearchAction("Foo bar"))
        self.assertEqual(read, ReadAction(("S0001", "S0002")))
        self.assertIsInstance(finish, FinishAction)

    def test_default_batch_read_accepts_6_ids_and_rejects_7(self) -> None:
        config = AgentLoopConfig()
        self.assertEqual(config.max_read_ids, 6)
        self.assertEqual(config.tool_result_budget, 32_768)
        self.assertEqual(config.max_atomic_unit_tokens, 65_536)
        accepted_ids = [f"F{number:04d}" for number in range(1, 7)]
        action = parse_action(
            json.dumps({"action": "read", "ids": accepted_ids}),
            max_read_ids=config.max_read_ids,
            max_search_characters=config.max_search_characters,
        )
        self.assertEqual(action, ReadAction(tuple(accepted_ids)))

        rejected_ids = [*accepted_ids, "F0007"]
        with self.assertRaises(ActionParameterError):
            parse_action(
                json.dumps({"action": "read", "ids": rejected_ids}),
                max_read_ids=config.max_read_ids,
                max_search_characters=config.max_search_characters,
            )

    def test_markdown_unknown_and_duplicate_json_fail_without_repair(self) -> None:
        values = (
            '```json\n{"action":"search","text":"x"}\n```',
            '{"action":"query","text":"x"}',
            '{"action":"search","action":"search","text":"x"}',
        )
        for value in values:
            with self.subTest(value=value), self.assertRaises(ProtocolError):
                parse_action(value, max_read_ids=6, max_search_characters=100)

    def test_known_action_parameter_error_is_distinct(self) -> None:
        with self.assertRaises(ActionParameterError) as raised:
            parse_action(
                '{"action":"read","ids":[]}',
                max_read_ids=6,
                max_search_characters=100,
            )
        self.assertEqual(raised.exception.action, "read")

    def test_concatenated_actions_use_only_the_first(self) -> None:
        action = parse_action(
            '{"action":"read","ids":["S0001"]}'
            '{"action":"finish","prediction":{"findings":[]}}',
            max_read_ids=6,
            max_search_characters=100,
        )

        self.assertEqual(action, ReadAction(("S0001",)))

    def test_action_array_uses_only_the_first(self) -> None:
        action = parse_action(
            '[{"action":"search","text":"first"},'
            '{"action":"read","ids":["S0001"]}]',
            max_read_ids=6,
            max_search_characters=100,
        )

        self.assertEqual(action, SearchAction("first"))

    def test_separator_text_between_actions_does_not_hide_the_first(self) -> None:
        action = parse_action(
            '{"action":"read","ids":["S0001"]}'
            'unexpected separator'
            '{"action":"search","text":"unused"}',
            max_read_ids=6,
            max_search_characters=100,
        )

        self.assertEqual(action, ReadAction(("S0001",)))

    def test_provider_garbage_after_action_is_rejected(self) -> None:
        with self.assertRaises(ProtocolError):
            parse_action(
                '{"action":"read","ids":["S0001"]}'
                'unexpected provider text',
                max_read_ids=6,
                max_search_characters=100,
            )

    def test_duplicate_key_error_names_the_duplicate(self) -> None:
        with self.assertRaisesRegex(ProtocolError, "duplicate JSON key: start"):
            parse_action(
                '{"action":"finish","prediction":{"start":1,"start":2}}',
                max_read_ids=6,
                max_search_characters=100,
            )

    def test_duplicate_start_line_range_is_repaired(self) -> None:
        action = parse_action(
            '{"action":"finish","prediction":{"line_ranges":'
            '[{"start":3,"start":7}]}}',
            max_read_ids=12,
            max_search_characters=100,
        )

        self.assertEqual(
            action.as_dict()["prediction"]["line_ranges"],
            [{"start": 3, "end": 7}],
        )


class FastTokenCounterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.counter = FastTokenCounter()

    def test_empty_ascii_chinese_and_emoji_use_utf8_byte_estimate(self) -> None:
        cases = {
            "": 0,
            "abcd": 2,
            "你": 1,
            "🙂": 2,
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(self.counter.count_text(text), expected)

    def test_message_count_uses_the_canonical_compact_json_envelope(self) -> None:
        messages = [
            {"role": "system", "content": "规则"},
            {"role": "user", "content": "read S0001 🙂"},
        ]
        serialized = json.dumps(
            {"messages": messages},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        expected = (len(serialized.encode("utf-8")) + 2) // 3

        self.assertEqual(self.counter.count_messages(messages), expected)


class ContextTests(unittest.TestCase):
    def setUp(self) -> None:
        self.counter = FastTokenCounter()
        self.config = AgentLoopConfig(
            context_window=50_000,
            max_output_tokens=500,
            provider_safety_margin=500,
            working_prompt_trigger=30_000,
            compression_target=20_000,
            index_budget=200,
            tool_result_budget=500,
            max_atomic_unit_tokens=1_000,
            max_search_hits=10,
            max_exact_path_hits=10,
            max_search_characters=100,
            max_read_ids=2,
            preferred_recent_reads=1,
        )

    def test_fixed_input_and_run_state_each_appear_once(self) -> None:
        history = [
            HistoryEntry(
                1,
                '{"action":"read","ids":["S0001"]}',
                {},
                self._read(self._symbol("S0001", source="LATEST-SOURCE")),
            )
        ]

        prepared = self._prepare(
            history,
            system="SYSTEM-ONCE",
            initial_user="DOCUMENT-ONCE INDEX-ONCE",
        )
        joined = self._joined(prepared)

        self.assertEqual(joined.count("SYSTEM-ONCE"), 1)
        self.assertEqual(joined.count("DOCUMENT-ONCE"), 1)
        self.assertEqual(joined.count("INDEX-ONCE"), 1)
        self.assertEqual(joined.count("[[RUN STATE]]"), 1)
        self.assertEqual(joined.count("[[END RUN STATE]]"), 1)

    def test_fully_consumed_search_becomes_a_receipt(self) -> None:
        search = ToolResult(
            action="search",
            status="ok",
            search_text="cache",
            total_matches=2,
            hits=(
                SearchHit(
                    "S0001", "pkg/cache.py::get", "pkg/cache.py", "get", "symbol"
                ),
                SearchHit(
                    "S0002", "pkg/cache.py::put", "pkg/cache.py", "put", "symbol"
                ),
            ),
        )
        history = [
            HistoryEntry(1, "search-action", {}, search),
            HistoryEntry(
                2,
                "read-action",
                {},
                self._read(
                    self._symbol("S0001", source="FIRST-SELECTED-SOURCE"),
                    self._symbol(
                        "S0002", source="SECOND-SELECTED-SOURCE", start=10
                    ),
                ),
            ),
        ]

        joined = self._joined(self._prepare(history, force_compression=True))
        receipt = joined.split("[[SEARCH RECEIPT]]", 1)[1].split(
            "[[END SEARCH RECEIPT]]", 1
        )[0]

        self.assertIn("IDs later selected for read: S0001, S0002", receipt)
        self.assertNotIn("pkg/cache.py::get", receipt)

    def test_unconsumed_search_hits_survive_forced_compression(self) -> None:
        search = ToolResult(
            action="search",
            status="ok",
            search_text="cache",
            total_matches=2,
            hits=(
                SearchHit("S0001", "get", "pkg/cache.py", "get", "symbol"),
                SearchHit("S0002", "put", "pkg/cache.py", "put", "symbol"),
            ),
        )
        history = [
            HistoryEntry(1, "search-action", {}, search),
            HistoryEntry(
                2,
                "read-one",
                {},
                self._read(self._symbol("S0002", source="SELECTED-SOURCE")),
            ),
        ]

        prepared = self._prepare(history, force_compression=True)
        joined = self._joined(prepared)

        self.assertEqual(prepared.manifest.search_receipt_turns, ())
        self.assertIn("[[SEARCH RESULT]]", joined)
        self.assertIn("S0001", joined)

    def test_deferred_read_id_does_not_consume_its_search_hit(self) -> None:
        search = ToolResult(
            action="search",
            status="ok",
            search_text="cache",
            total_matches=2,
            hits=(
                SearchHit("S0001", "get", "pkg/cache.py", "get", "symbol"),
                SearchHit("S0002", "put", "pkg/cache.py", "put", "symbol"),
            ),
        )
        first = self._symbol("S0001", source="RETURNED-SOURCE")
        partial_read = ToolResult(
            action="read",
            status="ok",
            requested_ids=("S0001", "S0002"),
            units=(first,),
            deferred_ids=("S0002",),
        )
        history = [
            HistoryEntry(1, "search-action", {}, search),
            HistoryEntry(2, "partial-read", {}, partial_read),
        ]

        prepared = self._prepare(history, force_compression=True)
        joined = self._joined(prepared)

        self.assertEqual(prepared.manifest.search_receipt_turns, ())
        self.assertIn("[[SEARCH RESULT]]", joined)
        self.assertIn("S0002", joined)

    def test_working_compression_starts_at_the_frozen_trigger(self) -> None:
        history = [
            HistoryEntry(
                1,
                "search-one",
                {},
                ToolResult(action="search", status="ok", search_text="first"),
            ),
            HistoryEntry(
                2,
                "search-two",
                {},
                ToolResult(action="search", status="ok", search_text="second"),
            ),
        ]
        baseline = self._prepare(history)
        below_threshold = replace(
            self.config,
            working_prompt_trigger=baseline.token_count + 1,
            compression_target=baseline.token_count,
        )
        at_threshold = replace(
            self.config,
            working_prompt_trigger=baseline.token_count,
            compression_target=baseline.token_count - 1,
        )

        not_triggered = prepare_messages(
            system="system",
            initial_user="initial",
            history=history,
            max_rounds=6,
            config=below_threshold,
            counter=self.counter,
        )
        triggered = prepare_messages(
            system="system",
            initial_user="initial",
            history=history,
            max_rounds=6,
            config=at_threshold,
            counter=self.counter,
        )

        self.assertFalse(not_triggered.manifest.working_compression_triggered)
        self.assertEqual(not_triggered.manifest.search_receipt_turns, ())
        self.assertTrue(triggered.manifest.working_compression_triggered)
        self.assertEqual(triggered.manifest.search_receipt_turns, (1,))

    def test_unique_old_and_latest_reads_both_keep_full_source(self) -> None:
        old = self._symbol(
            "S0001",
            source="OLD-FULL-SOURCE-MUST-STAY",
            document_excerpt="OLD-DOCUMENT-EXCERPT",
            evidence_text="OLD-BEHAVIOR-EVIDENCE",
        )
        latest = self._symbol(
            "S0002",
            source="LATEST-FULL-SOURCE-MUST-STAY",
            document_excerpt="LATEST-DOCUMENT-EXCERPT",
            evidence_text="LATEST-BEHAVIOR-EVIDENCE",
            start=10,
        )
        history = [
            HistoryEntry(1, "read-old", {}, self._read(old)),
            HistoryEntry(2, "read-latest", {}, self._read(latest)),
        ]

        prepared = self._prepare(history, force_compression=True)
        joined = self._joined(prepared)

        self.assertNotIn((1, "S0001"), prepared.manifest.read_receipt_units)
        self.assertNotIn((2, "S0002"), prepared.manifest.read_receipt_units)
        self.assertIn("OLD-DOCUMENT-EXCERPT", joined)
        self.assertNotIn("OLD-BEHAVIOR-EVIDENCE", joined)
        self.assertIn("OLD-FULL-SOURCE-MUST-STAY", joined)
        self.assertIn("LATEST-DOCUMENT-EXCERPT", joined)
        self.assertNotIn("LATEST-BEHAVIOR-EVIDENCE", joined)
        self.assertIn("LATEST-FULL-SOURCE-MUST-STAY", joined)

    def test_duplicate_reads_are_not_compacted_below_trigger(self) -> None:
        duplicate = self._symbol("S0001", source="DUPLICATE-SOURCE-BELOW-TRIGGER")
        history = [
            HistoryEntry(1, "read-one", {}, self._read(duplicate)),
            HistoryEntry(2, "read-two", {}, self._read(duplicate)),
        ]

        prepared = self._prepare(history)
        joined = self._joined(prepared)

        self.assertFalse(prepared.manifest.working_compression_triggered)
        self.assertEqual(prepared.manifest.duplicate_receipt_units, ())
        self.assertEqual(joined.count("DUPLICATE-SOURCE-BELOW-TRIGGER"), 2)

    def test_only_an_older_duplicate_read_is_compacted(self) -> None:
        unit = self._symbol("S0001", source="IDENTICAL-FULL-SOURCE")
        history = [
            HistoryEntry(1, "read-first", {}, self._read(unit)),
            HistoryEntry(2, "read-again", {}, self._read(unit)),
        ]

        prepared = self._prepare(history, force_compression=True)
        joined = self._joined(prepared)

        self.assertIn((1, "S0001"), prepared.manifest.duplicate_receipt_units)
        self.assertNotIn((2, "S0001"), prepared.manifest.duplicate_receipt_units)
        self.assertEqual(joined.count("IDENTICAL-FULL-SOURCE"), 1)

    def test_one_response_prints_only_one_full_source_for_contained_symbols(self) -> None:
        outer = self._symbol(
            "S0001",
            path="pkg/api.py",
            symbol="outer",
            start=1,
            end=8,
            source="OUTER-COMPLETE-SOURCE",
            evidence_text="OUTER-BEHAVIOR",
        )
        inner = self._symbol(
            "S0002",
            path="pkg/api.py",
            symbol="outer.inner",
            start=3,
            end=5,
            source="INNER-SOURCE-MUST-NOT-REPEAT",
            evidence_text="INNER-BEHAVIOR",
        )

        joined = self._joined(
            self._prepare(
                [HistoryEntry(1, "read-both", {}, self._read(outer, inner))]
            )
        )

        self.assertEqual(joined.count("Full symbol source:"), 1)
        self.assertIn("OUTER-COMPLETE-SOURCE", joined)
        self.assertNotIn("INNER-SOURCE-MUST-NOT-REPEAT", joined)
        self.assertNotIn("OUTER-BEHAVIOR", joined)
        self.assertNotIn("INNER-BEHAVIOR", joined)
        self.assertIn("Full source is included under Read ID S0001", joined)

    def test_explicit_child_keeps_its_anchor_when_parent_anchor_list_is_sampled(self) -> None:
        evidence = tuple(
            BehaviorEvidence(
                labels=(f"operation_{line}",),
                start=line,
                end=line,
                text=f"{line} | operation_{line}()",
                owner="outer.inner" if line == 3 else "outer",
            )
            for line in range(1, 11)
        )
        outer = replace(
            self._symbol(
                "S0001",
                path="pkg/api.py",
                symbol="outer",
                start=1,
                end=12,
                source="OUTER-COMPLETE-SOURCE",
            ),
            behavior_evidence=evidence,
        )
        inner = replace(
            self._symbol(
                "S0002",
                path="pkg/api.py",
                symbol="outer.inner",
                start=3,
                end=3,
                source="INNER-SOURCE-MUST-NOT-REPEAT",
            ),
            behavior_evidence=(evidence[2],),
        )

        joined = self._joined(
            self._prepare(
                [HistoryEntry(1, "read-both", {}, self._read(outer, inner))]
            )
        )

        self.assertEqual(joined.count("Full symbol source:"), 1)
        self.assertIn("operation_3 | key lines 3", joined)
        self.assertNotIn("INNER-SOURCE-MUST-NOT-REPEAT", joined)

    def test_identical_document_excerpt_appears_once_across_history(self) -> None:
        shared_excerpt = "SHARED-DOCUMENT-EXCERPT"
        first = self._symbol(
            "S0001",
            source="FIRST-SOURCE",
            document_excerpt=shared_excerpt,
            start=1,
        )
        second = self._symbol(
            "S0002",
            source="SECOND-SOURCE",
            document_excerpt=shared_excerpt,
            start=10,
        )
        history = [
            HistoryEntry(1, "read-first", {}, self._read(first)),
            HistoryEntry(2, "read-second", {}, self._read(second)),
        ]

        joined = self._joined(self._prepare(history))

        self.assertEqual(joined.count(shared_excerpt), 1)
        self.assertNotIn("S0001-BEHAVIOR-EVIDENCE", joined)
        self.assertNotIn("S0002-BEHAVIOR-EVIDENCE", joined)

    def test_run_state_lists_read_ids_without_an_unread_checklist(self) -> None:
        history = [
            HistoryEntry(
                1,
                "read-one",
                {},
                self._read(self._symbol("S0002", source="READ-SOURCE")),
            )
        ]
        priority_groups = {
            "Authentication": ("S0001", "S0002", "S0003"),
            "Caching": ("S0004",),
        }

        joined = self._joined(
            self._prepare(history, priority_groups=priority_groups)
        )

        self.assertIn("Successfully read repository units: 1", joined)
        self.assertNotIn("Remaining rounds including finish", joined)
        self.assertNotIn("default next action is finish", joined)
        self.assertIn("choose exactly one of search, read, or finish", joined)
        self.assertIn("Unread index entries are optional, not a checklist", joined)
        self.assertNotIn("Authentication", joined)
        self.assertNotIn("Caching", joined)
        self.assertNotIn("read, 2 unread", joined)
        self.assertNotIn("Action ledger", joined)

    def test_run_state_ends_with_the_current_round_number(self) -> None:
        first = self._joined(self._prepare([]))
        history = [
            HistoryEntry(
                1,
                "read-one",
                {},
                self._read(self._symbol("S0002", source="READ-SOURCE")),
            )
        ]
        second = self._joined(self._prepare(history))

        self.assertTrue(first.endswith("[[END RUN STATE]]\nRound 1"))
        self.assertTrue(second.endswith("[[END RUN STATE]]\nRound 2"))

    def test_search_after_read_does_not_prematurely_prompt_finish(self) -> None:
        history = [
            HistoryEntry(
                1,
                "read-one",
                {},
                self._read(self._symbol("S0002", source="READ-SOURCE")),
            ),
            HistoryEntry(
                2,
                "search-next",
                {},
                ToolResult(action="search", status="ok", search_text="dependency"),
            ),
        ]

        joined = self._joined(self._prepare(history))

        self.assertNotIn("default next action is finish", joined)

    def test_forced_compression_never_drops_any_unique_read_source(self) -> None:
        history = [
            HistoryEntry(
                1,
                "read-old",
                {},
                self._read(
                    self._symbol(
                        "S0001",
                        source="OLD-SOURCE " * 80,
                        evidence_text="OLD-EVIDENCE",
                    )
                ),
            ),
            HistoryEntry(
                2,
                "read-latest",
                {},
                self._read(
                    self._symbol(
                        "S0002",
                        source="LATEST-SOURCE " * 80,
                        evidence_text="LATEST-EVIDENCE",
                        start=10,
                    )
                ),
            ),
        ]

        prepared = self._prepare(history, force_compression=True)
        joined = self._joined(prepared)

        self.assertTrue(prepared.manifest.forced)
        self.assertTrue(prepared.manifest.working_compression_triggered)
        self.assertNotIn((1, "S0001"), prepared.manifest.read_receipt_units)
        self.assertNotIn((2, "S0002"), prepared.manifest.read_receipt_units)
        self.assertIn("OLD-SOURCE", joined)
        self.assertNotIn("OLD-EVIDENCE", joined)
        self.assertIn("LATEST-SOURCE", joined)
        self.assertNotIn("LATEST-EVIDENCE", joined)

    def test_protected_unique_evidence_may_remain_above_target(self) -> None:
        history = [
            HistoryEntry(
                1,
                "read-unique",
                {},
                self._read(self._symbol("S0001", source="PROTECTED-UNIQUE-SOURCE")),
            )
        ]
        config = replace(
            self.config,
            working_prompt_trigger=2,
            compression_target=1,
        )

        prepared = prepare_messages(
            system="system",
            initial_user="initial",
            history=history,
            max_rounds=6,
            config=config,
            counter=self.counter,
        )
        joined = self._joined(prepared)

        self.assertTrue(prepared.manifest.working_compression_triggered)
        self.assertTrue(prepared.manifest.protected_oversize)
        self.assertIn("PROTECTED-UNIQUE-SOURCE", joined)
        self.assertEqual(prepared.manifest.read_receipt_units, ())

    def _prepare(
        self,
        history: list[HistoryEntry],
        *,
        system: str = "system",
        initial_user: str = "initial",
        force_compression: bool = False,
        priority_groups: dict[str, tuple[str, ...]] | None = None,
    ):
        return prepare_messages(
            system=system,
            initial_user=initial_user,
            history=history,
            max_rounds=6,
            config=self.config,
            counter=self.counter,
            force_compression=force_compression,
            priority_groups=priority_groups,
        )

    @staticmethod
    def _joined(prepared) -> str:
        return "\n".join(message["content"] for message in prepared.messages)

    @staticmethod
    def _read(*units: EvidenceUnit) -> ToolResult:
        ids = tuple(unit.unit_id for unit in units)
        return ToolResult(
            action="read",
            status="ok",
            requested_ids=ids,
            units=tuple(units),
        )

    @staticmethod
    def _symbol(
        unit_id: str,
        *,
        source: str,
        path: str = "pkg/api.py",
        symbol: str | None = None,
        start: int = 1,
        end: int | None = None,
        document_excerpt: str | None = None,
        evidence_text: str | None = None,
    ) -> EvidenceUnit:
        number = int(unit_id[1:])
        symbol_name = symbol or f"symbol_{number}"
        range_end = end if end is not None else start + 2
        return EvidenceUnit(
            unit_id=unit_id,
            unit_kind="symbol",
            name=f"{path}::{symbol_name}",
            span=EvidenceSpan(path, symbol_name, start, range_end),
            source=source,
            behavior_evidence=(
                BehaviorEvidence(
                    labels=("condition", "result"),
                    start=start,
                    end=min(start + 1, range_end),
                    text=evidence_text or f"{unit_id}-BEHAVIOR-EVIDENCE",
                ),
            ),
            document_title="CURRENT DOCUMENT EXCERPT",
            document_section="API contract",
            document_excerpt=document_excerpt or f"{unit_id}-DOCUMENT-EXCERPT",
            code_title="CURRENT CODE",
        )


if __name__ == "__main__":
    unittest.main()
