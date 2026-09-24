from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.feedbacktrace.scripts.build_feedbacktrace import (
    EVIDENCE_UNIT_POLICY,
    EXTRACTED_TRAJECTORY_SCHEMA,
    extract_build_candidate_trajectory,
    extract_build_trajectory_layers,
    extract_split_assistant_response,
    validate_extracted_trajectory_output,
    validate_python_filter_output,
    write_extracted_trajectory_jsonl,
)


def source_row(
    turn_id: str,
    turn_number: int,
    turn_type: str,
    content: str,
    *,
    role: str = "assistant",
    tool_call_id: str | None = None,
    tool_name: str | None = None,
    tool_input_json: str | None = None,
) -> dict[str, object]:
    return {
        "turn_id": turn_id,
        "session_id": "session-1",
        "turn_number": turn_number,
        "role": role,
        "turn_type": turn_type,
        "content": content,
        "tool_name": tool_name,
        "tool_call_id": tool_call_id,
        "tool_input_json": tool_input_json,
        "file_path": None,
        "command": None,
        "pattern": None,
    }


def build_layers(
    rows: list[dict[str, object]],
    *,
    local_start: int = 0,
    cutoff: int = 100,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, int]]:
    return extract_build_trajectory_layers(
        rows,
        session_id="session-1",
        local_start_turn_number=local_start,
        cutoff_turn_number=cutoff,
    )


class ExtractEvidenceLayerTests(unittest.TestCase):
    def test_tool_use_and_result_remain_in_timeline_but_share_one_exchange(self) -> None:
        rows = [
            source_row(
                "use-1",
                2,
                "tool_use",
                "",
                tool_call_id="call-1",
                tool_name="Write",
                tool_input_json='{"file_path":"a.py","content":"x = 1"}',
            ),
            source_row(
                "result-1",
                5,
                "tool_result",
                "File written",
                tool_call_id="call-1",
            ),
        ]

        events, evidence_units, stats = build_layers(rows)

        self.assertEqual(
            [event["event_type"] for event in events],
            ["tool_use", "tool_result"],
        )
        self.assertTrue(all("evidence_id" not in event for event in events))
        self.assertEqual(len(evidence_units), 1)
        exchange = evidence_units[0]
        self.assertEqual(exchange["evidence_type"], "tool_exchange")
        self.assertEqual(exchange["evidence_id"], "e_2_tool_exchange_0")
        self.assertEqual(exchange["tool_use"]["event_id"], events[0]["event_id"])
        self.assertEqual(
            exchange["tool_result"]["event_id"], events[1]["event_id"]
        )
        self.assertEqual(stats["complete_tool_exchange_count"], 1)

    def test_nonadjacent_interleaved_calls_pair_by_call_id(self) -> None:
        rows = [
            source_row(
                "use-a", 1, "tool_use", "", tool_call_id="a", tool_name="Read"
            ),
            source_row(
                "use-b", 2, "tool_use", "", tool_call_id="b", tool_name="Bash"
            ),
            source_row("result-b", 3, "tool_result", "B", tool_call_id="b"),
            source_row("assistant", 4, "assistant_response", "Still working."),
            source_row("result-a", 5, "tool_result", "A", tool_call_id="a"),
        ]

        events, evidence_units, _ = build_layers(rows)

        exchanges = {
            unit["tool_name"]: unit
            for unit in evidence_units
            if unit["evidence_type"] == "tool_exchange"
        }
        self.assertEqual(exchanges["Read"]["tool_result"]["turn_number"], 5)
        self.assertEqual(exchanges["Bash"]["tool_result"]["turn_number"], 3)
        self.assertEqual(
            [event["turn_number"] for event in events], [1, 2, 3, 4, 5]
        )

    def test_same_turn_tool_calls_receive_stable_distinct_ids(self) -> None:
        rows = [
            source_row(
                "use-a", 2, "tool_use", "", tool_call_id="a", tool_name="Read"
            ),
            source_row(
                "use-b", 2, "tool_use", "", tool_call_id="b", tool_name="Read"
            ),
            source_row("result-a", 3, "tool_result", "A", tool_call_id="a"),
            source_row("result-b", 4, "tool_result", "B", tool_call_id="b"),
        ]

        _, evidence_units, _ = build_layers(rows)

        self.assertEqual(
            [unit["evidence_id"] for unit in evidence_units],
            ["e_2_tool_exchange_0", "e_2_tool_exchange_1"],
        )

    def test_orphan_result_is_dropped_and_unfinished_use_is_retained(self) -> None:
        rows = [
            source_row(
                "use-1",
                1,
                "tool_use",
                "",
                tool_call_id="known",
                tool_name="Edit",
            ),
            source_row(
                "orphan", 2, "tool_result", "orphan", tool_call_id="missing"
            ),
        ]

        events, evidence_units, stats = build_layers(rows)

        self.assertEqual([event["event_type"] for event in events], ["tool_use"])
        self.assertIsNone(evidence_units[0]["tool_result"])
        self.assertEqual(stats["orphan_tool_results_dropped"], 1)
        self.assertEqual(stats["incomplete_tool_exchange_count"], 1)

    def test_result_at_cutoff_is_not_attached(self) -> None:
        rows = [
            source_row(
                "use-1", 8, "tool_use", "", tool_call_id="x", tool_name="Bash"
            ),
            source_row("result-1", 10, "tool_result", "done", tool_call_id="x"),
        ]

        events, evidence_units, _ = build_layers(rows, cutoff=10)

        self.assertEqual(len(events), 1)
        self.assertIsNone(evidence_units[0]["tool_result"])
        self.assertNotIn("done", evidence_units[0]["content"])

    def test_local_does_not_inherit_result_for_a_pre_local_call(self) -> None:
        rows = [
            source_row(
                "use-1", 2, "tool_use", "", tool_call_id="x", tool_name="Read"
            ),
            source_row("local-user", 5, "user_prompt", "Next step", role="user"),
            source_row("result-1", 6, "tool_result", "done", tool_call_id="x"),
        ]

        events, evidence_units, _ = build_layers(rows, local_start=5)

        result = next(event for event in events if event["event_type"] == "tool_result")
        exchange = next(
            unit
            for unit in evidence_units
            if unit["evidence_type"] == "tool_exchange"
        )
        self.assertIs(result["in_local"], False)
        self.assertIs(exchange["in_local"], False)

    def test_assistant_response_is_split_only_in_evidence_layer(self) -> None:
        content = "Before.\n\n```python\nx = 1\n\nprint(x)\n```\n\nAfter."
        rows = [source_row("assistant", 3, "assistant_response", content)]

        events, evidence_units, _ = build_layers(rows)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["content"], content)
        self.assertEqual(len(evidence_units), 3)
        self.assertEqual(
            [unit["evidence_id"] for unit in evidence_units],
            [
                "e_3_assistant_response_0",
                "e_3_assistant_response_1",
                "e_3_assistant_response_2",
            ],
        )
        self.assertEqual(extract_split_assistant_response(content)[1], evidence_units[1]["content"])

    def test_candidate_record_contains_one_long_trace_with_local_membership(self) -> None:
        candidate = {
            "target_turn_id": "target",
            "session_id": "session-1",
            "target_turn_number": 9,
            "cutoff_turn_number": 9,
            "window_start_turn_id": "local-user",
            "window_start_turn_number": 4,
        }
        rows = [
            source_row("long-user", 1, "user_prompt", "Start", role="user"),
            source_row("assistant-1", 2, "assistant_response", "Earlier reply"),
            source_row("local-user", 4, "user_prompt", "Current task", role="user"),
            source_row("assistant-2", 5, "assistant_response", "Current reply"),
            source_row("target", 9, "user_prompt", "Correction", role="user"),
        ]

        record, reason, _ = extract_build_candidate_trajectory(
            candidate,
            rows,
            source_revision="revision",
        )

        self.assertIsNone(reason)
        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record["schema"], EXTRACTED_TRAJECTORY_SCHEMA)
        self.assertEqual(
            record["evidence_unit_policy"], EVIDENCE_UNIT_POLICY
        )
        self.assertEqual(
            [event["in_local"] for event in record["events"]],
            [False, False, True, True],
        )
        self.assertNotIn("Correction", str(record["events"]))


class ExtractedTrajectoryPathTests(unittest.TestCase):
    def test_default_outputs_use_sibling_derived_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory).resolve()
            source_dir = temporary_root / "source"
            source_dir.mkdir()

            candidate_path, derived_root = validate_python_filter_output(
                None,
                source_dir,
            )
            trajectory_path = validate_extracted_trajectory_output(
                None,
                source_dir,
                derived_root,
            )

            self.assertEqual(derived_root, temporary_root / "derived")
            self.assertEqual(
                candidate_path,
                derived_root / "python_event_candidates.parquet",
            )
            self.assertEqual(
                trajectory_path,
                derived_root / "python_candidate_trajectories.jsonl",
            )

    def test_content_artifact_must_stay_in_restricted_derived_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory).resolve()
            source_dir = temporary_root / "source"
            work_root = temporary_root / "work"
            source_dir.mkdir()
            work_root.mkdir()
            valid = validate_extracted_trajectory_output(
                work_root / "traces.jsonl",
                source_dir,
                work_root,
            )
            self.assertEqual(valid, (work_root / "traces.jsonl").resolve())
            with self.assertRaisesRegex(Exception, "restricted derived directory"):
                validate_extracted_trajectory_output(
                    temporary_root / "outside.jsonl",
                    source_dir,
                    work_root,
                )

    def test_writer_materializes_the_two_layers_as_restricted_jsonl(self) -> None:
        import duckdb

        connection = duckdb.connect(":memory:")
        self.addCleanup(connection.close)
        connection.execute(
            """
            CREATE TABLE python_event_candidates (
                target_turn_id VARCHAR,
                session_id VARCHAR,
                target_turn_number BIGINT,
                cutoff_turn_number BIGINT,
                window_start_turn_id VARCHAR,
                window_start_turn_number BIGINT
            )
            """
        )
        connection.execute(
            """
            INSERT INTO python_event_candidates
            VALUES ('target', 'session-1', 5, 5, 'user-1', 1)
            """
        )
        connection.execute(
            """
            CREATE TABLE connected_conversations (
                turn_id VARCHAR,
                session_id VARCHAR,
                turn_number BIGINT,
                role VARCHAR,
                turn_type VARCHAR,
                content VARCHAR,
                tool_name VARCHAR,
                tool_call_id VARCHAR,
                tool_input_json VARCHAR,
                file_path VARCHAR,
                command VARCHAR,
                pattern VARCHAR
            )
            """
        )
        connection.executemany(
            """
            INSERT INTO connected_conversations
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    "user-1",
                    "session-1",
                    1,
                    "user",
                    "user_prompt",
                    "Implement it",
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                ),
                (
                    "use-1",
                    "session-1",
                    2,
                    "assistant",
                    "tool_use",
                    "",
                    "Write",
                    "call-1",
                    '{"file_path":"a.py"}',
                    "a.py",
                    None,
                    None,
                ),
                (
                    "result-1",
                    "session-1",
                    3,
                    "tool",
                    "tool_result",
                    "written",
                    None,
                    "call-1",
                    None,
                    None,
                    None,
                    None,
                ),
                (
                    "assistant-1",
                    "session-1",
                    4,
                    "assistant",
                    "assistant_response",
                    "Implemented a.py.",
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                ),
                (
                    "target",
                    "session-1",
                    5,
                    "user",
                    "user_prompt",
                    "Please use another design.",
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                ),
            ],
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            work_root = Path(temporary_directory).resolve()
            output = work_root / "traces.jsonl"
            artifact = write_extracted_trajectory_jsonl(
                connection,
                output,
                work_root,
                source_revision="revision",
            )
            record = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(artifact["row_count"], 1)
        self.assertIs(artifact["contains_conversation_content"], True)
        self.assertEqual(
            [event["event_type"] for event in record["events"]],
            ["user_prompt", "tool_use", "tool_result", "assistant_response"],
        )
        self.assertEqual(
            [unit["evidence_type"] for unit in record["evidence_units"]],
            ["tool_exchange", "assistant_response"],
        )


if __name__ == "__main__":
    unittest.main()
