import json
import unittest

import duckdb

from benchmarks.feedbacktrace.scripts.build_feedbacktrace import (
    clean_explicit_path,
    create_additional_path_evidence,
    create_python_filter_views,
    create_top_level_path_evidence,
    extract_apply_patch_paths,
    extract_controlled_bash_paths,
    extract_structured_tool_paths,
)


def create_tool_event_table(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(
        """
        CREATE TEMP TABLE python_filter_tool_events (
            turn_id VARCHAR,
            session_id VARCHAR,
            turn_number BIGINT,
            turn_type VARCHAR,
            timestamp VARCHAR,
            file_path VARCHAR,
            tool_call_id VARCHAR,
            tool_name VARCHAR,
            tool_input_json VARCHAR,
            command VARCHAR
        )
        """
    )


class StructuredToolPathPolicyTests(unittest.TestCase):
    def test_accepts_only_the_exact_tool_and_key_allowlist(self) -> None:
        self.assertEqual(
            extract_structured_tool_paths(
                "Read", json.dumps({"file_path": "src/reader.py"})
            ),
            ("src/reader.py",),
        )
        self.assertEqual(
            extract_structured_tool_paths(
                "read", json.dumps({"filepath": "src/lowercase.py"})
            ),
            ("src/lowercase.py",),
        )

        # Tool names and field names are deliberately case- and spelling-sensitive.
        self.assertEqual(
            extract_structured_tool_paths(
                "Read", json.dumps({"filepath": "src/wrong_key.py"})
            ),
            (),
        )
        self.assertEqual(
            extract_structured_tool_paths(
                "read", json.dumps({"file_path": "src/wrong_key.py"})
            ),
            (),
        )
        self.assertEqual(
            extract_structured_tool_paths(
                "UnknownRead", json.dumps({"file_path": "src/unknown.py"})
            ),
            (),
        )

    def test_rejects_untrusted_or_non_object_json(self) -> None:
        self.assertEqual(extract_structured_tool_paths("Read", "not json"), ())
        self.assertEqual(extract_structured_tool_paths("Read", "[]"), ())
        self.assertEqual(extract_structured_tool_paths(None, "{}"), ())

    def test_accepts_trusted_absolute_paths_but_rejects_traversal(self) -> None:
        for path in ("/tmp/absolute.py", "C:/workspace/absolute.py"):
            with self.subTest(path=path):
                self.assertEqual(
                    extract_structured_tool_paths(
                        "Read", json.dumps({"file_path": path})
                    ),
                    (path,),
                )
        for path in ("../outside.py", "src/../../outside.py"):
            with self.subTest(path=path):
                self.assertEqual(
                    extract_structured_tool_paths(
                        "Read", json.dumps({"file_path": path})
                    ),
                    (),
                )

    def test_rejects_dynamic_paths_and_non_scalar_allowlist_values(self) -> None:
        for path in ("src/*.py", "$ROOT/app.py", "src/{one,two}.py"):
            with self.subTest(path=path):
                self.assertEqual(
                    extract_structured_tool_paths(
                        "Read", json.dumps({"file_path": path})
                    ),
                    (),
                )
        self.assertEqual(
            extract_structured_tool_paths(
                "Read", json.dumps({"file_path": ["src/app.py"]})
            ),
            (),
        )


class ApplyPatchPathPolicyTests(unittest.TestCase):
    def test_extracts_only_formal_patch_headers(self) -> None:
        patch = """*** Begin Patch
*** Update File: src/existing.py
*** Add File: tests/test_new.py
*** Delete File: old/retired.py
*** Move to: src/moved.py
This prose mentions src/not_a_header.py but is not a header.
 *** Update File: src/indented_header.py
*** End Patch
"""
        self.assertEqual(
            extract_apply_patch_paths(patch),
            (
                "old/retired.py",
                "src/existing.py",
                "src/moved.py",
                "tests/test_new.py",
            ),
        )

    def test_extracts_standard_diff_headers_and_strips_a_b_prefixes(self) -> None:
        patch = """diff --git a/pkg/old.py b/pkg/new.py
--- a/pkg/old.py
+++ b/pkg/new.py
"""
        self.assertEqual(
            extract_apply_patch_paths(patch),
            ("pkg/new.py", "pkg/old.py"),
        )

    def test_accepts_absolute_patch_headers_but_rejects_traversal(self) -> None:
        patch = """*** Update File: /tmp/absolute.py
*** Add File: C:/workspace/absolute.py
*** Delete File: ../outside.py
*** Move to: src/../../outside.py
"""
        self.assertEqual(
            extract_apply_patch_paths(patch),
            ("/tmp/absolute.py", "C:/workspace/absolute.py"),
        )


class ControlledBashPathPolicyTests(unittest.TestCase):
    def assertAccepted(self, command: str, expected: tuple[str, ...]) -> None:
        paths, status = extract_controlled_bash_paths(command)
        self.assertEqual(paths, expected, command)
        self.assertEqual(status, "accepted_manual_cwd_review", command)

    def assertRejected(self, command: str) -> None:
        paths, status = extract_controlled_bash_paths(command)
        self.assertEqual(paths, (), command)
        self.assertNotEqual(status, "accepted_manual_cwd_review", command)

    def test_accepts_controlled_commands_with_explicit_code_paths(self) -> None:
        cases = {
            "python scripts/check.py": ("scripts/check.py",),
            "python -m pytest tests/test_app.py": ("tests/test_app.py",),
            "pytest tests/test_app.py::test_feature": ("tests/test_app.py",),
            "ruff check src/app.py": ("src/app.py",),
            "cat src/app.py": ("src/app.py",),
            "git diff -- src/app.py tests/test_app.py": (
                "src/app.py",
                "tests/test_app.py",
            ),
            "grep needle -- src/app.py": ("src/app.py",),
        }
        for command, expected in cases.items():
            with self.subTest(command=command):
                self.assertAccepted(command, expected)

    def test_rejects_dynamic_or_composed_shell_syntax(self) -> None:
        for command in (
            "cat $TARGET.py",
            "cat src/*.py",
            "cat src/app.py | wc -l",
            "cat src/app.py && pytest tests/test_app.py",
            "cat src/app.py; pytest tests/test_app.py",
            "cat $(find src -name app.py)",
            "cat `find src -name app.py`",
            "cat src/app.py # fake.py",
        ):
            with self.subTest(command=command):
                self.assertRejected(command)

    def test_rejects_python_inline_code(self) -> None:
        self.assertRejected("python -c 'open(\"src/app.py\")'")

    def test_rejects_absolute_and_parent_traversal_paths(self) -> None:
        for command in (
            "cat /tmp/absolute.py",
            "cat C:/workspace/absolute.py",
            "cat ../outside.py",
            "cat src/../../outside.py",
            "cat ~someone/outside.py",
        ):
            with self.subTest(command=command):
                self.assertRejected(command)

    def test_requires_separator_for_git_and_search_commands(self) -> None:
        for command in (
            "git diff src/app.py",
            "git status src/app.py",
            "grep needle src/app.py",
            "rg needle src/app.py",
            "sed pattern src/app.py",
            "grep -- pattern.py src/app.py",
        ):
            with self.subTest(command=command):
                self.assertRejected(command)

    def test_ignores_controlled_commands_without_an_explicit_code_file(self) -> None:
        self.assertRejected("cat README.md")
        self.assertRejected("pytest tests")

    def test_rejects_command_specific_ambiguous_shapes(self) -> None:
        for command in (
            "cat src/app.py::fake_node",
            "cp src/app.py",
            "mv src/app.py dst/app.py extra.py",
            "diff src/app.py",
        ):
            with self.subTest(command=command):
                self.assertRejected(command)


class ExplicitPathCleanerTests(unittest.TestCase):
    def test_repo_relative_policy_rejects_absolute_and_parent_paths(self) -> None:
        for path in (
            "/tmp/absolute.py",
            "C:/workspace/absolute.py",
            "~/absolute.py",
            "~someone/absolute.py",
            "../outside.py",
            "src/../../outside.py",
            "src/*.py",
            "$ROOT/app.py",
        ):
            with self.subTest(path=path):
                self.assertIsNone(
                    clean_explicit_path(
                        path,
                        require_code_extension=True,
                        require_repo_relative=True,
                    )
                )


class BashTopLevelIsolationTests(unittest.TestCase):
    def test_bash_tool_and_paired_result_are_not_automatic_paths(self) -> None:
        connection = duckdb.connect(":memory:")
        try:
            create_tool_event_table(connection)
            connection.executemany(
                """
                INSERT INTO python_filter_tool_events VALUES
                    (?, 'session', ?, ?, '2026-01-01T00:00:00Z', ?, ?, ?, '{}', ?)
                """,
                [
                    ("read-use", 1, "tool_use", "src/read.py", "read-1", "Read", None),
                    ("read-result", 2, "tool_result", "src/read.py", "read-1", None, None),
                    ("bash-use", 3, "tool_use", "src/bash.py", "bash-1", "Bash", "cat src/bash.py"),
                    ("bash-result", 4, "tool_result", "src/bash.py", "bash-1", None, None),
                ],
            )
            stats = create_top_level_path_evidence(connection)
            retained = connection.execute(
                """
                SELECT turn_id
                FROM python_filter_top_level_path_evidence
                ORDER BY turn_id
                """
            ).fetchall()
            self.assertEqual(retained, [("read-result",), ("read-use",)])
            self.assertEqual(stats["top_level_nonempty_path_rows"], 4)
            self.assertEqual(stats["top_level_literal_path_rows"], 2)
        finally:
            connection.close()

    def test_bash_path_equal_to_top_level_stays_manual_review_evidence(self) -> None:
        connection = duckdb.connect(":memory:")
        try:
            create_tool_event_table(connection)
            payload = json.dumps({"command": "cat src/bash.py"})
            connection.execute(
                """
                INSERT INTO python_filter_tool_events VALUES (
                    'bash-use', 'session', 1, 'tool_use',
                    '2026-01-01T00:00:00Z', 'src/bash.py', 'bash-1',
                    'Bash', ?, 'cat src/bash.py'
                )
                """,
                [payload],
            )
            create_additional_path_evidence(connection)
            evidence = connection.execute(
                """
                SELECT evidence_source, file_path, auto_eligible
                FROM python_filter_additional_path_evidence
                """
            ).fetchall()
            self.assertEqual(
                evidence,
                [("controlled_bash_argument", "src/bash.py", False)],
            )
        finally:
            connection.close()


class PythonWindowSelectionPolicyTests(unittest.TestCase):
    def test_python_majority_applies_to_any_repository_language(self) -> None:
        connection = duckdb.connect(":memory:")
        try:
            connection.execute(
                """
                CREATE TEMP TABLE raw_repositories (
                    repo_id VARCHAR,
                    repo_github_metadata VARCHAR
                );
                CREATE TEMP TABLE connected_conversations (
                    turn_id VARCHAR,
                    session_id VARCHAR,
                    repo_id VARCHAR,
                    turn_number BIGINT,
                    conversation_turn_number BIGINT,
                    role VARCHAR,
                    turn_type VARCHAR,
                    prompt_pushback VARCHAR,
                    file_path VARCHAR,
                    tool_call_id VARCHAR,
                    tool_name VARCHAR,
                    tool_input_json VARCHAR,
                    command VARCHAR
                )
                """
            )
            connection.executemany(
                "INSERT INTO raw_repositories VALUES (?, ?)",
                [
                    ("nonpy-majority", json.dumps({"language": "TypeScript"})),
                    ("nonpy-half", json.dumps({"language": "TypeScript"})),
                    ("python-repo", json.dumps({"language": "Python"})),
                    ("missing-language", json.dumps({})),
                ],
            )

            def add_window(repo_id: str, paths: list[str]) -> None:
                session_id = f"session-{repo_id}"
                rows = [
                    (
                        f"start-{repo_id}", session_id, repo_id, 1, 1,
                        "user", "user_prompt", None, None, None, None,
                        None, None,
                    )
                ]
                for index, path in enumerate(paths, start=2):
                    rows.append(
                        (
                            f"tool-{repo_id}-{index}", session_id, repo_id,
                            index, index, "assistant", "tool_use", None,
                            path, f"call-{repo_id}-{index}", "Read", "{}",
                            None,
                        )
                    )
                target_turn = len(paths) + 2
                rows.append(
                    (
                        f"target-{repo_id}", session_id, repo_id,
                        target_turn, target_turn, "user", "user_prompt",
                        "correction", None, None, None, None, None,
                    )
                )
                connection.executemany(
                    "INSERT INTO connected_conversations VALUES "
                    "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    rows,
                )

            add_window("nonpy-majority", ["a.py", "b.pyi", "c.ts"])
            add_window("nonpy-half", ["a.py", "b.ts"])
            add_window("python-repo", ["a.py", "b.ts", "c.ts"])
            add_window("missing-language", ["a.py", "b.py", "c.go"])

            create_python_filter_views(connection)
            decisions = {
                target_id: (rule, passed)
                for target_id, rule, passed in connection.execute(
                    """
                    SELECT target_turn_id, python_filter_rule, python_filter_pass
                    FROM python_filter_decisions
                    """
                ).fetchall()
            }
            self.assertEqual(
                decisions["target-nonpy-majority"],
                ("python_file_majority", True),
            )
            self.assertEqual(
                decisions["target-nonpy-half"],
                ("non_python_repository", False),
            )
            self.assertEqual(
                decisions["target-python-repo"],
                ("python_repo_with_python_file", True),
            )
            self.assertEqual(
                decisions["target-missing-language"],
                ("python_file_majority", True),
            )
        finally:
            connection.close()


if __name__ == "__main__":
    unittest.main()
