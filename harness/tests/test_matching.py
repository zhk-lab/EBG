from __future__ import annotations

import unittest

from codex_harness.application.matching import direct_repo_roots, trace_signal
from codex_harness.application.repository import module_statement_span


class HarnessMatchingTests(unittest.TestCase):
    def test_prose_after_closing_backtick_does_not_match_method(self):
        graph = {"behaviors": [{"path": "sessions.py", "symbol": "SessionLog.root",
                                "symbol_lines": [1, 2]}], "evidence": []}
        text = "只选择 `calls/feeds` 且有 Root Behavior Evidence 支撑的关系。"
        self.assertEqual(direct_repo_roots(graph, text), {})
        for text in ("Read `root`.", "Call root().", "Use SessionLog.root."):
            with self.subTest(text=text):
                self.assertEqual(direct_repo_roots(graph, text), {"sessions.py": ("SessionLog.root",)})

    def test_plain_top_level_word_is_not_an_explicit_code_anchor(self):
        graph = {"behaviors": [{"path": "app.py", "symbol": "read", "symbol_lines": [1, 2]}],
                 "evidence": []}
        self.assertEqual(direct_repo_roots(graph, "Users read the document."), {})

    def test_navigation_fallback_is_not_a_requirement_match(self):
        graph = {
            "behaviors": [{"path": "app.py", "symbol": "run", "symbol_lines": [1, 2]}],
            "evidence": [],
        }
        self.assertEqual(direct_repo_roots(graph, "Keep existing behavior."), {})
        self.assertEqual(direct_repo_roots(graph, "Change `run`."), {"app.py": ("run",)})

    def test_ambiguous_symbol_does_not_reenter_through_navigation_fallback(self):
        graph = {"behaviors": [{"path": path, "symbol": "run", "symbol_lines": [1, 2]}
                               for path in ("app.py", "other.py")], "evidence": []}
        self.assertEqual(direct_repo_roots(graph, "Change `run`."), {})

    def test_codex_patch_is_a_modify_action(self):
        signal = trace_signal('{"command":"*** Begin Patch\\n*** Update File: src/app.py\\n*** End Patch"}', tool_name="apply_patch")
        self.assertEqual(signal["operations"], ["modify"])
        self.assertIn("src/app.py", signal["objects"])

    def test_shell_verification_uses_command_not_incidental_result_text(self):
        signal = trace_signal('{"command":"pytest tests/test_app.py"}', tool_name="Bash")
        self.assertEqual(signal["operations"], ["verify"])
        self.assertIn("tests/test_app.py", signal["objects"])

    def test_reading_a_test_file_is_not_verification(self):
        signal = trace_signal('{"path":"tests/test_app.py"}', tool_name="Read")
        self.assertEqual(signal["operations"], ["inspect"])

    def test_chinese_run_with_test_path_is_verification(self):
        signal = trace_signal("运行 tests/test_retry.py，确认重试次数。")
        self.assertIn("verify", signal["operations"])

    def test_adapter_does_not_change_ebg_tool_classification(self):
        from codex_harness.ebg.behavior_atomization import _trace_tool_operation

        item = {"content": "*** Update File: app.py", "locator": {"event_type": "tool_exchange", "tool_name": "apply_patch"}}
        before = _trace_tool_operation(item)
        self.assertEqual(trace_signal(item["content"], tool_name="apply_patch")["operations"], ["modify"])
        self.assertEqual(_trace_tool_operation(item), before)
        self.assertEqual(before, "execute")

    def test_module_excerpt_preserves_condition_and_alternative_definition(self):
        code = "if DEBUG:\n    LIMIT = 2\nelse:\n    LIMIT = 3\n\ndef unrelated():\n    return 1\n"
        self.assertEqual(module_statement_span("app.py", code, 2, 2), (1, 4))


if __name__ == "__main__":
    unittest.main()
