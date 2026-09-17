from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ebg.behavior_atomization import build_behaviors, validate_behaviors
from ebg.core.errors import BehaviorError
from ebg.evidence_intake import build_evidence, load_visible_bundle
from tests.support import ProjectTemporaryDirectory, make_repo_bundle, make_trace_bundle


SCHEMA = json.loads(
    (PROJECT_ROOT / "schemas" / "behaviors.schema.json").read_text(encoding="utf-8")
)
VALIDATE_BEHAVIORS = Draft202012Validator(SCHEMA).validate


SOURCE = """import requests

def choose(flag):
    if flag:
        value = requests.get("https://example.test")
        print(value)
        return value
    raise ValueError("bad")

class Box:
    def set(self, value):
        self.value = value
        return self.value

def gen():
    yield 1
    yield 2
"""


class BehaviorAtomizationTests(unittest.TestCase):
    def test_class_attributes_are_state_writes_but_function_locals_are_not(self) -> None:
        source = """class Visitor:
    required_fields = ['name']
    enabled: bool = True
    def visit(self, value):
        local = value
        return local
    visit_async = visit
"""
        with ProjectTemporaryDirectory() as temporary:
            bundle = load_visible_bundle(
                make_repo_bundle(temporary, repository_files={"visitor.py": source})
            )
            evidence = build_evidence(bundle)
            behaviors = build_behaviors(bundle, evidence)
            validate_behaviors(bundle, evidence, behaviors)
            by_id = {item["evidence_id"]: item for item in evidence}
            writes = [item for item in behaviors if item["result_type"] == "state_write"]
            self.assertEqual(len(writes), 3)
            self.assertEqual({item["symbol"] for item in writes}, {"Visitor"})
            text = "".join(
                by_id[eid]["content"]
                for item in writes
                for eid in item["result_evidence_ids"]
            )
            for field in ("required_fields", "enabled", "visit_async"):
                self.assertIn(field, text)
            self.assertNotIn("local =", text)

    def test_repo_behaviors_use_role_schema_and_every_result_is_anchored(self) -> None:
        with ProjectTemporaryDirectory() as temporary:
            bundle = load_visible_bundle(
                make_repo_bundle(temporary, repository_files={"src/app.py": SOURCE})
            )
            evidence = build_evidence(bundle)
            behaviors = build_behaviors(bundle, evidence)
            VALIDATE_BEHAVIORS(behaviors)

            self.assertEqual(
                {item["result_type"] for item in behaviors},
                {"return", "raise", "state_write", "output", "external_call", "yield"},
            )
            self.assertTrue(all("behavior_name" not in item and "l3_ids" not in item for item in behaviors))
            by_id = {item["evidence_id"]: item for item in evidence}
            for behavior in behaviors:
                self.assertTrue(behavior["result_evidence_ids"])
                role_ids = [
                    *behavior["trigger_evidence_ids"],
                    *behavior["operation_evidence_ids"],
                    *behavior["result_evidence_ids"],
                ]
                self.assertEqual(len(role_ids), len(set(role_ids)))
                self.assertTrue(
                    all(by_id[evidence_id]["locator"]["symbol"] == behavior["symbol"] for evidence_id in role_ids)
                )

            choose_return = next(
                item
                for item in behaviors
                if item["symbol"] == "choose" and item["result_type"] == "return"
            )
            trigger_text = "".join(by_id[item]["content"] for item in choose_return["trigger_evidence_ids"])
            operation_text = "".join(by_id[item]["content"] for item in choose_return["operation_evidence_ids"])
            result_text = "".join(by_id[item]["content"] for item in choose_return["result_evidence_ids"])
            self.assertIn("if flag", trigger_text)
            self.assertIn("requests.get", operation_text)
            self.assertNotIn("print(value)", operation_text)
            self.assertIn("return value", result_text)

    def test_mutually_exclusive_routes_to_one_return_become_two_behaviors(self) -> None:
        source = """def choose(flag):
    if flag:
        value = 1
    else:
        value = 2
    return value
"""
        with ProjectTemporaryDirectory() as temporary:
            bundle = load_visible_bundle(
                make_repo_bundle(temporary, repository_files={"src/routes.py": source})
            )
            evidence = build_evidence(bundle)
            behaviors = build_behaviors(bundle, evidence)
            returns = [item for item in behaviors if item["result_type"] == "return"]
            self.assertEqual(len(returns), 2)
            by_id = {item["evidence_id"]: item for item in evidence}
            operations = [
                "".join(by_id[evidence_id]["content"] for evidence_id in item["operation_evidence_ids"])
                for item in returns
            ]
            self.assertEqual(sum("value = 1" in text for text in operations), 1)
            self.assertEqual(sum("value = 2" in text for text in operations), 1)
            self.assertEqual(
                {tuple(item["result_evidence_ids"]) for item in returns},
                {("E000006",)},
            )

    def test_implicit_else_route_keeps_only_the_reaching_definition(self) -> None:
        source = """def choose(flag):
    value = 0
    if flag:
        value = 1
    return value
"""
        with ProjectTemporaryDirectory() as temporary:
            bundle = load_visible_bundle(
                make_repo_bundle(temporary, repository_files={"src/default.py": source})
            )
            evidence = build_evidence(bundle)
            behaviors = build_behaviors(bundle, evidence)

        returns = [item for item in behaviors if item["result_type"] == "return"]
        self.assertEqual(len(returns), 2)
        by_id = {item["evidence_id"]: item["content"] for item in evidence}
        operation_texts = [
            "".join(by_id[evidence_id] for evidence_id in item["operation_evidence_ids"])
            for item in returns
        ]
        self.assertEqual(sum("value = 0" in text for text in operation_texts), 1)
        self.assertEqual(sum("value = 1" in text for text in operation_texts), 1)
        self.assertTrue(
            all(
                any(
                    "if flag:" in by_id[evidence_id]
                    for evidence_id in item["trigger_evidence_ids"]
                )
                for item in returns
            )
        )

    def test_calls_in_mutually_exclusive_branches_do_not_mix(self) -> None:
        source = """def report(enabled):
    if enabled:
        print("on")
    else:
        print("off")
"""
        with ProjectTemporaryDirectory() as temporary:
            bundle = load_visible_bundle(
                make_repo_bundle(temporary, repository_files={"src/report.py": source})
            )
            evidence = build_evidence(bundle)
            behaviors = build_behaviors(bundle, evidence)

        outputs = [item for item in behaviors if item["result_type"] == "output"]
        self.assertEqual(len(outputs), 2)
        by_id = {item["evidence_id"]: item["content"].strip() for item in evidence}
        for behavior in outputs:
            referenced = {
                by_id[evidence_id]
                for field in (
                    "trigger_evidence_ids",
                    "operation_evidence_ids",
                    "result_evidence_ids",
                )
                for evidence_id in behavior[field]
            }
            self.assertIn("if enabled:", referenced)
            self.assertEqual(len(referenced & {'print("on")', 'print("off")'}), 1)

    def test_call_in_condition_does_not_absorb_branch_body(self) -> None:
        source = """import requests

def reachable(url):
    message = "reachable"
    if requests.get(url):
        print(message)
"""
        with ProjectTemporaryDirectory() as temporary:
            bundle = load_visible_bundle(
                make_repo_bundle(temporary, repository_files={"src/check.py": source})
            )
            evidence = build_evidence(bundle)
            behaviors = build_behaviors(bundle, evidence)

        external = next(
            item
            for item in behaviors
            if item["symbol"] == "reachable" and item["result_type"] == "external_call"
        )
        by_id = {item["evidence_id"]: item["content"] for item in evidence}
        referenced = "".join(by_id[evidence_id] for evidence_id in external["operation_evidence_ids"])
        self.assertNotIn('message = "reachable"', referenced)
        self.assertNotIn("print(message)", referenced)

    def test_import_inside_symbol_supports_external_call_anchor(self) -> None:
        source = """def fetch(url):
    import requests
    return requests.get(url)
"""
        with ProjectTemporaryDirectory() as temporary:
            bundle = load_visible_bundle(
                make_repo_bundle(temporary, repository_files={"src/local_import.py": source})
            )
            evidence = build_evidence(bundle)
            behaviors = build_behaviors(bundle, evidence)

        self.assertTrue(
            any(
                item["symbol"] == "fetch" and item["result_type"] == "external_call"
                for item in behaviors
            )
        )

    def test_behavior_never_crosses_symbol_boundary(self) -> None:
        source = """def outer(flag):
    if flag:
        return inner()
    raise RuntimeError("stop")

def inner():
    return 1
"""
        with ProjectTemporaryDirectory() as temporary:
            bundle = load_visible_bundle(
                make_repo_bundle(temporary, repository_files={"src/scopes.py": source})
            )
            evidence = build_evidence(bundle)
            behaviors = build_behaviors(bundle, evidence)
            by_id = {item["evidence_id"]: item for item in evidence}
            for behavior in behaviors:
                referenced_symbols = {
                    by_id[evidence_id]["locator"]["symbol"]
                    for key in ("trigger_evidence_ids", "operation_evidence_ids", "result_evidence_ids")
                    for evidence_id in behavior[key]
                }
                self.assertEqual(referenced_symbols, {behavior["symbol"]})

    def test_evidence_without_observable_result_does_not_create_fake_behavior(self) -> None:
        with ProjectTemporaryDirectory() as temporary:
            bundle = load_visible_bundle(
                make_repo_bundle(
                    temporary,
                    repository_files={"src/imports.py": "import os\nfrom pathlib import Path\n"},
                )
            )
            evidence = build_evidence(bundle)
            behaviors = build_behaviors(bundle, evidence)
            self.assertEqual(behaviors, [])
            VALIDATE_BEHAVIORS(behaviors)

    def test_same_result_evidence_does_not_create_duplicate_call_behaviors(self) -> None:
        source = """import httpx
import requests

def fetch():
    return requests.post("https://example.test", data=httpx.get("https://source.test"))
"""
        with ProjectTemporaryDirectory() as temporary:
            bundle = load_visible_bundle(
                make_repo_bundle(temporary, repository_files={"src/fetch.py": source})
            )
            evidence = build_evidence(bundle)
            behaviors = build_behaviors(bundle, evidence)
            external = [item for item in behaviors if item["result_type"] == "external_call"]
            self.assertEqual(len(external), 1)
            self.assertEqual(
                external[0]["result_evidence_ids"],
                next(item["result_evidence_ids"] for item in behaviors if item["result_type"] == "return"),
            )

    def test_configuration_and_template_results_remain_source_backed(self) -> None:
        with ProjectTemporaryDirectory() as temporary:
            bundle = load_visible_bundle(
                make_repo_bundle(
                    temporary,
                    repository_files={
                        "settings.toml": "[runtime]\nmode = 'safe'\n",
                        "templates/page.jinja2": "{% block body %}\nHello {{ name }}\n{% endblock %}\n",
                    },
                )
            )
            evidence = build_evidence(bundle)
            behaviors = build_behaviors(bundle, evidence)
            self.assertTrue(
                any(item["artifact_kind"] == "configuration" and item["result_type"] == "state_write" for item in behaviors)
            )
            template = next(
                item
                for item in behaviors
                if item["artifact_kind"] == "runtime_template"
                and item["result_type"] == "output"
            )
            self.assertEqual(template["symbol"], "body")
            self.assertEqual(template["symbol_lines"], [1, 3])
            validate_behaviors(bundle, evidence, behaviors)

    def test_trace_interactions_keep_system_context_outside_user_trigger(self) -> None:
        events = [
            {"event_type": "system", "turn_number": 0, "content": "context"},
            {"event_type": "user_prompt", "turn_number": 1, "content": "Change it."},
            {"event_type": "assistant_response", "turn_number": 2, "content": "Working."},
            {
                "event_type": "tool_exchange",
                "turn_number": 3,
                "content": "write result",
                "evidence_id": "raw-E3",
                "tool_name": "write_file",
            },
            {"event_type": "assistant_response", "turn_number": 4, "content": "Done.", "evidence_id": "raw-E4"},
            {"event_type": "user_prompt", "turn_number": 5, "content": "Verify it."},
            {"event_type": "assistant_response", "turn_number": 6, "content": "Verified.", "evidence_id": "raw-E6"},
        ]
        with ProjectTemporaryDirectory() as temporary:
            bundle = load_visible_bundle(make_trace_bundle(temporary, events, cutoff=7))
            evidence = build_evidence(bundle)
            behaviors = build_behaviors(bundle, evidence)
            VALIDATE_BEHAVIORS(behaviors)
            self.assertEqual([item["behavior_id"] for item in behaviors], ["T0001", "T0002"])
            self.assertEqual([item["task_id"] for item in behaviors], ["K0001", "K0001"])
            self.assertEqual([item["sequence_index"] for item in behaviors], [1, 2])
            self.assertEqual(behaviors[0]["start_turn"], 1)
            self.assertEqual(behaviors[0]["end_turn"], 4)
            self.assertEqual(behaviors[0]["demand_refs"], [{"evidence_id": "E000002"}])
            self.assertEqual(
                behaviors[0]["action_evidence_ids"],
                ["E000004"],
            )
            self.assertEqual(
                behaviors[0]["response_refs"],
                [{"evidence_id": "E000003"}, {"evidence_id": "E000005"}],
            )
            referenced = {
                evidence_id
                for behavior in behaviors
                for evidence_id in [
                    *(item["evidence_id"] for item in behavior["demand_refs"]),
                    *behavior["action_evidence_ids"],
                    *(item["evidence_id"] for item in behavior["response_refs"]),
                ]
            }
            self.assertNotIn("E000001", referenced)

            crossed = copy.deepcopy(behaviors)
            crossed[0]["response_refs"].append({"evidence_id": "E000007"})
            with self.assertRaisesRegex(BehaviorError, "interaction boundary"):
                validate_behaviors(bundle, evidence, crossed)

    def test_trace_splits_only_explicitly_matched_demand_chains(self) -> None:
        user_content = "1. Update api.py.\n2. Update db.py."
        events = [
            {"event_type": "user_prompt", "turn_number": 1, "content": user_content},
            {
                "event_type": "tool_exchange",
                "turn_number": 2,
                "content": "Updated api.py",
                "evidence_id": "raw-E2",
                "tool_name": "Edit",
            },
            {
                "event_type": "tool_exchange",
                "turn_number": 3,
                "content": "Updated db.py",
                "evidence_id": "raw-E3",
                "tool_name": "Edit",
            },
            {
                "event_type": "assistant_response",
                "turn_number": 4,
                "content": "Both updates are complete.",
                "evidence_id": "raw-E4",
            },
        ]
        with ProjectTemporaryDirectory() as temporary:
            bundle = load_visible_bundle(make_trace_bundle(temporary, events, cutoff=5))
            evidence = build_evidence(bundle)
            behaviors = build_behaviors(bundle, evidence)

        self.assertEqual(len(behaviors), 2)
        self.assertEqual([item["task_id"] for item in behaviors], ["K0001", "K0001"])
        self.assertEqual([item["sequence_index"] for item in behaviors], [1, 2])
        self.assertEqual(
            [item["action_evidence_ids"] for item in behaviors],
            [["E000002"], ["E000003"]],
        )
        selected_demands = [
            user_content[slice(*item["demand_refs"][0]["char_range"])]
            for item in behaviors
        ]
        self.assertEqual(selected_demands, ["Update api.py.", "Update db.py."])
        VALIDATE_BEHAVIORS(behaviors)

    def test_trace_keeps_one_broad_demand_as_one_behavior(self) -> None:
        events = [
            {
                "event_type": "user_prompt",
                "turn_number": 1,
                "content": "Review the proposed CLI fixes and report the safe choices.",
            },
            {
                "event_type": "tool_exchange",
                "turn_number": 2,
                "content": "Modified pyproject.toml",
                "tool_name": "Edit",
            },
            {
                "event_type": "tool_exchange",
                "turn_number": 3,
                "content": "Modified src/cli.py",
                "tool_name": "Edit",
            },
            {
                "event_type": "assistant_response",
                "turn_number": 4,
                "content": "### Version source",
            },
            {
                "event_type": "assistant_response",
                "turn_number": 4,
                "content": "Modified pyproject.toml to keep one version source.",
            },
            {
                "event_type": "assistant_response",
                "turn_number": 4,
                "content": "### CLI validation",
            },
            {
                "event_type": "assistant_response",
                "turn_number": 4,
                "content": "Modified validation in src/cli.py after loading config.",
            },
        ]
        with ProjectTemporaryDirectory() as temporary:
            bundle = load_visible_bundle(make_trace_bundle(temporary, events, cutoff=5))
            evidence = build_evidence(bundle)
            behaviors = build_behaviors(bundle, evidence)

        self.assertEqual(len(behaviors), 1)
        self.assertEqual(len(behaviors[0]["action_evidence_ids"]), 2)
        self.assertEqual(len(behaviors[0]["response_refs"]), 4)

    def test_trace_groups_read_modify_verify_for_the_same_object(self) -> None:
        events = [
            {
                "event_type": "user_prompt",
                "turn_number": 1,
                "content": "1. Apply the src/api.py fix.\n2. Apply the src/db.py fix.",
            },
            {"event_type": "tool_exchange", "turn_number": 2, "content": "Read src/api.py", "tool_name": "Read"},
            {"event_type": "tool_exchange", "turn_number": 3, "content": "Edited src/api.py", "tool_name": "Edit"},
            {"event_type": "tool_exchange", "turn_number": 4, "content": "Tested src/api.py", "tool_name": "Bash"},
            {"event_type": "tool_exchange", "turn_number": 5, "content": "Read src/db.py", "tool_name": "Read"},
            {"event_type": "tool_exchange", "turn_number": 6, "content": "Edited src/db.py", "tool_name": "Edit"},
            {"event_type": "tool_exchange", "turn_number": 7, "content": "Tested src/db.py", "tool_name": "Bash"},
            {"event_type": "assistant_response", "turn_number": 8, "content": "Updated src/api.py."},
            {"event_type": "assistant_response", "turn_number": 9, "content": "Updated src/db.py."},
        ]
        with ProjectTemporaryDirectory() as temporary:
            bundle = load_visible_bundle(make_trace_bundle(temporary, events, cutoff=10))
            evidence = build_evidence(bundle)
            behaviors = build_behaviors(bundle, evidence)

        self.assertEqual(len(behaviors), 2)
        self.assertEqual(
            [len(item["action_evidence_ids"]) for item in behaviors],
            [3, 3],
        )
        self.assertEqual(
            [len(item["response_refs"]) for item in behaviors],
            [1, 1],
        )

    def test_trace_splits_one_multi_result_response_into_verbatim_refs(self) -> None:
        response = "Mypy passed cleanly. Preflight was blocked by missing gitleaks."
        events = [
            {
                "event_type": "user_prompt",
                "turn_number": 1,
                "content": "1. Run mypy.\n2. Run preflight.",
            },
            {
                "event_type": "assistant_response",
                "turn_number": 2,
                "content": response,
            },
        ]
        with ProjectTemporaryDirectory() as temporary:
            bundle = load_visible_bundle(make_trace_bundle(temporary, events, cutoff=3))
            evidence = build_evidence(bundle)
            behaviors = build_behaviors(bundle, evidence)

        self.assertEqual(len(behaviors), 2)
        refs = [item["response_refs"][0] for item in behaviors]
        self.assertEqual([item["evidence_id"] for item in refs], ["E000002", "E000002"])
        self.assertEqual(
            [response[slice(*item["char_range"])] for item in refs],
            ["Mypy passed cleanly.", "Preflight was blocked by missing gitleaks."],
        )

    def test_trace_matches_result_with_incidental_object_to_requested_operation(self) -> None:
        response = (
            "Updated tests/demo.py. "
            "Preflight was blocked by missing `gitleaks` binary."
        )
        events = [
            {
                "event_type": "user_prompt",
                "turn_number": 1,
                "content": (
                    "1. Edit tests/demo.py.\n"
                    "2. Verify scripts/run_pr_preflight.py."
                ),
            },
            {
                "event_type": "tool_exchange",
                "turn_number": 2,
                "content": "Edited tests/demo.py",
                "tool_name": "Edit",
            },
            {
                "event_type": "tool_exchange",
                "turn_number": 3,
                "content": "Ran scripts/run_pr_preflight.py",
                "tool_name": "Bash",
            },
            {
                "event_type": "assistant_response",
                "turn_number": 4,
                "content": response,
            },
        ]
        with ProjectTemporaryDirectory() as temporary:
            bundle = load_visible_bundle(make_trace_bundle(temporary, events, cutoff=5))
            evidence = build_evidence(bundle)
            behaviors = build_behaviors(bundle, evidence)

        self.assertEqual(len(behaviors), 2)
        refs = [item["response_refs"][0] for item in behaviors]
        self.assertEqual(
            [response[slice(*item["char_range"])] for item in refs],
            [
                "Updated tests/demo.py.",
                "Preflight was blocked by missing `gitleaks` binary.",
            ],
        )

    def test_trace_keeps_ambiguous_shared_work_as_one_composite_behavior(self) -> None:
        events = [
            {"event_type": "user_prompt", "turn_number": 1, "content": "Review all of these notes."},
            {"event_type": "tool_exchange", "turn_number": 2, "content": "Read notes", "tool_name": "Read"},
            {"event_type": "assistant_response", "turn_number": 3, "content": "Review complete."},
            {"event_type": "assistant_response", "turn_number": 3, "content": "Everything is summarized together."},
        ]
        with ProjectTemporaryDirectory() as temporary:
            bundle = load_visible_bundle(make_trace_bundle(temporary, events, cutoff=4))
            evidence = build_evidence(bundle)
            behaviors = build_behaviors(bundle, evidence)

        self.assertEqual(len(behaviors), 1)

    def test_trace_uses_outer_list_items_as_demand_boundaries(self) -> None:
        events = [
            {
                "event_type": "user_prompt",
                "turn_number": 1,
                "content": (
                    "Do these fixes:\n"
                    "* **src/a.py**\n  * Replace direct assignment.\n  * Keep fallback.\n"
                    "* **src/b.py**\n  * Replace dynamic access.\n  * Add a test.\n"
                    "* **src/c.py**\n  * Use getattr.\n"
                ),
            },
            {"event_type": "assistant_response", "turn_number": 2, "content": "1. Modified src/a.py."},
            {"event_type": "assistant_response", "turn_number": 2, "content": "2. Modified src/b.py."},
            {"event_type": "assistant_response", "turn_number": 2, "content": "3. Modified src/c.py."},
        ]
        with ProjectTemporaryDirectory() as temporary:
            bundle = load_visible_bundle(make_trace_bundle(temporary, events, cutoff=3))
            evidence = build_evidence(bundle)
            behaviors = build_behaviors(bundle, evidence)

        self.assertEqual(len(behaviors), 3)
        selected_demands = [
            events[0]["content"][slice(*item["demand_refs"][0]["char_range"])]
            for item in behaviors
        ]
        self.assertTrue(selected_demands[0].startswith("**src/a.py**"))
        self.assertTrue(selected_demands[1].startswith("**src/b.py**"))
        self.assertTrue(selected_demands[2].startswith("**src/c.py**"))

    def test_trace_recognizes_a_first_inline_markdown_demand(self) -> None:
        events = [
            {
                "event_type": "user_prompt",
                "turn_number": 1,
                "content": "Review carefully. * **src/a.py** — fix A.\n* **src/b.py** — fix B.",
            },
            {"event_type": "assistant_response", "turn_number": 2, "content": "Fixed src/a.py."},
            {"event_type": "assistant_response", "turn_number": 2, "content": "Fixed src/b.py."},
        ]
        with ProjectTemporaryDirectory() as temporary:
            bundle = load_visible_bundle(make_trace_bundle(temporary, events, cutoff=3))
            evidence = build_evidence(bundle)
            behaviors = build_behaviors(bundle, evidence)

        self.assertEqual(len(behaviors), 2)

    def test_trace_task_scope_can_resume_after_an_unrelated_task(self) -> None:
        events = [
            {"event_type": "user_prompt", "turn_number": 1, "content": "Implement disk cache."},
            {"event_type": "assistant_response", "turn_number": 2, "content": "Implemented disk cache."},
            {"event_type": "user_prompt", "turn_number": 3, "content": "Document release notes."},
            {"event_type": "assistant_response", "turn_number": 4, "content": "Release notes written."},
            {"event_type": "user_prompt", "turn_number": 5, "content": "Verify disk cache behavior."},
            {"event_type": "assistant_response", "turn_number": 6, "content": "Disk cache verified."},
        ]
        with ProjectTemporaryDirectory() as temporary:
            bundle = load_visible_bundle(make_trace_bundle(temporary, events, cutoff=7))
            evidence = build_evidence(bundle)
            behaviors = build_behaviors(bundle, evidence)

        self.assertEqual(
            [(item["task_id"], item["sequence_index"]) for item in behaviors],
            [("K0001", 1), ("K0002", 1), ("K0001", 2)],
        )
        VALIDATE_BEHAVIORS(behaviors)

    def test_trace_direct_continuation_does_not_require_term_overlap(self) -> None:
        events = [
            {"event_type": "user_prompt", "turn_number": 1, "content": "Implement runner fallback."},
            {"event_type": "assistant_response", "turn_number": 2, "content": "Runner fallback implemented."},
            {"event_type": "user_prompt", "turn_number": 3, "content": "Go on. Then run the smoke test again."},
            {"event_type": "assistant_response", "turn_number": 4, "content": "Smoke test passed."},
        ]
        with ProjectTemporaryDirectory() as temporary:
            bundle = load_visible_bundle(make_trace_bundle(temporary, events, cutoff=5))
            evidence = build_evidence(bundle)
            behaviors = build_behaviors(bundle, evidence)

        self.assertEqual(
            [(item["task_id"], item["sequence_index"]) for item in behaviors],
            [("K0001", 1), ("K0001", 2)],
        )

    def test_trace_answer_to_assistant_choice_stays_in_task_scope(self) -> None:
        events = [
            {"event_type": "user_prompt", "turn_number": 1, "content": "Review the failed workflow."},
            {
                "event_type": "assistant_response",
                "turn_number": 2,
                "content": "Three fixes are available: 1. retry; 2. skip; 3. replace.",
            },
            {
                "event_type": "user_prompt",
                "turn_number": 3,
                "content": "Implement 1. Think carefully about how to implement 3.",
            },
            {"event_type": "assistant_response", "turn_number": 4, "content": "Implemented the selected fixes."},
        ]
        with ProjectTemporaryDirectory() as temporary:
            bundle = load_visible_bundle(make_trace_bundle(temporary, events, cutoff=5))
            evidence = build_evidence(bundle)
            behaviors = build_behaviors(bundle, evidence)

        self.assertEqual([item["task_id"] for item in behaviors], ["K0001", "K0001"])

    def test_trace_requested_recheck_stays_in_task_scope(self) -> None:
        events = [
            {"event_type": "user_prompt", "turn_number": 1, "content": "Review the PAT setup."},
            {
                "event_type": "assistant_response",
                "turn_number": 2,
                "content": "Build Read is sufficient. Want me to verify the documentation?",
            },
            {
                "event_type": "user_prompt",
                "turn_number": 3,
                "content": "Please double-check the exact permissions.",
            },
            {"event_type": "assistant_response", "turn_number": 4, "content": "Verified the permissions."},
        ]
        with ProjectTemporaryDirectory() as temporary:
            bundle = load_visible_bundle(make_trace_bundle(temporary, events, cutoff=5))
            evidence = build_evidence(bundle)
            behaviors = build_behaviors(bundle, evidence)

        self.assertEqual([item["task_id"] for item in behaviors], ["K0001", "K0001"])

    def test_trace_long_failure_report_stays_in_current_task_scope(self) -> None:
        events = [
            {"event_type": "user_prompt", "turn_number": 1, "content": "Fix the sidecar startup."},
            {"event_type": "assistant_response", "turn_number": 2, "content": "Applied the startup fix."},
            {
                "event_type": "user_prompt",
                "turn_number": 3,
                "content": "Sidecar failed the health check.\n" + ("diagnostic output\n" * 80),
            },
            {"event_type": "assistant_response", "turn_number": 4, "content": "Fixed the health check."},
        ]
        with ProjectTemporaryDirectory() as temporary:
            bundle = load_visible_bundle(make_trace_bundle(temporary, events, cutoff=5))
            evidence = build_evidence(bundle)
            behaviors = build_behaviors(bundle, evidence)

        self.assertEqual([item["task_id"] for item in behaviors], ["K0001", "K0001"])

    def test_trace_explicit_correction_stays_in_task_scope(self) -> None:
        events = [
            {"event_type": "user_prompt", "turn_number": 1, "content": "Fix multi-chart generation."},
            {"event_type": "assistant_response", "turn_number": 2, "content": "Updated chart generation."},
            {
                "event_type": "user_prompt",
                "turn_number": 3,
                "content": "Only keep the multi-select guidance and revert the other updates.",
            },
            {"event_type": "assistant_response", "turn_number": 4, "content": "Kept only the requested change."},
        ]
        with ProjectTemporaryDirectory() as temporary:
            bundle = load_visible_bundle(make_trace_bundle(temporary, events, cutoff=5))
            evidence = build_evidence(bundle)
            behaviors = build_behaviors(bundle, evidence)

        self.assertEqual([item["task_id"] for item in behaviors], ["K0001", "K0001"])

    def test_trace_explicit_switch_overrides_shared_topic_words(self) -> None:
        events = [
            {
                "event_type": "user_prompt",
                "turn_number": 1,
                "content": "Evaluate the released model with the official harness on TB2.",
            },
            {
                "event_type": "assistant_response",
                "turn_number": 2,
                "content": "The released model evaluation with the official harness is running on TB2.",
            },
            {
                "event_type": "user_prompt",
                "turn_number": 3,
                "content": (
                    "Come back to that later. Shift our attention to the released model "
                    "with the official harness on SWE-Bench."
                ),
            },
            {"event_type": "assistant_response", "turn_number": 4, "content": "The 131k SWE-Bench evaluation started."},
        ]
        with ProjectTemporaryDirectory() as temporary:
            bundle = load_visible_bundle(make_trace_bundle(temporary, events, cutoff=5))
            evidence = build_evidence(bundle)
            behaviors = build_behaviors(bundle, evidence)

        self.assertEqual([item["task_id"] for item in behaviors], ["K0001", "K0002"])

    def test_trace_separable_demands_with_distinct_stable_ids_use_distinct_tasks(self) -> None:
        user_content = "1. Update issue #271.\n2. Check trial 2 for job eval-32k."
        events = [
            {"event_type": "user_prompt", "turn_number": 1, "content": user_content},
            {
                "event_type": "tool_exchange",
                "turn_number": 2,
                "content": "Updated issue #271",
                "evidence_id": "raw-E2",
                "tool_name": "Edit",
            },
            {
                "event_type": "tool_exchange",
                "turn_number": 3,
                "content": "Checked trial 2 for job eval-32k",
                "evidence_id": "raw-E3",
                "tool_name": "Bash",
            },
        ]
        with ProjectTemporaryDirectory() as temporary:
            bundle = load_visible_bundle(make_trace_bundle(temporary, events, cutoff=4))
            evidence = build_evidence(bundle)
            behaviors = build_behaviors(bundle, evidence)

        self.assertEqual([item["task_id"] for item in behaviors], ["K0001", "K0002"])
        self.assertEqual([item["sequence_index"] for item in behaviors], [1, 1])

    def test_repeated_build_is_deterministic(self) -> None:
        with ProjectTemporaryDirectory() as temporary:
            bundle = load_visible_bundle(
                make_repo_bundle(temporary, repository_files={"src/app.py": SOURCE})
            )
            evidence = build_evidence(bundle)
            first = build_behaviors(bundle, evidence)
            second = build_behaviors(bundle, evidence)
            self.assertEqual(first, second)
            self.assertEqual(
                [item["behavior_id"] for item in first],
                [f"B{index:04d}" for index in range(1, len(first) + 1)],
            )


if __name__ == "__main__":
    unittest.main()
