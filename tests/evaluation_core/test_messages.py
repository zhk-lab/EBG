from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agentloop import DEFAULT_CONFIG
from agentloop.benchmark_configs import repo_benchmark_config
from agentloop.prompts import system_prompt
from evaluation_core.contracts import (
    EvaluationCoreError,
    load_prediction_schema,
    model_prediction_schema,
)
from evaluation_core.messages import (
    build_baseline_messages,
    build_repo_initial_user_prompt,
    build_trace_review_messages,
    load_baseline_prompt,
    load_task_prompt,
)
from tests.support import ProjectTemporaryDirectory


class PromptAssetTests(unittest.TestCase):
    def test_repo_pair_shares_proof_rules_and_output_contract(self) -> None:
        for benchmark in ("specgap", "silentswap"):
            raw = load_task_prompt("baseline", benchmark)
            graph = load_task_prompt("BEG", benchmark)
            marker = "For each candidate"
            raw_rules = raw.split(marker, 1)[1].replace("`R0001`", "`F0001`")
            graph_rules = graph.split(marker, 1)[1].replace(
                "directory rows, search hits, Behavior labels, or edges", "directory rows or search hits"
            )
            if benchmark == "specgap":
                # BEG now has a count prior; evidence rules and answer fields stay shared.
                self.assertEqual(
                    raw_rules.split("\n\nReject", 1)[0],
                    graph_rules.split("\n\nReject", 1)[0],
                )
                self.assertEqual(
                    re.search(r"```json\s*(.*?)\s*```", raw, re.DOTALL).group(1),
                    re.search(r"```json\s*(.*?)\s*```", graph, re.DOTALL).group(1),
                )
                self.assertEqual(
                    raw_rules.split("- `finding_id`", 1)[1],
                    graph_rules.split("- `finding_id`", 1)[1],
                )
            else:
                # Localization retains grouping and taxonomy guidance, while
                # baseline still requires the full behavioral answer.
                self.assertEqual(
                    re.sub(r"\n- Cite .*?(?=\n- )", "",
                           raw_rules.split("TARGET AND GROUPING RULES", 1)[1].split("OUTPUT CONTRACT", 1)[0],
                           flags=re.DOTALL),
                    re.sub(r"\n- Cite .*?(?=\n- )", "",
                           graph_rules.split("TARGET AND GROUPING RULES", 1)[1].split("OUTPUT CONTRACT", 1)[0],
                           flags=re.DOTALL),
                )
            if benchmark == "silentswap":
                common = "these locations can implement a swap as directly as a public function."
                self.assertIn(common, raw)
                self.assertIn(common, graph)

    def test_agent_protocol_forbids_concatenated_future_actions(self) -> None:
        prompt = system_prompt(DEFAULT_CONFIG)
        normalized = " ".join(prompt.split())

        self.assertIn("exactly one bare JSON object", normalized)
        self.assertIn('{"action":"search","text":"keywords"}', normalized)
        self.assertIn('{"action":"read","ids":["known Read ID"]}', normalized)
        self.assertIn('{"action":"finish","prediction":{}}', normalized)
        self.assertIn("Choose only the next action", normalized)
        self.assertIn("Never simulate a tool result", normalized)
        self.assertIn("exists only after the runner sends", normalized)
        self.assertIn("mutually exclusive templates", normalized)
        self.assertIn("response ends immediately", normalized)
        self.assertIn("Use search only when the needed Read ID", prompt)
        self.assertIn("Read accepts 1 to", prompt)
        self.assertIn("Finish submits the exact prediction object", prompt)
        self.assertNotIn("LINEAR LOCAL GRAPH JSON", prompt)
        self.assertNotIn("implementation evidence", prompt)
        self.assertNotIn("Context compression", prompt)

    def test_all_six_allowlisted_prompts_are_non_empty(self) -> None:
        for variant in ("BEG", "baseline"):
            for benchmark in ("specgap", "silentswap", "feedbacktrace"):
                with self.subTest(variant=variant, benchmark=benchmark):
                    prompt = load_task_prompt(variant, benchmark)
                    self.assertTrue(prompt.strip())

    def test_repo_task_prompts_end_with_single_action_guidance(self) -> None:
        ending = (
            "Each round must return exactly one action. When you have the answer and "
            "the evidence is sufficient, return finish. Check the JSON action format "
            "carefully."
        )
        for variant in ("BEG", "baseline"):
            for benchmark in ("specgap", "silentswap"):
                with self.subTest(variant=variant, benchmark=benchmark):
                    prompt = " ".join(load_task_prompt(variant, benchmark).split())
                    self.assertTrue(prompt.endswith(ending))

    def test_feedbacktrace_prompts_share_structure_and_decision_rules(self) -> None:
        headings = (
            "ROLE AND OBJECTIVE",
            "INPUT FORMAT",
            "HOW TO READ THE INPUT",
            "DECISION SELECTION RULES",
            "VERIFICATION POINT AND EVIDENCE DEFINITIONS",
            "OUTPUT JSON CONTRACT",
            "FINAL CHECK",
        )
        prompts = {
            variant: load_task_prompt(variant, "feedbacktrace")
            for variant in ("baseline", "BEG")
        }

        def sections(prompt: str) -> dict[str, str]:
            positions = [prompt.index(heading) for heading in headings]
            self.assertEqual(positions, sorted(positions))
            return {
                heading: prompt[start + len(heading):end].strip()
                for heading, start, end in zip(
                    headings,
                    positions,
                    positions[1:] + [len(prompt)],
                )
            }

        baseline = sections(prompts["baseline"])
        graph = sections(prompts["BEG"])
        for heading in headings:
            if heading in {"INPUT FORMAT", "HOW TO READ THE INPUT"}:
                self.assertNotEqual(baseline[heading], graph[heading])
            else:
                self.assertEqual(baseline[heading], graph[heading])
        for prompt in prompts.values():
            normalized = " ".join(prompt.split())
            self.assertIn("three semantic parts", normalized)
            self.assertIn("A direct decision anchor", normalized)
            self.assertIn("A minimal direct sufficient Evidence set", normalized)
            self.assertIn("Every formal sample contains such a decision", normalized)
            self.assertNotIn("NO_KEY", prompt)
            self.assertNotIn("KEY", prompt)
            self.assertNotIn('"verdict"', prompt)
            self.assertNotIn("max_output_tokens", prompt)
            self.assertNotIn("network retries", prompt.lower())
            self.assertNotIn("context window", prompt.lower())

        beg_prompt = prompts["BEG"]
        for term in (
            "`task_scopes`",
            "`current_task_id`",
            "`status`",
            "`task_id`",
            "`sequence_index`",
            "`demand`",
            "`action`",
            "`response`",
            "`relations`",
            "`informs`",
            "`supersedes`",
            "`content`",
        ):
            self.assertIn(term, beg_prompt)
        for obsolete in ("`root`", "`steps`", "`cross_links`", "`precedes`", "`continues`"):
            self.assertNotIn(obsolete, beg_prompt)

    def test_beg_prompts_contain_their_model_visible_contracts(self) -> None:
        for benchmark in ("specgap", "silentswap", "feedbacktrace"):
            with self.subTest(benchmark=benchmark):
                prompt = load_task_prompt("BEG", benchmark)
                schema = model_prediction_schema(
                    load_prediction_schema(PROJECT_ROOT / "schemas",
                        "silentswap_location_prediction.schema.json" if benchmark == "silentswap" else benchmark)
                )
                for field in _property_names(schema):
                    self.assertIn(f'"{field}"', prompt)
                for value in _enum_strings(schema):
                    self.assertIn(value, prompt)

    def test_beg_repo_prompts_leave_action_protocol_to_system(self) -> None:
        for benchmark in ("specgap", "silentswap"):
            with self.subTest(benchmark=benchmark):
                prompt = load_task_prompt("BEG", benchmark)
                self.assertNotIn("ACTION PROTOCOL", prompt)
                self.assertNotIn('{"action":"search"', prompt)
                self.assertNotIn('{"action":"read"', prompt)
                self.assertIn("finish.prediction", prompt)

    def test_beg_repo_prompt_output_examples_are_valid_json(self) -> None:
        for benchmark in ("specgap", "silentswap"):
            with self.subTest(benchmark=benchmark):
                prompt = load_task_prompt("BEG", benchmark)
                blocks = re.findall(r"```json\s*(.*?)\s*```", prompt, re.DOTALL)
                self.assertEqual(len(blocks), 1)
                self.assertIsInstance(json.loads(blocks[0]), dict)

    def test_beg_specgap_prompt_prioritizes_and_completes_source_review(self) -> None:
        prompt = load_task_prompt("BEG", "specgap")
        normalized = " ".join(prompt.split())

        self.assertIn("SpecGap recovery", normalized)
        self.assertIn("otherwise largely complete task document", normalized)
        self.assertIn("ranked file table", normalized)
        self.assertIn("Direct document matches rank first", normalized)
        self.assertIn("compact linear Local Graph", prompt)
        self.assertIn("complete numbered Root source", normalized)
        self.assertIn("[DIRECT ROOT]", prompt)
        self.assertIn("[CONTEXT via calls]", prompt)
        self.assertIn("falls back to the complete Symbol source", normalized)
        self.assertNotIn("Local Graph JSON", prompt)
        self.assertNotIn("Compression preserves", prompt)
        self.assertIn("directory entries are options, not a checklist", normalized.lower())
        self.assertIn("complete TASK DOCUMENT", normalized)
        self.assertIn("defaults, null/empty handling, exceptions, ordering", normalized)
        self.assertIn("2–5 externally meaningful implemented rules have been removed", normalized)
        self.assertIn("identify at least two distinct", normalized)
        self.assertIn("re-examine all previously read code", normalized)
        self.assertIn("MUST contain at least 2 items", normalized)
        self.assertIn("There is no final-round exception", normalized)
        self.assertNotIn("only the supported findings", normalized)
        self.assertNotIn("empty `findings` array is correct", normalized)
        self.assertNotIn("there is no target count", normalized)
        self.assertIn("current production source", normalized)
        for heading in (
            "INPUT AND GRAPH FORMAT",
            "REVIEW WORKFLOW AND PROOF STANDARD",
            "OUTPUT CONTRACT",
            "FINAL CHECK",
        ):
            self.assertIn(heading, prompt)
        self.assertNotIn("CODE IDENTIFIED FROM DOCUMENT", normalized)
        self.assertNotIn("ONE-HOP CONNECTED CODE", normalized)

    def test_beg_silentswap_prompt_requires_unique_grounded_slots(self) -> None:
        prompt = load_task_prompt("BEG", "silentswap")
        normalized = " ".join(prompt.split())

        self.assertIn("SilentSwap recovery", normalized)
        self.assertIn("original_document", normalized)
        self.assertIn("BEFORE -> AFTER difference", normalized)
        self.assertIn("numbered source returned by read defines AFTER", normalized)
        self.assertNotIn("Local Graph JSON", prompt)
        self.assertNotIn("Compression preserves", prompt)
        self.assertIn("TASK DOCUMENT defines BEFORE", normalized)
        self.assertIn("For each candidate, establish all four facts", normalized)
        self.assertIn("Distinguishing trigger", normalized)
        self.assertIn("Review every returned source block before finish", normalized)
        self.assertIn("Report five distinct swaps", normalized)
        self.assertIn("AFTER source actually read", normalized)
        self.assertIn("Most swaps need only 1-2 lines", normalized)
        self.assertIn("Include ALL coordinated sites", normalized)
        self.assertIn("Report independent changes separately", normalized)
        self.assertIn("use separate entries for non-contiguous lines", normalized)
        self.assertIn("Output only target for each swap", normalized)
        self.assertIn("a top-level assignment is a `field`", normalized)
        for heading in (
            "INPUT AND GRAPH FORMAT",
            "REVIEW WORKFLOW AND PROOF STANDARD",
            "TARGET AND GROUPING RULES",
            "SWAP TYPE MEANINGS",
            "OUTPUT CONTRACT",
            "FINAL CHECK",
        ):
            self.assertIn(heading, prompt)
        self.assertNotIn("ORIGINAL-DOCUMENT-LINKED CODE", normalized)
        self.assertNotIn("ONE-HOP CONNECTED CODE", normalized)

    def test_repo_configs_require_the_ranked_file_directory_sections(self) -> None:
        specgap = repo_benchmark_config("specgap")
        silentswap = repo_benchmark_config("silentswap")

        self.assertEqual(
            specgap.directory_strategy, "specgap_ranked_file_directory_v4"
        )
        self.assertEqual(
            specgap.required_directory_sections,
            (
                "FILES LINKED TO THE TASK DOCUMENT",
                "ADDITIONAL FILES WITH OBSERVABLE BEHAVIOR",
            ),
        )
        self.assertEqual(
            silentswap.directory_strategy,
            "silentswap_ranked_file_directory_v4",
        )
        self.assertEqual(
            silentswap.required_directory_sections,
            ("FILES LINKED TO THE ORIGINAL DOCUMENT",),
        )

    def test_prompt_selection_rejects_unknown_or_path_like_values(self) -> None:
        invalid = (
            ("beg", "specgap"),
            ("BEG", "../specgap"),
            ("baseline", "unknown"),
            ("../baseline", "feedbacktrace"),
        )
        for variant, benchmark in invalid:
            with self.subTest(variant=variant, benchmark=benchmark):
                with self.assertRaises(EvaluationCoreError):
                    load_task_prompt(variant, benchmark)

    def test_baseline_repo_templates_separate_protocol_task_and_input(self) -> None:
        document = "UNIQUE_TASK_DOCUMENT_73"
        manifest = "UNIQUE_REPOSITORY_MANIFEST_91"

        specgap = build_baseline_messages(
            "specgap",
            task_document=document,
            repository_manifest=manifest,
        )
        silentswap = build_baseline_messages(
            "silentswap",
            task_document=document,
            repository_manifest=manifest,
        )

        self.assertEqual([message["role"] for message in specgap], ["system", "user"])
        self.assertEqual([message["role"] for message in silentswap], ["system", "user"])
        self.assertEqual(specgap[0]["content"], system_prompt(DEFAULT_CONFIG))
        self.assertEqual(silentswap[0]["content"], system_prompt(DEFAULT_CONFIG))
        self.assertIn('"action":"search"', specgap[0]["content"])
        self.assertIn('"action":"read"', specgap[0]["content"])
        self.assertIn('"action":"finish"', specgap[0]["content"])
        self.assertNotIn("read_file", specgap[0]["content"])
        self.assertNotIn("read_files", silentswap[0]["content"])
        self.assertNotIn("SilentSwap", silentswap[0]["content"])
        self.assertNotIn("SpecGap", specgap[0]["content"])
        self.assertIn("[[BENCHMARK TASK]]", specgap[1]["content"])
        self.assertIn("[[IMMUTABLE SAMPLE INPUT]]", specgap[1]["content"])
        self.assertIn("[[BENCHMARK TASK]]", silentswap[1]["content"])
        self.assertIn("[[IMMUTABLE SAMPLE INPUT]]", silentswap[1]["content"])
        self.assertIn("exactly five", silentswap[1]["content"])
        self.assertIn("INPUT AND SOURCE FORMAT", specgap[1]["content"])
        self.assertIn("INPUT AND SOURCE FORMAT", silentswap[1]["content"])
        self.assertIn("Each R Read ID", specgap[1]["content"])
        self.assertIn("Each R Read ID", silentswap[1]["content"])
        self.assertNotIn("Local Graph", specgap[1]["content"])
        self.assertNotIn("Local Graph", silentswap[1]["content"])
        for messages in (specgap, silentswap):
            rendered = "\n".join(message["content"] for message in messages)
            self.assertEqual(rendered.count(document), 1)
            self.assertEqual(rendered.count(manifest), 1)
            self.assertNotIn("{{TASK_DOCUMENT}}", rendered)
            self.assertNotIn("{{REPOSITORY_MANIFEST}}", rendered)

    def test_feedbacktrace_baseline_renders_canonical_payload(self) -> None:
        payload = {
            "input_id": "ft_render_long",
            "track": "long",
            "selectable_evidence_ids": ["E2", "E7"],
            "events": [
                {"turn": 1, "type": "user", "text": "Inspect this."},
                {"turn": 2, "type": "assistant", "text": "Done.", "evidence_id": "E2"},
            ],
            "early_history_summary": "Earlier setup was routine.",
        }
        messages = build_baseline_messages(
            "feedbacktrace",
            trace_payload=payload,
        )
        expected = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        template = load_baseline_prompt("feedbacktrace")
        self.assertEqual(messages[0]["content"], template.system)
        self.assertNotIn("{{TRACE_PAYLOAD_JSON}}", messages[0]["content"])
        self.assertIn(
            "Do not predict or reconstruct the next user message",
            " ".join(messages[0]["content"].split()),
        )
        self.assertIn(expected, messages[1]["content"])
        self.assertNotIn("{{TRACE_PAYLOAD_JSON}}", messages[1]["content"])

    def test_baseline_repo_loader_binds_the_shared_agentloop_system(self) -> None:
        with ProjectTemporaryDirectory() as prompt_root:
            path = prompt_root / "baseline" / "specgap.txt"
            path.parent.mkdir(parents=True)
            path.write_text(
                "benchmark task only\n",
                encoding="utf-8",
            )
            template = load_baseline_prompt("specgap", prompt_root=prompt_root)
            self.assertEqual(template.system, system_prompt(DEFAULT_CONFIG))
            self.assertEqual(template.benchmark, "benchmark task only")

    def test_repo_module7_selects_raw_and_graph_benchmark_prompts(self) -> None:
        cases = (
            ("specgap", "3_document_after.md"),
            ("silentswap", "original_document.md"),
        )
        for benchmark, document_name in cases:
            with self.subTest(benchmark=benchmark):
                config = repo_benchmark_config(benchmark)
                common = {
                    "input_id": f"{benchmark}_prompt",
                    "task_document_name": document_name,
                    "task_document": "complete task document",
                }
                raw = config.bind_module7(
                    PROJECT_ROOT / "schemas",
                    **common,
                    initial_index="R0001 | pkg/api.py",
                    prompt_variant="baseline",
                )
                graph = config.bind_module7(
                    PROJECT_ROOT / "schemas",
                    **common,
                    initial_index="F0001 | pkg/api.py",
                    prompt_variant="BEG",
                )
                self.assertIn("INPUT AND SOURCE FORMAT", raw.initial_user_prompt)
                self.assertIn("Each R Read ID", raw.initial_user_prompt)
                self.assertNotIn("Local Graph JSON", raw.initial_user_prompt)
                self.assertIn("INPUT AND GRAPH FORMAT", graph.initial_user_prompt)
                if benchmark == "specgap":
                    self.assertRegex(graph.initial_user_prompt, r"compact (?:linear )?Local Graph")
                    self.assertIn("[DIRECT ROOT]", graph.initial_user_prompt)
                else:
                    self.assertIn("Output only target for each swap", graph.initial_user_prompt)
                self.assertNotIn("Local Graph JSON", graph.initial_user_prompt)

    def test_beg_repo_request_explains_all_runtime_blocks(self) -> None:
        prompt = build_repo_initial_user_prompt(
            input_id="sg_prompt",
            benchmark="specgap",
            task_prompt=load_task_prompt("BEG", "specgap"),
            task_document="complete document",
            initial_index="[REPOSITORY SOURCE UNITS]\nR0001  pkg/api.py",
        )
        for title in (
            "RUN INPUT",
            "BENCHMARK TASK",
            "TASK DOCUMENT",
            "INITIAL REPOSITORY INDEX",
        ):
            self.assertIn(f"[[{title}]]", prompt)
        self.assertNotIn("[[OUTPUT SCHEMA]]", prompt)
        self.assertIn('"code_evidence"', prompt)

    def test_beg_trace_request_contains_semantic_task_and_complete_view(self) -> None:
        messages = build_trace_review_messages(
            input_id="ft_prompt_long",
            task_prompt=load_task_prompt("BEG", "feedbacktrace"),
            trace_view=(
                '{"input_id":"ft_prompt_long","benchmark":"feedbacktrace",'
                '"current_task_id":"K0001","task_scopes":['
                '{"task_id":"K0001","status":"current","behaviors":[]}]}\n'
            ),
        )
        rendered = "\n".join(message["content"] for message in messages)
        self.assertIn("Input ID: ft_prompt_long", rendered)
        self.assertIn("[[COMPLETE TRACE VIEW]]", rendered)
        self.assertIn(
            "concrete, consequential Agent decision",
            " ".join(rendered.split()),
        )
        self.assertIn("DECISION SELECTION RULES", rendered)
        self.assertNotIn("NO_KEY", rendered)
        self.assertNotIn('"verdict"', rendered)
        self.assertIn('"current_task_id":"K0001"', rendered)
        self.assertIn('"task_id":"K0001","status":"current"', rendered)
        self.assertNotIn("[[INTERACTION]]", rendered)
        self.assertNotIn("[[OUTPUT SCHEMA]]", rendered)


def _property_names(value):
    result = set()
    if isinstance(value, dict):
        properties = value.get("properties")
        if isinstance(properties, dict):
            result.update(properties)
        for item in value.values():
            result.update(_property_names(item))
    elif isinstance(value, list):
        for item in value:
            result.update(_property_names(item))
    return result


def _enum_strings(value):
    result = set()
    if isinstance(value, dict):
        enum = value.get("enum")
        if isinstance(enum, list):
            result.update(item for item in enum if isinstance(item, str))
        for item in value.values():
            result.update(_enum_strings(item))
    elif isinstance(value, list):
        for item in value:
            result.update(_enum_strings(item))
    return result


if __name__ == "__main__":
    unittest.main()
