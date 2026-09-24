"""Offline regression tests for the SpecGAP 2.2 dual-mapping contract."""

from __future__ import annotations

import copy
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ..scripts import generate_specgap_v2 as generator
from ..scripts import llm_source
from ..scripts import remap_dual_mappings_v2_2 as remap
from ..scripts import validate_specgap_v2 as validator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUALIFIED_ROOT = PROJECT_ROOT / "SpecGAP"
REFERENCE_SAMPLE = QUALIFIED_ROOT / "29:hinthornw_dydantic_pr4"


def _deleted_part(condition_id: str, marker: str) -> dict[str, object]:
    return {
        "condition_id": condition_id,
        "type": "boundary_behavior",
        "source_text": marker,
        "normalized_condition": marker,
        "deleted_spans": [{"exact_text": marker}],
        "why_important": f"{marker} matters",
    }


def _minimal_mapping(condition_id: str) -> dict[str, object]:
    return {
        "condition_id": condition_id,
        "mapping_status": "no_direct_mapping",
        "coverage": "none",
        "evidence_basis": [],
        "mapping_explanation": "No reliable repository evidence was found.",
        "no_direct_mapping_reason": {
            "code": "evidence_not_found",
            "detail": "The indexed repository contains no matching evidence.",
        },
        "evidence": [],
        "related_tests": {
            "status": "none_found",
            "evidence_ids": [],
            "search_notes": "All indexed test snippets were checked.",
        },
    }


class PromptIsolationTests(unittest.TestCase):
    def test_condition_prompt_contains_only_the_requested_condition(self) -> None:
        first = _deleted_part("kc_001", "UNIQUE_FIRST_CONDITION")
        second = _deleted_part("kc_002", "UNIQUE_SECOND_CONDITION")

        messages = generator.condition_shard_prompt(
            {"github_url": "https://github.com/example/repo", "parent_commit": "a" * 40},
            first,
            shard_id="shard_001",
            shard_count=1,
            repo_context="[repository evidence]",
        )
        prompt_text = "\n".join(message["content"] for message in messages)

        self.assertIn("kc_001", prompt_text)
        self.assertIn("UNIQUE_FIRST_CONDITION", prompt_text)
        self.assertNotIn("kc_002", prompt_text)
        self.assertNotIn("UNIQUE_SECOND_CONDITION", prompt_text)
        self.assertNotIn(str(second), prompt_text)

    def test_holistic_shard_contains_full_document_and_all_conditions(
        self,
    ) -> None:
        deleted_parts = [
            _deleted_part("kc_001", "SHARD_FIRST"),
            _deleted_part("kc_002", "SHARD_SECOND"),
        ]
        document = "complete document\nSHARD_DOCUMENT_END_MARKER"
        messages = generator.holistic_shard_prompt(
            {"github_url": "https://github.com/example/repo", "parent_commit": "a" * 40},
            document,
            deleted_parts,
            shard_id="shard_001",
            shard_count=1,
            repo_context="[repository evidence]",
        )
        prompt_text = "\n".join(message["content"] for message in messages)

        self.assertIn(document, prompt_text)
        self.assertIn("kc_001", prompt_text)
        self.assertIn("kc_002", prompt_text)
        self.assertNotIn("_cbc_ev", prompt_text)

    def test_holistic_aggregation_contains_full_document_and_all_conditions(
        self,
    ) -> None:
        deleted_parts = [
            _deleted_part("kc_001", "HOLISTIC_FIRST"),
            _deleted_part("kc_002", "HOLISTIC_SECOND"),
        ]
        document = "complete document\nFULL_DOCUMENT_END_MARKER"
        messages = generator.holistic_aggregation_prompt(
            {"github_url": "https://github.com/example/repo", "parent_commit": "a" * 40},
            document,
            deleted_parts,
            [{"shard_id": "shard_001", "condition_findings": []}],
        )
        prompt_text = "\n".join(message["content"] for message in messages)

        self.assertIn(document, prompt_text)
        self.assertIn("kc_001", prompt_text)
        self.assertIn("kc_002", prompt_text)
        self.assertIn("HOLISTIC_FIRST", prompt_text)
        self.assertIn("HOLISTIC_SECOND", prompt_text)
        self.assertNotIn("_cbc_ev", prompt_text)

    def test_condition_quality_audit_stays_single_condition_and_sees_full_repo(
        self,
    ) -> None:
        first = _deleted_part("kc_001", "QUALITY_FIRST")
        second = _deleted_part("kc_002", "QUALITY_SECOND")
        current = {
            **_minimal_mapping("kc_001"),
            "mapping_explanation": "CBC_DRAFT_MARKER",
        }
        messages = generator.condition_quality_audit_prompt(
            {
                "github_url": "https://github.com/example/repo",
                "parent_commit": "a" * 40,
            },
            first,
            current,
            repo_context="FULL_REPOSITORY_END_MARKER",
        )
        prompt_text = "\n".join(message["content"] for message in messages)

        self.assertIn("QUALITY_FIRST", prompt_text)
        self.assertIn("CBC_DRAFT_MARKER", prompt_text)
        self.assertIn("FULL_REPOSITORY_END_MARKER", prompt_text)
        self.assertNotIn("QUALITY_SECOND", prompt_text)
        self.assertNotIn(str(second), prompt_text)
        self.assertNotIn("ROUTE_B_SECRET", prompt_text)

    def test_holistic_quality_audit_sees_full_inputs_but_no_route_a_output(
        self,
    ) -> None:
        deleted_parts = [
            _deleted_part("kc_001", "QUALITY_HA_FIRST"),
            _deleted_part("kc_002", "QUALITY_HA_SECOND"),
        ]
        document = "complete document\nQUALITY_DOCUMENT_END_MARKER"
        current = [
            {
                **_minimal_mapping("kc_001"),
                "mapping_explanation": "HOLISTIC_DRAFT_MARKER",
            },
            _minimal_mapping("kc_002"),
        ]
        messages = generator.holistic_quality_audit_prompt(
            {
                "github_url": "https://github.com/example/repo",
                "parent_commit": "a" * 40,
            },
            document,
            deleted_parts,
            "current holistic summary",
            current,
            repo_context="FULL_HOLISTIC_REPOSITORY_END_MARKER",
        )
        prompt_text = "\n".join(message["content"] for message in messages)

        self.assertIn(document, prompt_text)
        self.assertIn("QUALITY_HA_FIRST", prompt_text)
        self.assertIn("QUALITY_HA_SECOND", prompt_text)
        self.assertIn("HOLISTIC_DRAFT_MARKER", prompt_text)
        self.assertIn("FULL_HOLISTIC_REPOSITORY_END_MARKER", prompt_text)
        self.assertNotIn("ROUTE_A_SECRET", prompt_text)
        self.assertNotIn("_cbc_ev", prompt_text)

    def test_route_b_receives_no_route_a_output(self) -> None:
        deleted_parts = [
            _deleted_part("kc_001", "first"),
            _deleted_part("kc_002", "second"),
        ]
        route_a = [
            {**_minimal_mapping("kc_001"), "mapping_explanation": "ROUTE_A_SECRET"},
            _minimal_mapping("kc_002"),
        ]
        route_b = [
            _minimal_mapping("kc_001"),
            _minimal_mapping("kc_002"),
        ]
        captured: dict[str, object] = {}

        def fake_holistic(**kwargs: object) -> tuple[str, list[dict[str, object]], dict[str, int]]:
            captured.update(kwargs)
            return (
                "independent holistic result",
                route_b,
                {"indexed_files": 1, "indexed_snippets": 1, "shards": 1},
            )

        with tempfile.TemporaryDirectory() as temporary:
            with (
                patch.object(
                    generator,
                    "generate_condition_by_condition_mappings",
                    return_value=route_a,
                ),
                patch.object(
                    generator,
                    "generate_holistic_alignment",
                    side_effect=fake_holistic,
                ),
                patch.object(
                    generator,
                    "render_repository_shards",
                    return_value=[
                        {"context": "[repository evidence]", "snippets": ()}
                    ],
                ),
            ):
                bundle = generator.generate_dual_mappings(
                    row={},
                    document_before="complete document",
                    deleted_parts=deleted_parts,
                    repo_root=Path(temporary),
                    snippets=[SimpleNamespace(file_path="src/example.py")],
                    run_dir=Path(temporary) / "run",
                    args=SimpleNamespace(holistic_shard_chars=60_000),
                    provider="test",
                    model="test",
                    base_url="",
                    api_key="",
                )

        self.assertNotIn("condition_by_condition", captured)
        self.assertNotIn("ROUTE_A_SECRET", repr(captured))
        self.assertEqual(
            bundle["comparison_items"][0]["condition_by_condition"][
                "mapping_explanation"
            ],
            "ROUTE_A_SECRET",
        )
        self.assertEqual(
            bundle["comparison_items"][0]["manual_comparison"],
            {"agreement": None, "notes": ""},
        )
        self.assertEqual(
            bundle["methods"]["condition_by_condition"]["mode"],
            "one_condition_full_repository_sharded_alignment",
        )


class SafetyRegressionTests(unittest.TestCase):
    def test_generator_defaults_to_deepseek_pro_with_ten_workers(self) -> None:
        with patch("sys.argv", ["generate_specgap_v2.py"]):
            args = generator.parse_args()

        self.assertEqual(args.provider, "deepseek")
        self.assertIsNone(args.model)
        self.assertEqual(
            llm_source.get_model(args.provider, args.model),
            "deepseek-v4-pro",
        )
        self.assertEqual(args.workers, 10)
        self.assertIsNone(args.thinking)
        self.assertIsNone(args.reasoning_effort)

    def test_api_telemetry_preserves_provider_usage_without_secrets(self) -> None:
        response = SimpleNamespace(
            status_code=200,
            json=lambda: {
                "id": "response-test",
                "choices": [
                    {
                        "message": {"content": '{"ok": true}'},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 120,
                    "completion_tokens": 30,
                    "total_tokens": 150,
                    "prompt_cache_hit_tokens": 100,
                },
            },
        )
        args = SimpleNamespace(
            temperature=0.0,
            max_tokens=1024,
            retries=1,
            timeout=30,
        )
        with tempfile.TemporaryDirectory() as directory:
            telemetry_path = Path(directory) / "attempt_01_api.json"
            with patch.object(llm_source.requests, "post", return_value=response):
                content = llm_source.call_chat_completion(
                    messages=[{"role": "user", "content": "hello"}],
                    provider="deepseek",
                    model="deepseek-v4-pro",
                    base_url="https://api.deepseek.example/v1",
                    api_key="secret-key-that-must-not-be-recorded",
                    args=args,
                    attempts=1,
                    telemetry_path=telemetry_path,
                )

            telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
            self.assertEqual(content, '{"ok": true}')
            self.assertEqual(telemetry["usage"]["total_tokens"], 150)
            self.assertEqual(telemetry["request"]["input_characters"], 5)
            self.assertNotIn("secret-key-that-must-not-be-recorded", str(telemetry))

    def test_account_api_error_is_fatal_and_not_retried(self) -> None:
        response = SimpleNamespace(
            status_code=402,
            text='{"error":{"message":"Insufficient Balance"}}',
        )
        args = SimpleNamespace(
            temperature=0.0,
            max_tokens=1024,
            retries=8,
            timeout=30,
            batch_abort_event=threading.Event(),
        )
        with tempfile.TemporaryDirectory() as directory:
            telemetry_path = Path(directory) / "attempt_01_api.json"
            with patch.object(
                llm_source.requests, "post", return_value=response
            ) as post:
                with self.assertRaises(llm_source.FatalAPIError):
                    llm_source.call_chat_completion(
                        messages=[{"role": "user", "content": "hello"}],
                        provider="deepseek",
                        model="deepseek-v4-pro",
                        base_url="https://api.deepseek.example/v1",
                        api_key="secret-key-that-must-not-be-recorded",
                        args=args,
                        attempts=8,
                        telemetry_path=telemetry_path,
                    )

            telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
            self.assertEqual(post.call_count, 1)
            self.assertEqual(telemetry["http_status"], 402)
            self.assertEqual(telemetry["status"], "failed")
            self.assertTrue(args.batch_abort_event.is_set())

            with patch.object(llm_source.requests, "post") as peer_post:
                with self.assertRaises(llm_source.BatchAbortError):
                    llm_source.call_chat_completion(
                        messages=[{"role": "user", "content": "peer"}],
                        provider="deepseek",
                        model="deepseek-v4-pro",
                        base_url="https://api.deepseek.example/v1",
                        api_key="secret-key-that-must-not-be-recorded",
                        args=args,
                        attempts=1,
                    )
            peer_post.assert_not_called()

    def test_candidate_stage_propagates_fatal_api_error_without_repair(self) -> None:
        args = SimpleNamespace(resume=False, retries=8)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(
                generator,
                "call_chat_completion",
                side_effect=llm_source.FatalAPIError(
                    402, '{"error":{"message":"Insufficient Balance"}}'
                ),
            ) as call:
                with self.assertRaises(llm_source.FatalAPIError):
                    generator.call_candidate_stage(
                        stage="fatal_stage",
                        base_messages=[{"role": "user", "content": "hello"}],
                        normalize=lambda value: value,
                        run_dir=Path(directory),
                        args=args,
                        provider="deepseek",
                        model="deepseek-v4-pro",
                        base_url="https://api.deepseek.example/v1",
                        api_key="secret-key-that-must-not-be-recorded",
                    )

            self.assertEqual(call.call_count, 1)
            validation = json.loads(
                (Path(directory) / "fatal_stage" / "attempt_01_validation.json")
                .read_text(encoding="utf-8")
            )
            self.assertFalse(validation["valid"])

    def test_all_account_auth_statuses_abort_without_retry(self) -> None:
        args = SimpleNamespace(
            temperature=0.0,
            max_tokens=1024,
            retries=3,
            timeout=30,
        )
        for status_code in (401, 402, 403):
            with self.subTest(status_code=status_code):
                response = SimpleNamespace(
                    status_code=status_code,
                    text='{"error":{"message":"account error"}}',
                )
                with patch.object(
                    llm_source.requests, "post", return_value=response
                ) as post:
                    with self.assertRaises(llm_source.FatalAPIError):
                        llm_source.call_chat_completion(
                            messages=[{"role": "user", "content": "hello"}],
                            provider="deepseek",
                            model="deepseek-v4-pro",
                            base_url="https://api.deepseek.example/v1",
                            api_key="secret-key-that-must-not-be-recorded",
                            args=args,
                            attempts=3,
                        )
                self.assertEqual(post.call_count, 1)

    def test_transient_http_errors_keep_normal_retry_behavior(self) -> None:
        args = SimpleNamespace(
            temperature=0.0,
            max_tokens=1024,
            retries=2,
            timeout=30,
            batch_abort_event=threading.Event(),
        )
        for status_code in (429, 500):
            with self.subTest(status_code=status_code):
                response = SimpleNamespace(
                    status_code=status_code,
                    text='{"error":{"message":"transient error"}}',
                )
                with (
                    patch.object(
                        llm_source.requests, "post", return_value=response
                    ) as post,
                    patch.object(llm_source.time, "sleep"),
                ):
                    with self.assertRaises(RuntimeError):
                        llm_source.call_chat_completion(
                            messages=[{"role": "user", "content": "hello"}],
                            provider="deepseek",
                            model="deepseek-v4-pro",
                            base_url="https://api.deepseek.example/v1",
                            api_key="secret-key-that-must-not-be-recorded",
                            args=args,
                            attempts=2,
                        )
                self.assertEqual(post.call_count, 2)
                self.assertFalse(args.batch_abort_event.is_set())

    def test_fatal_api_error_does_not_record_candidate_failure(self) -> None:
        row = {
            "instance_id": "example_repo_pr1",
            "document": "A sufficiently long source document. " * 5,
            "github_url": "https://github.com/example/repo",
            "parent_commit": "a" * 40,
            "repo": "repo",
            "user": "example",
        }
        args = SimpleNamespace(
            resume=False,
            overwrite_output=False,
            condition_seed_dir=None,
            keep_failed_stage=False,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(
                generator,
                "generate_audited_conditions",
                side_effect=llm_source.FatalAPIError(
                    402, '{"error":{"message":"Insufficient Balance"}}'
                ),
            ):
                with self.assertRaises(llm_source.FatalAPIError):
                    generator.generate_one(
                        row,
                        output_dir=root / "output",
                        work_dir=root / "work",
                        args=args,
                        provider="deepseek",
                        model="deepseek-v4-pro",
                        base_url="https://api.deepseek.example/v1",
                        api_key="secret-key-that-must-not-be-recorded",
                    )

            self.assertFalse(
                (root / "work" / "example_repo_pr1" / "run.json").exists()
            )

    def test_peer_error_after_batch_abort_remains_retryable(self) -> None:
        row = {
            "instance_id": "example_repo_pr2",
            "document": "A sufficiently long source document. " * 5,
            "github_url": "https://github.com/example/repo",
            "parent_commit": "a" * 40,
            "repo": "repo",
            "user": "example",
        }
        batch_abort_event = threading.Event()
        args = SimpleNamespace(
            resume=False,
            overwrite_output=False,
            condition_seed_dir=None,
            keep_failed_stage=False,
            batch_abort_event=batch_abort_event,
        )

        def abort_then_fail(**_kwargs: object) -> None:
            batch_abort_event.set()
            raise generator.CandidateError("concurrent local candidate error")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(
                generator,
                "generate_audited_conditions",
                side_effect=abort_then_fail,
            ):
                with self.assertRaises(llm_source.BatchAbortError):
                    generator.generate_one(
                        row,
                        output_dir=root / "output",
                        work_dir=root / "work",
                        args=args,
                        provider="deepseek",
                        model="deepseek-v4-pro",
                        base_url="https://api.deepseek.example/v1",
                        api_key="secret-key-that-must-not-be-recorded",
                    )

            self.assertFalse(
                (root / "work" / "example_repo_pr2" / "run.json").exists()
            )

    def test_resume_reuses_only_a_revalidated_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            stage_dir = run_dir / "stage"
            stage_dir.mkdir()
            (stage_dir / "attempt_01_raw.txt").write_text(
                '{"value": 7}',
                encoding="utf-8",
            )
            (stage_dir / "attempt_01_validation.json").write_text(
                '{"valid": true, "errors": []}',
                encoding="utf-8",
            )

            value = generator.call_candidate_stage(
                stage="stage",
                base_messages=[],
                normalize=lambda payload: int(payload["value"]),
                run_dir=run_dir,
                args=SimpleNamespace(resume=True, retries=1),
                provider="test",
                model="test",
                base_url="",
                api_key="",
            )

            self.assertEqual(value, 7)

    def test_generation_rejects_low_type_diversity(self) -> None:
        payload = {
            "considered_conditions": [
                {
                    "condition_id": f"kc_{index:03d}",
                    "type": "api_constraint",
                }
                for index in range(1, 7)
            ],
            "selected_condition_ids": [
                "kc_001",
                "kc_002",
                "kc_003",
                "kc_004",
            ],
        }
        with self.assertRaisesRegex(
            generator.CandidateError,
            "at least 3 distinct condition types",
        ):
            generator.normalize_condition_output(payload, "document")

    def test_complete_sample_needs_only_numbered_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            sample_dir = Path(temporary) / "sample"
            sample_dir.mkdir()
            for name in (
                "1_document_before.md",
                "2_deleted_parts.json",
                "3_document_after.md",
                "4_code_mapping.json",
            ):
                (sample_dir / name).touch()
            (sample_dir / "5_original_repo").mkdir()

            self.assertTrue(generator.is_complete_sample(sample_dir))
            self.assertFalse((sample_dir / "sample.json").exists())
            self.assertFalse((sample_dir / "metadata.json").exists())

    def test_generated_annotation_note_is_idempotent(self) -> None:
        note = "Dual mapping regenerated."
        prior = f"Owner-reviewed sample. {note} {note}"

        self.assertEqual(
            remap.merge_annotation_note(prior, note),
            f"Owner-reviewed sample. {note}",
        )

    def test_generation_and_remap_paths_must_be_disjoint(self) -> None:
        self.assertTrue(
            generator.paths_overlap(
                Path("SpecGAP"),
                Path("SpecGAP/run"),
            )
        )
        self.assertTrue(
            remap.paths_overlap(
                Path("SpecGAP"),
                Path("SpecGAP"),
            )
        )
        self.assertFalse(
            remap.paths_overlap(
                Path("SpecGAP"),
                Path("runs"),
            )
        )

    def test_filled_manual_comparison_is_detected(self) -> None:
        sample = {
            "code_mappings": {
                "comparison_items": [
                    {
                        "manual_comparison": {
                            "agreement": "consistent",
                            "notes": "checked by owner",
                        }
                    }
                ]
            }
        }
        self.assertTrue(remap.manual_comparison_is_filled(sample))

    def test_atomic_overwrite_rolls_back_on_publish_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage = root / "stage"
            destination = root / "sample"
            stage.mkdir()
            destination.mkdir()
            (stage / "new.txt").write_text("new", encoding="utf-8")
            (destination / "old.txt").write_text("old", encoding="utf-8")
            real_replace = Path.replace

            def fail_new_publish(path: Path, target: Path) -> Path:
                if path == stage and Path(target) == destination:
                    raise OSError("injected publish failure")
                return real_replace(path, target)

            with patch.object(Path, "replace", new=fail_new_publish):
                with self.assertRaises(OSError):
                    generator.atomic_publish(
                        stage, destination, overwrite=True
                    )

            self.assertEqual(
                (destination / "old.txt").read_text(encoding="utf-8"),
                "old",
            )
            self.assertTrue(stage.is_dir())

    def test_resume_identity_ignores_only_mapping_metadata(self) -> None:
        original = {
            "sample_id": "sample",
            "deleted_parts": [{"condition_id": "kc_001"}],
            "code_mappings": {"old": True},
            "annotation": {"model": "old"},
            "schema_version": "old",
        }
        remapped = copy.deepcopy(original)
        remapped["code_mappings"] = {"new": True}
        remapped["annotation"] = {"model": "new"}
        remapped["schema_version"] = "new"
        self.assertEqual(
            remap.immutable_sample_view(original),
            remap.immutable_sample_view(remapped),
        )
        remapped["deleted_parts"][0]["condition_id"] = "kc_999"
        self.assertNotEqual(
            remap.immutable_sample_view(original),
            remap.immutable_sample_view(remapped),
        )

    def test_resume_requires_both_routes_quality_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            route_a = (
                run_dir
                / "condition_code_mapping"
                / "condition_by_condition"
                / "kc_001"
            )
            route_b = (
                run_dir
                / "condition_code_mapping"
                / "holistic_alignment"
            )
            for route in (route_a, route_b):
                route.mkdir(parents=True)
                (route / "normalized_evidence.json").write_text(
                    "{}",
                    encoding="utf-8",
                )
                audit = route / "quality_audit"
                audit.mkdir()
                (audit / "attempt_01_validation.json").write_text(
                    '{"valid": true}',
                    encoding="utf-8",
                )
            self.assertTrue(
                remap.has_complete_quality_audit(run_dir, ["kc_001"])
            )
            (
                route_b
                / "quality_audit"
                / "attempt_01_validation.json"
            ).write_text('{"valid": false}', encoding="utf-8")
            self.assertFalse(
                remap.has_complete_quality_audit(run_dir, ["kc_001"])
            )


class SemanticCoverageGuardTests(unittest.TestCase):
    def _scope_mapping(self, excerpt: str) -> dict[str, object]:
        return {
            "condition_id": "kc_001",
            "mapping_status": "direct",
            "coverage": "full",
            "evidence": [
                {
                    "evidence_type": "implementation",
                    "relation": "implements",
                    "explanation": "scope boundary implementation",
                    "locations": [
                        {
                            "file_path": "src/example.py",
                            "symbol": {
                                "kind": "function",
                                "qualified_name": "build",
                            },
                            "excerpt": excerpt,
                        }
                    ],
                }
            ],
        }

    def test_full_mapping_cannot_omit_named_deleted_subclause(self) -> None:
        deleted_part = _deleted_part(
            "kc_001",
            "`TopHook` applies at the top while recursion keeps `root_schema`.",
        )
        mapping = self._scope_mapping("nested(root_schema=root_schema)")

        with self.assertRaisesRegex(generator.CandidateError, "TopHook"):
            generator.enforce_full_mapping_covers_explicit_code_terms(
                [mapping],
                [deleted_part],
                scope="test",
            )

    def test_full_mapping_accepts_evidence_covering_all_named_terms(self) -> None:
        deleted_part = _deleted_part(
            "kc_001",
            "`TopHook` applies at the top while recursion keeps `root_schema`.",
        )
        mapping = self._scope_mapping(
            "TopHook is forwarded; nested(root_schema=root_schema)"
        )

        generator.enforce_full_mapping_covers_explicit_code_terms(
            [mapping],
            [deleted_part],
            scope="test",
        )

    def test_scope_boundary_requires_hook_forwarding_and_nested_symbol(
        self,
    ) -> None:
        deleted_part = _deleted_part(
            "kc_001",
            "`__hook__` applies at the top-level but is not propagated; "
            "recursion keeps `root_schema`.",
        )
        mapping = self._scope_mapping(
            "def outer(__hook__): nested(root_schema=root_schema)"
        )

        with self.assertRaisesRegex(
            generator.CandidateError,
            "accepted interface and top-level forwarding",
        ):
            generator.enforce_full_mapping_covers_explicit_code_terms(
                [mapping],
                [deleted_part],
                scope="test",
            )

    def test_scope_boundary_accepts_both_sides_of_implementation(
        self,
    ) -> None:
        deleted_part = _deleted_part(
            "kc_001",
            "`__hook__` applies at the top-level but is not propagated; "
            "recursion keeps `root_schema`.",
        )
        mapping = self._scope_mapping(
            "def outer(__hook__): create(__hook__)"
        )
        mapping["evidence"][0]["locations"].append(
            {
                "file_path": "src/example.py",
                "symbol": {
                    "kind": "function",
                    "qualified_name": "nested",
                },
                "excerpt": "nested(root_schema=root_schema)",
            }
        )

        generator.enforce_full_mapping_covers_explicit_code_terms(
            [mapping],
            [deleted_part],
            scope="test",
        )

    def test_explanation_cannot_cite_unrecorded_code_line(self) -> None:
        mapping = {
            **self._scope_mapping("return value"),
            "mapping_explanation": "The passthrough occurs at line 140.",
        }
        mapping["evidence"][0]["locations"][0]["line_ranges"] = [
            {"start": 107, "end": 113}
        ]

        with self.assertRaisesRegex(
            generator.CandidateError,
            "line\\(s\\) 140-140",
        ):
            generator.enforce_explanation_line_references_are_evidenced(
                [mapping],
                scope="test",
            )

    def test_explanation_accepts_recorded_code_line(self) -> None:
        mapping = {
            **self._scope_mapping("return value"),
            "mapping_explanation": "The passthrough occurs at lines 140-141.",
        }
        mapping["evidence"][0]["locations"][0]["line_ranges"] = [
            {"start": 140, "end": 141}
        ]

        generator.enforce_explanation_line_references_are_evidenced(
            [mapping],
            scope="test",
        )


@unittest.skipUnless(
    REFERENCE_SAMPLE.is_dir(),
    "the checked-in qualified reference sample is unavailable",
)
class ValidatorRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        mapping_artifact = json.loads(
            (REFERENCE_SAMPLE / "4_code_mapping.json").read_text(
                encoding="utf-8"
            )
        )
        selection = json.loads(
            (REFERENCE_SAMPLE / "condition_selection.json").read_text(
                encoding="utf-8"
            )
        )
        cls.sample = {
            "code_mappings": mapping_artifact["code_mappings"],
            "selected_condition_ids": selection["selected_condition_ids"],
        }
        cls.repo_root = REFERENCE_SAMPLE / "5_original_repo"

    def mapping_with_gold_labels(self) -> dict[str, object]:
        mapping = copy.deepcopy(self.sample["code_mappings"])
        sources = (
            "condition_by_condition",
            "holistic_alignment",
            "manually_rewritten",
        )
        mapping["gold_labels"] = [
            {
                "condition_id": item["condition_id"],
                "mapping_explanation": (
                    f"Human-verified gold explanation for {item['condition_id']}."
                ),
                "gold_source": sources[index % len(sources)],
            }
            for index, item in enumerate(mapping["comparison_items"])
        ]
        return mapping

    def validate_mapping(self, mapping: dict[str, object]) -> list[str]:
        context = validator.ValidationContext(sample_dir=REFERENCE_SAMPLE)
        validator._validate_code_mappings(
            context,
            mapping,
            self.sample["selected_condition_ids"],
            self.repo_root,
        )
        return context.errors

    def test_current_dual_mapping_passes(self) -> None:
        self.assertEqual(self.validate_mapping(self.sample["code_mappings"]), [])

    def test_missing_gold_labels_remains_valid(self) -> None:
        mapping = copy.deepcopy(self.sample["code_mappings"])
        mapping.pop("gold_labels", None)

        self.assertEqual(self.validate_mapping(mapping), [])

    def test_valid_optional_gold_labels_pass(self) -> None:
        mapping = self.mapping_with_gold_labels()

        self.assertEqual(self.validate_mapping(mapping), [])

    def test_gold_labels_must_follow_comparison_order(self) -> None:
        mapping = self.mapping_with_gold_labels()
        mapping["gold_labels"][0], mapping["gold_labels"][1] = (
            mapping["gold_labels"][1],
            mapping["gold_labels"][0],
        )

        errors = self.validate_mapping(mapping)

        self.assertTrue(
            any("comparison_items condition_id order" in error for error in errors),
            errors,
        )

    def test_gold_label_condition_ids_must_be_unique(self) -> None:
        mapping = self.mapping_with_gold_labels()
        mapping["gold_labels"][1]["condition_id"] = mapping["gold_labels"][0][
            "condition_id"
        ]

        errors = self.validate_mapping(mapping)

        self.assertTrue(
            any("condition_id values must be unique" in error for error in errors),
            errors,
        )

    def test_gold_label_requires_exact_fields(self) -> None:
        mapping = self.mapping_with_gold_labels()
        del mapping["gold_labels"][0]["mapping_explanation"]
        mapping["gold_labels"][0]["unexpected"] = True

        errors = self.validate_mapping(mapping)

        self.assertTrue(
            any("missing keys: mapping_explanation" in error for error in errors),
            errors,
        )
        self.assertTrue(
            any("unexpected keys: unexpected" in error for error in errors),
            errors,
        )

    def test_gold_label_rejects_empty_explanation_and_unknown_source(
        self,
    ) -> None:
        mapping = self.mapping_with_gold_labels()
        mapping["gold_labels"][0]["mapping_explanation"] = "   "
        mapping["gold_labels"][0]["gold_source"] = "unknown"

        errors = self.validate_mapping(mapping)

        self.assertTrue(
            any("mapping_explanation: must not be empty" in error for error in errors),
            errors,
        )
        self.assertTrue(
            any("gold_source: must be one of" in error for error in errors),
            errors,
        )

    def test_wrong_condition_order_is_rejected(self) -> None:
        mapping = copy.deepcopy(self.sample["code_mappings"])
        mapping["comparison_items"][0], mapping["comparison_items"][1] = (
            mapping["comparison_items"][1],
            mapping["comparison_items"][0],
        )
        errors = self.validate_mapping(mapping)
        self.assertTrue(
            any("selected_condition_ids order" in error for error in errors),
            errors,
        )

    def test_wrong_route_evidence_prefix_is_rejected(self) -> None:
        mapping = copy.deepcopy(self.sample["code_mappings"])
        evidence = mapping["comparison_items"][0]["holistic_alignment"]["evidence"][0]
        evidence["evidence_id"] = evidence["evidence_id"].replace("_ha_", "_cbc_")
        errors = self.validate_mapping(mapping)
        self.assertTrue(
            any("must equal" in error and "evidence_id" in error for error in errors),
            errors,
        )

    def test_missing_mapping_field_is_rejected(self) -> None:
        mapping = copy.deepcopy(self.sample["code_mappings"])
        del mapping["comparison_items"][0]["condition_by_condition"][
            "related_tests"
        ]
        errors = self.validate_mapping(mapping)
        self.assertTrue(
            any("missing keys: related_tests" in error for error in errors),
            errors,
        )

    def test_false_holistic_input_scope_is_rejected(self) -> None:
        mapping = copy.deepcopy(self.sample["code_mappings"])
        mapping["methods"]["holistic_alignment"]["input_scope"] = (
            "deliberately_false_scope"
        )
        errors = self.validate_mapping(mapping)
        self.assertTrue(
            any(
                "complete_document_all_selected_conditions_and_repository"
                in error
                for error in errors
            ),
            errors,
        )

    def test_no_direct_mapping_requires_reason(self) -> None:
        mapping = copy.deepcopy(self.sample["code_mappings"])
        branch = mapping["comparison_items"][0]["condition_by_condition"]
        branch.update(_minimal_mapping(branch["condition_id"]))
        branch["no_direct_mapping_reason"] = None
        errors = self.validate_mapping(mapping)
        self.assertTrue(
            any(
                "no_direct_mapping requires a non-null reason object" in error
                for error in errors
            ),
            errors,
        )

    def test_fake_python_symbol_is_rejected(self) -> None:
        mapping = copy.deepcopy(self.sample["code_mappings"])
        location = mapping["comparison_items"][0]["condition_by_condition"][
            "evidence"
        ][0]["locations"][0]
        location["symbol"]["qualified_name"] = "TotallyFakeFunction"
        errors = self.validate_mapping(mapping)
        self.assertTrue(
            any("does not exist in the file" in error for error in errors),
            errors,
        )

    def test_fake_repository_scan_count_is_rejected(self) -> None:
        mapping = copy.deepcopy(self.sample["code_mappings"])
        mapping["methods"]["condition_by_condition"]["repository_scan"][
            "indexed_files"
        ] = 999
        mapping["methods"]["holistic_alignment"]["repository_scan"][
            "indexed_files"
        ] = 999
        errors = self.validate_mapping(mapping)
        self.assertTrue(
            any("deterministic repository index value" in error for error in errors),
            errors,
        )

    def test_full_direct_mapping_cannot_only_contradict(self) -> None:
        mapping = copy.deepcopy(self.sample["code_mappings"])
        branch = mapping["comparison_items"][0]["condition_by_condition"]
        for evidence in branch["evidence"]:
            if (
                evidence["evidence_type"] == "implementation"
                and evidence["strength"] == "direct"
            ):
                evidence["relation"] = "contradicts"
        errors = self.validate_mapping(mapping)
        self.assertTrue(
            any("positive implementation evidence" in error for error in errors),
            errors,
        )


if __name__ == "__main__":
    unittest.main()
