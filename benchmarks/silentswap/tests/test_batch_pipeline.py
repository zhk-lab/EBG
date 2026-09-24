import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import build_batch_samples as batch  # noqa: E402


class BatchPipelineTests(unittest.TestCase):
    def test_batch_defaults(self):
        self.assertEqual(100, batch.TARGET_COUNT)
        self.assertEqual(2, batch.DEFAULT_WORKERS)
        self.assertEqual(1000, batch.DEFAULT_CANDIDATE_LIMIT)
        self.assertEqual("docker", batch.DEFAULT_DOCKER)
        self.assertEqual("linux/amd64", batch.DEFAULT_DOCKER_PLATFORM)

    def test_resume_expands_a_smaller_candidate_cache(self):
        def source_record(index):
            return {
                "instance_id": f"case-{index}",
                "github_url": f"https://example.com/repo-{index}",
                "parent_commit": f"commit-{index}",
                "document": f"document-{index}",
                "image_url": f"example/image:{index}",
                "workdir": f"/workspace/repo-{index}",
                "license_spdx_id": "MIT",
                "package_setup_files": ["pyproject.toml"],
            }

        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "source_records.jsonl"
            cache.write_text(json.dumps(source_record(0)) + "\n", encoding="utf-8")
            dataset = "".join(
                json.dumps(source_record(index)) + "\n" for index in range(3)
            ).encode()

            with patch.object(batch.core, "HfFileSystem") as filesystem:
                filesystem.return_value.open.return_value = io.BytesIO(dataset)
                records = batch.load_candidate_records(cache, 3, 1, resume=True)

        self.assertEqual(3, len(records))
        self.assertEqual(["case-0", "case-1", "case-2"], [r["instance_id"] for r in records])

    def test_record_projection_keeps_docker_environment(self):
        record = {
            "instance_id": "case-1",
            "github_url": "https://example.com/repo",
            "parent_commit": "abc",
            "document": "spec",
            "image_url": "aweaiteam/denovoswe:case-1",
            "workdir": "/workspace/repo",
            "license_spdx_id": "MIT",
        }

        projected = batch.record_projection(record)

        self.assertEqual(record, projected)

    def test_docker_command_mounts_repository_at_instance_workdir(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory) / "repo"
            repo.mkdir()
            semantic_test = Path(directory) / "semantic_test.py"
            semantic_test.write_text("pass\n", encoding="utf-8")
            runtime = {
                "docker": "docker",
                "platform": "linux/amd64",
                "image": "aweaiteam/denovoswe:case-1",
                "workdir": "/workspace/repo",
            }

            command = batch.docker_command(
                runtime,
                repo,
                ["python", "/tmp/silentswap_semantic_test.py"],
                [(semantic_test, "/tmp/silentswap_semantic_test.py", "ro")],
            )

        self.assertEqual("docker", command[0])
        self.assertIn(f"{repo.resolve()}:/workspace/repo:rw", command)
        self.assertIn(
            f"{semantic_test.resolve()}:/tmp/silentswap_semantic_test.py:ro",
            command,
        )
        self.assertIn("PYTHONPATH=/workspace/repo", command)
        self.assertEqual(
            ["--entrypoint", "python", "aweaiteam/denovoswe:case-1", "/tmp/silentswap_semantic_test.py"],
            command[-4:],
        )

    def test_docker_environment_fields_are_required(self):
        record = {
            "instance_id": "case-1",
            "github_url": "https://example.com/repo",
            "parent_commit": "abc",
            "document": "spec",
            "image_url": "aweaiteam/denovoswe:case-1",
            "package_setup_files": ["pyproject.toml"],
        }

        self.assertFalse(batch.eligible_record(record))
        record["workdir"] = "/workspace/repo"
        self.assertTrue(batch.eligible_record(record))

    def test_prepare_runtime_pulls_image_and_runs_baseline_in_docker(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory) / "repo"
            repo.mkdir()
            record = {
                "image_url": "aweaiteam/denovoswe:case-1",
                "workdir": "/workspace/repo",
            }
            results = [
                {"exit_code": 0, "stdout": "", "stderr": ""},
                {"exit_code": 0, "stdout": "3 passed\n", "stderr": ""},
                {"exit_code": 0, "stdout": "Python 3.11.9\n", "stderr": ""},
            ]

            with patch.object(batch.core, "run", side_effect=results) as run:
                runtime = batch.prepare_runtime(
                    record, repo, "docker", "linux/amd64",
                )

        self.assertEqual("Python 3.11.9", runtime["python_version"])
        self.assertEqual(
            [
                "docker", "pull", "--platform", "linux/amd64",
                "aweaiteam/denovoswe:case-1",
            ],
            run.call_args_list[0].args[0],
        )
        self.assertEqual(["python", "-m", "pytest", "-q"], runtime["test_command"])
        self.assertIn("docker", run.call_args_list[1].args[0])
        self.assertIn("aweaiteam/denovoswe:case-1", run.call_args_list[1].args[0])

    def test_remove_docker_image_records_the_cleanup_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory) / "repo"
            repo.mkdir()
            runtime = {
                "docker": "docker",
                "image": "aweaiteam/denovoswe:case-1",
                "image_removal_attempted": False,
            }
            result = {"exit_code": 0, "stdout": "Untagged\n", "stderr": ""}

            with patch.object(batch.core, "run", return_value=result) as run:
                removal = batch.remove_docker_image(runtime, repo)

        self.assertEqual(result, removal)
        self.assertTrue(runtime["image_removal_attempted"])
        self.assertEqual(result, runtime["image_removal"])
        self.assertEqual(
            ["docker", "image", "rm", "aweaiteam/denovoswe:case-1"],
            run.call_args.args[0],
        )

    def test_run_record_removes_image_when_runtime_preparation_times_out(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            repo.mkdir()
            args = SimpleNamespace(
                work=root / "work",
                output=root / "data",
                docker="docker",
                docker_platform="linux/amd64",
                timeout=600,
                resume=True,
            )
            record = {
                "instance_id": "case-1",
                "image_url": "aweaiteam/denovoswe:case-1",
            }
            cleanup = {"exit_code": 0, "stdout": "Deleted", "stderr": ""}

            with (
                patch.object(batch, "prepare_repository", return_value=repo),
                patch.object(
                    batch,
                    "prepare_runtime",
                    side_effect=subprocess.TimeoutExpired(["docker", "run"], 1200),
                ),
                patch.object(batch, "remove_docker_image", return_value=cleanup) as remove,
                patch.object(batch, "append_progress"),
            ):
                success, case_id = batch.run_record(
                    record, args, "secret", root / "progress.jsonl",
                )

        self.assertFalse(success)
        self.assertEqual("case-1", case_id)
        runtime = remove.call_args.args[0]
        self.assertEqual("aweaiteam/denovoswe:case-1", runtime["image"])
        self.assertEqual("docker", runtime["docker"])

    def test_resume_counts_only_docker_validated_samples(self):
        local_case = {
            "status": "validated",
            "test_environment": {"local_python": "Python 3.12.0"},
        }
        docker_case = {
            "status": "validated",
            "test_environment": {
                "docker_platform": "linux/amd64",
                "container_workdir": "/workspace/repo",
            },
        }

        self.assertFalse(batch.is_docker_validated(local_case))
        self.assertTrue(batch.is_docker_validated(docker_case))

    def test_numbered_sample_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for number, case_id in ((2, "case-2"), (1, "case-1")):
                sample = root / str(number)
                sample.mkdir()
                (sample / "case.json").write_text(
                    json.dumps({"case_id": case_id}), encoding="utf-8",
                )
            (root / "not-a-sample").mkdir()

            self.assertEqual(
                ["1", "2"],
                [path.name for path in batch.core.sample_directories(root)],
            )
            self.assertEqual(3, batch.core.next_sample_number(root))
            self.assertEqual(
                root / "2", batch.core.find_sample_directory(root, "case-2"),
            )

            (root / "1").rename(root / "4")
            self.assertEqual(1, batch.core.next_sample_number(root))

    def test_source_diversity_accepts_distinct_instances_and_repositories(self):
        records = [
            {"instance_id": f"case-{index}", "github_url": f"https://example.com/repo-{index}"}
            for index in range(3)
        ]

        batch.validate_source_diversity(records, 3)

    def test_source_diversity_rejects_avoidable_repository_repetition(self):
        records = [
            {"instance_id": "case-1", "github_url": "https://example.com/repo-1"},
            {"instance_id": "case-2", "github_url": "https://example.com/repo-2"},
            {"instance_id": "case-3", "github_url": "https://example.com/repo-2"},
        ]

        with self.assertRaises(batch.core.PipelineError):
            batch.validate_source_diversity(records, 2)

    def test_instance_ids_must_be_unique(self):
        records = [
            {"instance_id": "same", "github_url": "https://example.com/repo-1"},
            {"instance_id": "same", "github_url": "https://example.com/repo-2"},
        ]

        with self.assertRaises(batch.core.PipelineError):
            batch.validate_source_diversity(records, 2)

    def test_repositories_repeat_only_when_unique_sources_are_insufficient(self):
        unique = [
            {"instance_id": "a", "github_url": "repo-a"},
            {"instance_id": "b", "github_url": "repo-b"},
        ]
        repeated = [
            {"instance_id": "a-2", "github_url": "repo-a"},
            {"instance_id": "a-3", "github_url": "repo-a"},
        ]

        enough_unique = batch.select_diverse_records(unique, repeated, 3, 2)
        insufficient_unique = batch.select_diverse_records(unique, repeated, 3, 3)

        self.assertEqual(["a", "b"], [record["instance_id"] for record in enough_unique])
        self.assertEqual(
            ["a", "b", "a-2"],
            [record["instance_id"] for record in insufficient_unique],
        )

    def test_selector_text_must_be_unique(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            (repo / "src.py").write_text("value = 1\n", encoding="utf-8")
            (repo / "tests").mkdir()
            (repo / "tests" / "test_src.py").write_text("pass\n", encoding="utf-8")
            selection = {
                "original_document_text": "same",
                "original_semantics": "meaning",
                "target_files": ["src.py"],
                "related_test_files": ["tests/test_src.py"],
                "swap_direction": "change behavior",
                "selection_reason": "not covered",
            }

            with self.assertRaises(batch.core.PipelineError):
                batch.validate_selection(selection, {"document": "same and same"}, repo)


if __name__ == "__main__":
    unittest.main()
