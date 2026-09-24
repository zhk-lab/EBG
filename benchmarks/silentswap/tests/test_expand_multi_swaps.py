from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "expand_multi_swaps", ROOT / "scripts" / "expand_multi_swaps.py"
)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_parse_sample_numbers_supports_ranges_and_deduplicates() -> None:
    assert module.parse_sample_numbers("1,3-5,4") == [1, 3, 4, 5]


def test_load_original_builder_uses_trajectory_output(
    tmp_path: Path,
) -> None:
    (tmp_path / "construction_trajectory.jsonl").write_text(
        json.dumps(
            {
                "actor": "deepseek_builder",
                "action": "construct_swap",
                "output": {
                    "original_document_text": "stale original detail",
                    "replacement_document_text": "stale swapped detail",
                },
            }
        ),
        encoding="utf-8",
    )

    builder = module.load_original_builder(tmp_path)

    assert builder["original_document_text"] == "stale original detail"
    assert builder["replacement_document_text"] == "stale swapped detail"


def test_runtime_uses_explicit_docker_mirror(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "case.json").write_text(
        json.dumps(
            {
                "test_environment": {
                    "docker_platform": "linux/amd64",
                    "source_image": "owner/image:tag",
                    "container_workdir": "/workspace/repo",
                    "test_command": "python -m pytest -q",
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SILENTSWAP_DOCKER_MIRROR", "mirror.example/")

    runtime = module.runtime_from_case(tmp_path, "docker")

    assert runtime["image"] == "owner/image:tag"
    assert runtime["pull_image"] == "mirror.example/owner/image:tag"


def test_preloaded_image_must_match_case_platform(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = {
        "docker": "docker",
        "platform": "linux/amd64",
        "image": "owner/image:tag",
        "pull_image": "owner/image:tag",
        "preloaded_image": True,
    }
    calls: list[list[str]] = []

    def fake_run(command, cwd, timeout):
        calls.append(command)
        return {"exit_code": 0, "stdout": "linux/arm64\n", "stderr": ""}

    monkeypatch.setattr(module, "run", fake_run)

    with pytest.raises(module.ExpansionError, match="platform mismatch"):
        module.pull_and_baseline(runtime, tmp_path)

    assert calls[0][:3] == ["docker", "image", "inspect"]


def test_crane_transport_loads_image_and_removes_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = {
        "docker": "docker",
        "crane": "crane",
        "platform": "linux/amd64",
        "image": "owner/image:tag",
        "pull_image": "owner/image:tag",
        "preloaded_image": False,
        "crane_transport": True,
        "test_command": ["python", "-m", "pytest", "-q"],
    }
    commands: list[list[str]] = []

    def fake_run(command, cwd, timeout):
        commands.append(command)
        if command[:2] == ["crane", "pull"]:
            Path(command[-1]).write_text("archive", encoding="utf-8")
        return {"exit_code": 0, "stdout": "", "stderr": ""}

    monkeypatch.setattr(module, "run", fake_run)
    monkeypatch.setattr(
        module,
        "run_in_docker",
        lambda *args, **kwargs: {"exit_code": 0, "stdout": "", "stderr": ""},
    )

    module.pull_and_baseline(runtime, tmp_path)

    assert commands[0][:4] == ["crane", "pull", "--platform", "linux/amd64"]
    assert commands[1][:2] == ["docker", "load"]
    assert not (tmp_path.parent / "source-image.tar").exists()


def test_trim_docker_storage_records_serialized_trim(monkeypatch: pytest.MonkeyPatch) -> None:
    result = {
        "command": "colima ssh -- sudo fstrim -av",
        "exit_code": 0,
        "stdout": "trimmed",
        "stderr": "",
        "skipped": False,
    }
    monkeypatch.setattr(module.docker_trim, "trim", lambda: result)
    runtime = {"image": "owner/image:tag"}

    assert module.trim_docker_storage(runtime) == result
    assert runtime["disk_trim"] == result


@pytest.mark.parametrize("value", ["", "0", "100-101"])
def test_parse_sample_numbers_rejects_out_of_range_values(value: str) -> None:
    with pytest.raises(Exception):
        module.parse_sample_numbers(value)


def test_map_changed_lines_accounts_for_earlier_insertions() -> None:
    isolated = "def f():\n    return 2\n"
    final = "# added elsewhere\ndef f():\n    return 2\n"

    assert module.map_changed_lines(isolated, final, {2}) == {3}


def test_non_python_localization_uses_module_symbol() -> None:
    assert module.symbol_for_responsible_lines(
        "templates/field.jinja2", "{{ invalid as python }}\n", {1}
    ) == {"kind": "module", "qualified_name": []}


def test_failure_fingerprint_uses_last_diagnostic_line() -> None:
    result = {
        "stdout": "observed changed behavior\n",
        "stderr": (
            "/workspace/pkg.py:4: UserWarning: deprecated API\n"
            "  import old_api\n"
            "Traceback (most recent call last):\n"
            "AssertionError: expected old behavior\n"
        ),
    }

    assert module.failure_fingerprint(result) == "AssertionError: expected old behavior"


def test_failure_fingerprint_ignores_warning_only_stderr() -> None:
    result = {
        "stdout": "observed changed behavior\nFAIL: semantic mismatch\n",
        "stderr": (
            "/workspace/pkg.py:4: UserWarning: deprecated API\n"
            "  import old_api\n"
        ),
    }

    assert module.failure_fingerprint(result) == "FAIL: semantic mismatch"


def test_failure_fingerprint_rejects_import_failures() -> None:
    result = {
        "stdout": "",
        "stderr": "ModuleNotFoundError: No module named 'dependency'\n",
    }

    with pytest.raises(module.CandidateRejected, match="environment or code-loading"):
        module.failure_fingerprint(result)


def test_failure_fingerprint_allows_expected_caught_import_error() -> None:
    result = {
        "stdout": (
            "names('broken') raised ImportError: expected broken candidate\n"
            "FAIL: ImportError from one candidate aborted discovery\n"
        ),
        "stderr": "",
    }

    assert module.failure_fingerprint(result) == (
        "FAIL: ImportError from one candidate aborted discovery"
    )


def test_combined_import_failure_regenerates_interfering_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = iter(
        [
            {"exit_code": 0, "stdout": "original passes\n", "stderr": ""},
            {
                "exit_code": 1,
                "stdout": "",
                "stderr": "AssertionError: own semantic difference\n",
            },
            {"exit_code": 0, "stdout": "without own swap passes\n", "stderr": ""},
            {
                "exit_code": 1,
                "stdout": "",
                "stderr": "ModuleNotFoundError: No module named 'broken'\n",
            },
        ]
    )
    runtime = {
        "original": tmp_path / "original",
        "state": tmp_path / "state",
        "test_command": ["python", "-m", "pytest", "-q"],
    }
    runtime["original"].mkdir()
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    records = [
        {
            "slot": 2,
            "builder": {
                "semantic_test": "assert True",
                "target_file": "module.py",
                "original_code_text": "old",
                "replacement_code_text": "new",
            },
        }
    ]
    monkeypatch.setattr(
        module,
        "run_in_docker",
        lambda *args, **kwargs: {"exit_code": 0, "stdout": "suite passes", "stderr": ""},
    )
    monkeypatch.setattr(module, "semantic_test", lambda *args, **kwargs: next(results))
    monkeypatch.setattr(module, "clone_repository", lambda source, target: target)
    monkeypatch.setattr(module, "apply_builder", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "diagnose_suppressing_slot", lambda *args, **kwargs: 4)

    with pytest.raises(module.CombinationRejected) as raised:
        module.counterfactual_verification(tmp_path, runtime, candidate, records)

    assert raised.value.regenerate_from_slot == 4
    assert "combined repository" in str(raised.value)


def test_isolated_import_failure_regenerates_own_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = iter(
        [
            {"exit_code": 0, "stdout": "original passes\n", "stderr": ""},
            {
                "exit_code": 1,
                "stdout": "",
                "stderr": "ModuleNotFoundError: No module named 'broken'\n",
            },
        ]
    )
    runtime = {
        "original": tmp_path / "original",
        "state": tmp_path / "state",
        "test_command": ["python", "-m", "pytest", "-q"],
    }
    runtime["original"].mkdir()
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    records = [
        {
            "slot": 3,
            "builder": {
                "semantic_test": "assert True",
                "target_file": "module.py",
                "original_code_text": "old",
                "replacement_code_text": "new",
            },
        }
    ]
    monkeypatch.setattr(
        module,
        "run_in_docker",
        lambda *args, **kwargs: {"exit_code": 0, "stdout": "suite passes", "stderr": ""},
    )
    monkeypatch.setattr(module, "semantic_test", lambda *args, **kwargs: next(results))
    monkeypatch.setattr(module, "clone_repository", lambda source, target: target)
    monkeypatch.setattr(module, "apply_builder", lambda *args, **kwargs: None)

    with pytest.raises(module.CombinationRejected) as raised:
        module.counterfactual_verification(tmp_path, runtime, candidate, records)

    assert raised.value.regenerate_from_slot == 3
    assert "isolated semantic test" in str(raised.value)


def test_failure_fingerprint_normalizes_dynamic_values() -> None:
    first = {
        "stdout": "FAIL: invalid word 'alpha beta' at offset 12\n",
        "stderr": "",
    }
    second = {
        "stdout": "FAIL: invalid word 'gamma delta' at offset 37\n",
        "stderr": "",
    }

    assert module.failure_fingerprint(first) == module.failure_fingerprint(second)


def test_failure_fingerprint_normalizes_temporary_directory_only() -> None:
    first = {
        "stdout": "",
        "stderr": (
            "AssertionError: Expected /tmp/tmpkn0_z0up/real/foo.nix, "
            "got /tmp/tmpkn0_z0up/link/foo.nix\n"
        ),
    }
    second = {
        "stdout": "",
        "stderr": (
            "AssertionError: Expected /tmp/tmp6mpoq_yu/real/foo.nix, "
            "got /tmp/tmp6mpoq_yu/link/foo.nix\n"
        ),
    }

    fingerprint = module.failure_fingerprint(first)
    assert fingerprint == module.failure_fingerprint(second)
    assert "/real/foo.nix" in fingerprint
    assert "/link/foo.nix" in fingerprint
