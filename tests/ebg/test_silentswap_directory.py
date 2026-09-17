from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ebg.behavior_directory import _RepoFile, build_ranked_directory
from ebg.silentswap_directory import rank_files
from tests.ebg.test_behavior_directory import assemble
from tests.support import ProjectTemporaryDirectory, make_repo_bundle


def file(path: str, sections: tuple[int, ...], behaviors: int = 1) -> _RepoFile:
    return _RepoFile(path, path, "source", behaviors, (), (), sections)


class SilentSwapDirectoryTests(unittest.TestCase):
    def test_broad_configuration_matches_do_not_displace_production_source(self) -> None:
        source = file("pkg/api.py", (0,))
        config = replace(
            file(".devcontainer/devcontainer.json", tuple(range(20))),
            artifact_kind="configuration",
        )
        self.assertEqual(rank_files([config, source]), [source, config])

    def test_artifact_tiers_keep_their_own_contract_coverage(self) -> None:
        source = file("pkg/api.py", (0, 1, 2))
        first = replace(file("a.toml", (0, 1)), artifact_kind="configuration")
        second = replace(file("b.toml", (3,)), artifact_kind="configuration")
        self.assertEqual(rank_files([second, source, first]), [source, first, second])

    def test_specific_contracts_outrank_large_overview_only_file(self) -> None:
        overview = file("overview.py", (0,), behaviors=100)
        focused = file("focused.py", (0, 1, 2))
        self.assertEqual(rank_files([overview, focused])[0], focused)

    def test_new_contracts_precede_redundant_coverage(self) -> None:
        first = file("a.py", (0, 1, 2))
        repeated = file("b.py", (0, 1, 2))
        independent = file("c.py", (3,))
        self.assertEqual(
            rank_files([repeated, independent, first]),
            [first, independent, repeated],
        )

    def test_preserves_every_file_with_stable_ties_and_no_input_mutation(self) -> None:
        files = [file("b.py", ()), file("a.py", ())]
        self.assertEqual([f.path for f in rank_files(files)], ["a.py", "b.py"])
        self.assertEqual([f.path for f in files], ["b.py", "a.py"])
        self.assertEqual(rank_files([]), [])

    def test_only_silentswap_uses_coverage_ranking(self) -> None:
        repository = {
            "a.py": "def a():\n    return 1\n",
            "b.py": "def b():\n    return 2\n",
        }
        document = "# First\nUse a.py.\n# Second\nUse b.py.\n"
        for benchmark in ("specgap", "silentswap"):
            with self.subTest(benchmark=benchmark), ProjectTemporaryDirectory() as root:
                bundle, graph = assemble(
                    make_repo_bundle(
                        root, benchmark=benchmark, repository_files=repository,
                        document=document,
                    )
                )
                with patch(
                    "ebg.silentswap_directory.rank_files",
                    side_effect=lambda files: list(reversed(files)),
                ) as rank:
                    result = build_ranked_directory(bundle, graph, count_tokens=len)
                self.assertEqual(rank.called, benchmark == "silentswap")
                self.assertEqual(
                    result["entries"][0]["path"],
                    "b.py" if benchmark == "silentswap" else "a.py",
                )


if __name__ == "__main__":
    unittest.main()
