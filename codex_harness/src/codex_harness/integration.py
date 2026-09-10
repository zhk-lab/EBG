"""Generate reviewable MCP, Hook and Skill installation files."""

from __future__ import annotations

import json
import shlex
from importlib.resources import files
from pathlib import Path


def write_integration(output: Path, state_dir: Path, python: str, *, token_budget: int = 12_000,
                      review_call_threshold: int = 10, review_seconds_threshold: float = 300) -> None:
    output.mkdir(parents=True, exist_ok=True)
    python = str(Path(python).resolve())
    args = ["-m", "codex_harness", "--state-dir", str(state_dir.resolve()), "--token-budget", str(token_budget)]
    args += ["--review-call-threshold", str(review_call_threshold),
             "--review-seconds-threshold", str(review_seconds_threshold)]
    config = "[mcp_servers.beg_disclose]\ncommand = " + json.dumps(python) + "\nargs = " + json.dumps(args + ["serve"]) + "\ntool_timeout_sec = 300\n"
    (output / "config.toml").write_text(config, encoding="utf-8")
    command_args = [python, *args, "hook"]
    # PowerShell requires & to invoke a quoted executable. Literal quoting also
    # keeps spaces, apostrophes and $ in paths from being interpreted as code.
    windows = '& ' + ' '.join("'" + arg.replace("'", "''") + "'" for arg in command_args)
    unix = shlex.join(command_args)
    hooks = {name: [{"hooks": [{"type": "command", "command": unix, "commandWindows": windows, "timeout": 300}]}]
             for name in ("SessionStart", "UserPromptSubmit", "PreToolUse", "PostToolUse", "Stop")}
    (output / "hooks.json").write_text(json.dumps({"hooks": hooks}, indent=2), encoding="utf-8")
    skills = files("codex_harness").joinpath("skills")
    for relative in ("beg-review/SKILL.md", "beg-review/references/tool-workflows.md",
                     "beg-review/references/review-rules.md", "beg-ambiguity/SKILL.md",
                     "beg-adjustment/SKILL.md", "beg-result-review/SKILL.md"):
        destination = output / "skills" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(skills.joinpath(relative).read_bytes())
