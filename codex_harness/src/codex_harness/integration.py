"""Generate reviewable MCP, Hook and Skill installation files."""

from __future__ import annotations

import json
import shlex
from importlib.resources import files
from pathlib import Path


def write_integration(output: Path, state_dir: Path, python: str, *, token_budget: int = 12_000) -> None:
    output.mkdir(parents=True, exist_ok=True)
    python = str(Path(python).resolve())
    args = ["-m", "codex_harness", "--state-dir", str(state_dir.resolve()), "--token-budget", str(token_budget)]
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
    target = output / "skills" / "beg-disclose"
    skill = files("codex_harness").joinpath("skills", "beg-disclose")
    for relative in ("SKILL.md", "references/tool-workflows.md"):
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(skill.joinpath(relative).read_bytes())
