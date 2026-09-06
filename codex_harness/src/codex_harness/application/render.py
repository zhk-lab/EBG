"""Compact YAML rendering and token estimates."""

from __future__ import annotations

from importlib.resources import files
from typing import Any

import tiktoken
import yaml


PROMPT = files("codex_harness").joinpath("prompts", "disclose.md").read_text(encoding="utf-8").rstrip("\r\n")


class EvidenceDumper(yaml.SafeDumper):
    def increase_indent(self, flow: bool = False, indentless: bool = False) -> None:
        return super().increase_indent(flow, False)

    def ignore_aliases(self, data: Any) -> bool:
        return True


def _string(dumper: yaml.SafeDumper, value: str) -> yaml.ScalarNode:
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style="|" if "\n" in value else None)


EvidenceDumper.add_representer(str, _string)


def render(value: Any) -> str:
    return yaml.dump(value, Dumper=EvidenceDumper, allow_unicode=True, sort_keys=False, width=110, indent=2)


def tokens(text: str) -> int:
    return len(tiktoken.get_encoding("o200k_base").encode(text, disallowed_special=()))

