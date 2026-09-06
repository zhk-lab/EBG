"""Source-backed requirement normalization; never infer requirements from code."""

from __future__ import annotations

import re
from typing import Any

from .storage import HarnessError


def source_ref(source: dict[str, Any], start: int, end: int) -> dict[str, Any]:
    content = source["content"]
    return {
        "source_id": source["id"], "start": start, "end": end,
        "source": f"{source['id']} {source['label']}@{content.count(chr(10), 0, start) + 1}-{content.count(chr(10), 0, max(start, end - 1)) + 1}",
        "kind": source["kind"], "content": content[start:end],
    }


def normalize_requirements(
    values: list[dict[str, Any]], sources: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Accept Codex's checklist, resolving every quote against stored originals."""
    if not values:
        raise HarnessError("requirements must contain at least one source-backed item.")
    known = {item["id"]: item for item in sources}
    result = []
    ids: set[str] = set()
    for value in values:
        identifier = value.get("id", "")
        check = value.get("check", "")
        if not isinstance(identifier, str) or not re.fullmatch(r"R[1-9][0-9]*", identifier) or identifier in ids:
            raise HarnessError("Each requirement needs a unique R-number, such as R1.")
        if not isinstance(check, str) or not check.strip() or not value.get("refs"):
            raise HarnessError(f"{identifier} needs check text and at least one original source reference.")
        refs = []
        for ref in value["refs"]:
            source = known.get(ref.get("source_id"))
            if source is None:
                raise HarnessError(f"Unknown requirement source: {ref.get('source_id')}")
            quote = ref.get("quote")
            if not isinstance(quote, str) or not quote.strip():
                raise HarnessError("Each source reference needs a nonempty verbatim quote.")
            content = source["content"]
            start = ref.get("start", content.find(quote))
            if not isinstance(start, int) or start < 0 or content[start:start + len(quote)] != quote:
                raise HarnessError(f"Quote is not verbatim in {source['id']}.")
            if "start" not in ref and content.find(quote, start + 1) >= 0:
                raise HarnessError("Quote occurs more than once; supply its character start offset.")
            resolved = source_ref(source, start, start + len(quote))
            origin = ref.get('origin')
            if origin is not None:
                if source['kind'] != 'plan' or origin not in {'user_plan', 'agent_plan', 'unknown'}:
                    raise HarnessError("origin applies only to Plan references: user_plan, agent_plan or unknown.")
                resolved['kind'] = origin if origin != 'unknown' else 'plan'
                resolved['source'] += f' ({origin})'
            refs.append(resolved)
        ids.add(identifier)
        result.append({"id": identifier, "check": check.strip(), "refs": refs})
    return result
