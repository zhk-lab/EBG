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
        for index, ref in enumerate(value["refs"]):
            source = known.get(ref.get("source_id"))
            if source is None:
                raise HarnessError(f"Unknown requirement source: {ref.get('source_id')}")
            quote = ref.get("quote")
            if not isinstance(quote, str) or not quote.strip():
                raise HarnessError("Each source reference needs a nonempty verbatim quote.")
            content = source["content"]
            start = ref.get("start", content.find(quote))
            if not isinstance(start, int):
                raise HarnessError(f'{identifier} refs[{index}]: start must be an integer character offset.')
            end = start + len(quote)
            ambiguous = False
            valid = start >= 0 and content[start:end] == quote
            if not valid and ('\r\n' in content or '\r\n' in quote):
                # JSON/YAML readers may present CRLF as LF. Match only that
                # presentation difference; references keep the original bytes.
                chars = list(re.finditer(r'\r\n|[^\r]|\r', content))
                normalized = ''.join(m.group().replace('\r\n', '\n') for m in chars)
                offsets = [m.start() for m in chars] + [len(content)]
                normalized_quote = quote.replace('\r\n', '\n')
                position = (offsets.index(start) if 'start' in ref and start in offsets
                            else -1 if 'start' in ref else normalized.find(normalized_quote))
                if position >= 0 and normalized[position:position + len(normalized_quote)] == normalized_quote:
                    start, end = offsets[position], offsets[position + len(normalized_quote)]
                    ambiguous = 'start' not in ref and normalized.find(normalized_quote, position + 1) >= 0
                    valid = True
            if not valid:
                raise HarnessError(
                    f"{identifier} refs[{index}]: quote is not verbatim in {source['id']}. "
                    "Copy a continuous original-language span; use separate refs for disjoint spans. "
                    "Omit start unless the quote repeats; do not guess offsets, translate or insert ellipses.")
            if ambiguous or ("start" not in ref and content.find(quote, start + 1) >= 0):
                raise HarnessError("Quote occurs more than once; supply its character start offset.")
            resolved = source_ref(source, start, end)
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
