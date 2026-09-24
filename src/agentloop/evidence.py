"""Structured evidence blocks shared by both AgentLoop backends."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Literal

from evaluation_core.contracts import EvidenceSpan


BehaviorRenderKey = tuple[
    str,
    str | None,
    tuple[str, ...],
    tuple[tuple[int, int], ...],
    str,
]
_MAX_FULL_SOURCE_ANCHORS = 8


@dataclass(frozen=True, slots=True)
class RelatedUnit:
    label: str
    unit_id: str
    name: str
    via: str | None = None


@dataclass(frozen=True, slots=True)
class BehaviorEvidence:
    labels: tuple[str, ...]
    start: int
    end: int
    text: str
    owner: str | None = None
    ranges: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True, slots=True)
class BehaviorPathStep:
    label: str
    ranges: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class BehaviorPathAnchor:
    owner: str
    steps: tuple[BehaviorPathStep, ...]


@dataclass(frozen=True, slots=True)
class DocumentEvidence:
    title: str
    section: str
    excerpt: str


@dataclass(frozen=True, slots=True)
class PackageMember:
    symbol: str
    start: int
    end: int
    focus: bool = False


@dataclass(frozen=True, slots=True)
class SourceRegion:
    """One complete, continuous region rendered from a repository file."""

    symbols: tuple[str, ...]
    start: int
    end: int
    source: str
    path: str | None = None


@dataclass(frozen=True, slots=True)
class EvidenceUnit:
    unit_id: str
    unit_kind: Literal["file", "package", "symbol", "source", "local_graph"]
    name: str
    span: EvidenceSpan
    source: str
    behavior_evidence: tuple[BehaviorEvidence, ...] = ()
    document_title: str | None = None
    document_section: str | None = None
    document_excerpt: str | None = None
    code_title: str | None = None
    related: tuple[RelatedUnit, ...] = ()
    documents: tuple[DocumentEvidence, ...] = ()
    members: tuple[PackageMember, ...] = ()
    behavior_total: int = 0
    member_total: int = 0
    document_total: int = 0
    source_regions: tuple[SourceRegion, ...] = ()
    behavior_paths: tuple[BehaviorPathAnchor, ...] = ()
    show_document_alignment: bool = False
    groundable: bool = True

    @property
    def source_key(self) -> tuple[Any, ...]:
        if self.source_regions:
            return (
                self.span.path,
                tuple(
                    (region.path, region.start, region.end, region.source)
                    for region in self.source_regions
                ),
            )
        return (self.span.path, self.span.start, self.span.end, self.source)

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "unit_kind": self.unit_kind,
            "name": self.name,
            "span": asdict(self.span),
            "source": self.source,
            "behavior_evidence": [asdict(item) for item in self.behavior_evidence],
            "document_title": self.document_title,
            "document_section": self.document_section,
            "document_excerpt": self.document_excerpt,
            "code_title": self.code_title,
            "related": [asdict(item) for item in self.related],
            "documents": [asdict(item) for item in self.documents],
            "members": [asdict(item) for item in self.members],
            "behavior_total": self.behavior_total,
            "member_total": self.member_total,
            "document_total": self.document_total,
            "source_regions": [asdict(item) for item in self.source_regions],
            "behavior_paths": [asdict(item) for item in self.behavior_paths],
            "show_document_alignment": self.show_document_alignment,
            "groundable": self.groundable,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EvidenceUnit":
        return cls(
            unit_id=str(value["unit_id"]),
            unit_kind=value["unit_kind"],
            name=str(value["name"]),
            span=EvidenceSpan(**value["span"]),
            source=str(value["source"]),
            behavior_evidence=tuple(
                BehaviorEvidence(
                    labels=tuple(item["labels"]),
                    start=int(item["start"]),
                    end=int(item["end"]),
                    text=str(item["text"]),
                    owner=item.get("owner"),
                    ranges=tuple(
                        (int(span[0]), int(span[1]))
                        for span in item.get("ranges", [])
                    ),
                )
                for item in value.get("behavior_evidence", [])
            ),
            document_title=value.get("document_title"),
            document_section=value.get("document_section"),
            document_excerpt=value.get("document_excerpt"),
            code_title=value.get("code_title"),
            related=tuple(RelatedUnit(**item) for item in value.get("related", [])),
            documents=tuple(
                DocumentEvidence(**item) for item in value.get("documents", [])
            ),
            members=tuple(PackageMember(**item) for item in value.get("members", [])),
            behavior_total=int(value.get("behavior_total", 0)),
            member_total=int(value.get("member_total", 0)),
            document_total=int(value.get("document_total", 0)),
            source_regions=tuple(
                SourceRegion(
                    symbols=tuple(item.get("symbols", [])),
                    start=int(item["start"]),
                    end=int(item["end"]),
                    source=str(item["source"]),
                    path=item.get("path"),
                )
                for item in value.get("source_regions", [])
            ),
            behavior_paths=tuple(
                BehaviorPathAnchor(
                    owner=str(item["owner"]),
                    steps=tuple(
                        BehaviorPathStep(
                            label=str(step["label"]),
                            ranges=tuple(
                                (int(span[0]), int(span[1]))
                                for span in step.get("ranges", [])
                            ),
                        )
                        for step in item.get("steps", [])
                    ),
                )
                for item in value.get("behavior_paths", [])
            ),
            show_document_alignment=bool(
                value.get(
                    "show_document_alignment",
                    # Read older saved turns without preserving the old verbose view.
                    value.get("show_behavior_focus", False),
                )
            ),
            groundable=bool(value.get("groundable", True)),
        )


@dataclass(frozen=True, slots=True)
class SearchHit:
    unit_id: str
    name: str
    path: str
    symbol: str
    matched_field: str


@dataclass(frozen=True, slots=True)
class ToolResult:
    action: Literal["search", "read"]
    status: Literal["ok", "error"]
    search_text: str | None = None
    total_matches: int = 0
    hits: tuple[SearchHit, ...] = ()
    group_hints: tuple[str, ...] = ()
    requested_ids: tuple[str, ...] = ()
    units: tuple[EvidenceUnit, ...] = ()
    deferred_ids: tuple[str, ...] = ()
    oversize_unit: bool = False
    error_code: str | None = None

    @property
    def returned_ids(self) -> tuple[str, ...]:
        if self.action == "search":
            return tuple(hit.unit_id for hit in self.hits)
        return tuple(unit.unit_id for unit in self.units)

    @property
    def displayed_spans(self) -> tuple[EvidenceSpan, ...]:
        if self.status != "ok":
            return ()
        spans: list[EvidenceSpan] = []
        for unit in self.units:
            if not unit.groundable:
                continue
            if unit.source_regions:
                spans.extend(
                    EvidenceSpan(
                        path=region.path or unit.span.path,
                        symbol=", ".join(region.symbols) or "<source region>",
                        start=region.start,
                        end=region.end,
                    )
                    for region in unit.source_regions
                )
            else:
                spans.append(unit.span)
        return tuple(spans)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "status": self.status,
            "search_text": self.search_text,
            "total_matches": self.total_matches,
            "hits": [asdict(item) for item in self.hits],
            "group_hints": list(self.group_hints),
            "requested_ids": list(self.requested_ids),
            "units": [item.to_dict() for item in self.units],
            "deferred_ids": list(self.deferred_ids),
            "oversize_unit": self.oversize_unit,
            "error_code": self.error_code,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ToolResult":
        return cls(
            action=value["action"],
            status=value["status"],
            search_text=value.get("search_text"),
            total_matches=int(value.get("total_matches", 0)),
            hits=tuple(SearchHit(**item) for item in value.get("hits", [])),
            group_hints=tuple(value.get("group_hints", [])),
            requested_ids=tuple(value.get("requested_ids", [])),
            units=tuple(EvidenceUnit.from_dict(item) for item in value.get("units", [])),
            deferred_ids=tuple(value.get("deferred_ids", [])),
            oversize_unit=bool(value.get("oversize_unit", False)),
            error_code=value.get("error_code"),
        )


def render_tool_result(
    result: ToolResult,
    *,
    full_unit_ids: set[str] | None = None,
    full_search: bool = True,
    selected_search_ids: set[str] | None = None,
    shown_document_keys: set[tuple[str, str, str]] | None = None,
) -> str:
    """Render complete latest evidence and compact semantic history."""

    if result.status == "error":
        return (
            "[[TOOL ERROR]]\n"
            f"Action: {result.action}\n"
            f"Code: {result.error_code or 'invalid_parameters'}\n"
            "The action consumed one round. Return one corrected JSON action.\n"
            "[[END TOOL ERROR]]"
        )
    if result.action == "search":
        return _render_search(
            result,
            full=full_search,
            selected_ids=selected_search_ids or set(),
        )
    allowed = (
        {unit.unit_id for unit in result.units}
        if full_unit_ids is None
        else full_unit_ids
    )
    return _render_read(result, allowed, shown_document_keys)


def _render_search(
    result: ToolResult, *, full: bool, selected_ids: set[str]
) -> str:
    title = "SEARCH RESULT" if full else "SEARCH RECEIPT"
    lines = [
        f"[[{title}]]",
        f"Search: {result.search_text or ''}",
        f"Matches: {len(result.hits)} shown of {result.total_matches}",
    ]
    if result.group_hints:
        lines.append("The search is broad. Refine it with one of these paths or namespaces:")
        lines.extend(f"- {hint}" for hint in result.group_hints)
    elif not result.hits:
        lines.append("No matching repository unit.")
    elif not full:
        selected = [hit for hit in result.hits if hit.unit_id in selected_ids]
        if selected:
            lines.append(
                "IDs later selected for read: "
                + ", ".join(hit.unit_id for hit in selected)
            )
        else:
            lines.append("No ID from this result was later selected for read.")
    else:
        for hit in result.hits:
            lines.extend(
                [
                    f"Read ID: {hit.unit_id}",
                    f"Name: {hit.name}",
                    f"Location: {hit.path} :: {hit.symbol}",
                    f"Matched field: {hit.matched_field}",
                    "",
                ]
            )
    lines.append(f"[[END {title}]]")
    return "\n".join(lines).rstrip()


def _render_read(
    result: ToolResult,
    full_ids: set[str],
    shown_document_keys: set[tuple[str, str, str]] | None = None,
) -> str:
    lines = ["[[READ RESULT]]"]
    shown_documents = (
        shown_document_keys if shown_document_keys is not None else set()
    )
    shown_behaviors: set[BehaviorRenderKey] = set()
    source_ids, covered_by = _source_owners(result.units, full_ids)
    compact_units = 0
    for unit in result.units:
        if unit.unit_id in full_ids:
            lines.extend(
                _render_full_unit(
                    unit,
                    shown_documents,
                    shown_behaviors,
                    include_source=unit.unit_id in source_ids,
                    source_owner=covered_by.get(unit.unit_id),
                )
            )
        elif unit.unit_kind == "file":
            lines.extend(_render_file_evidence(unit, full=False))
            compact_units += 1
        elif unit.unit_kind == "package":
            lines.extend(
                _render_package_evidence(
                    unit,
                    shown_documents,
                    shown_behaviors,
                    full=False,
                )
            )
            compact_units += 1
        elif unit.unit_kind == "symbol":
            lines.extend(
                _render_symbol_evidence(
                    unit,
                    shown_documents,
                    shown_behaviors,
                    full=False,
                    detailed=True,
                )
            )
            compact_units += 1
        else:
            lines.extend(_render_unit_receipt(unit))
            compact_units += 1
    if compact_units:
        lines.extend(
            [
                "",
                "Older evidence keeps selected exact behavior/source lines but omits repeated "
                "full source. The complete task document remains in context, and these source "
                "lines remain valid for finish; do not reread merely to refresh source.",
            ]
        )
    if result.deferred_ids:
        lines.extend(
            [
                "",
                "Deferred IDs: " + ", ".join(result.deferred_ids),
                "Read a deferred ID only if its unseen source is still needed.",
            ]
        )
    if result.oversize_unit:
        lines.extend(
            [
                "",
                "Oversize unit: the first complete unit exceeded the soft tool budget "
                "and was kept intact.",
            ]
        )
    lines.append("[[END READ RESULT]]")
    return "\n".join(lines).rstrip()


def _render_full_unit(
    unit: EvidenceUnit,
    shown_documents: set[tuple[str, str, str]],
    shown_behaviors: set[BehaviorRenderKey],
    *,
    include_source: bool,
    source_owner: str | None,
) -> list[str]:
    if unit.unit_kind == "file":
        return _render_file_evidence(unit, full=include_source)
    if unit.unit_kind == "package":
        lines = _render_package_evidence(
            unit,
            shown_documents,
            shown_behaviors,
            full=include_source,
        )
        if source_owner is not None:
            lines.append(
                f"Continuous source is included under Read ID {source_owner} in this result."
            )
        if include_source and unit.related:
            lines.extend(["", "Related source packages (navigation only):"])
            for relation in unit.related:
                suffix = f" via {relation.via}" if relation.via else ""
                lines.append(
                    f"- {relation.label}: {relation.unit_id} | {relation.name}{suffix}"
                )
        return lines
    if unit.unit_kind == "symbol":
        lines = _render_symbol_evidence(
            unit,
            shown_documents,
            shown_behaviors,
            full=include_source,
            detailed=not include_source and source_owner is None,
        )
        if source_owner is not None:
            lines.append(
                f"Full source is included under Read ID {source_owner} in this result."
            )
        if unit.related:
            lines.extend(["", "Related symbols:"])
            for relation in unit.related:
                suffix = f" via {relation.via}" if relation.via else ""
                lines.append(
                    f"- {relation.label}: {relation.unit_id} | {relation.name}{suffix}"
                )
        return lines
    if unit.unit_kind == "local_graph":
        if not include_source:
            return _render_unit_receipt(unit)
        return [
            "",
            "[[LINEAR LOCAL GRAPH]]",
            f"Read ID: {unit.unit_id}",
            unit.source.rstrip("\n"),
            "[[END LINEAR LOCAL GRAPH]]",
        ]
    if not include_source:
        return _render_unit_receipt(unit)
    return [
        "",
        "[[SOURCE UNIT]]",
        f"Unit ID: {unit.unit_id}",
        f"File: {unit.span.path}",
        f"Lines: {unit.span.start}-{unit.span.end}",
        unit.source.rstrip("\n"),
        "[[END SOURCE UNIT]]",
    ]


def _render_file_evidence(unit: EvidenceUnit, *, full: bool) -> list[str]:
    """Render graph-selected source as ordinary, file-ordered code regions."""

    if unit.show_document_alignment:
        return _render_silentswap_file_evidence(unit, full=full)

    sections = list(dict.fromkeys(item.section for item in unit.documents))
    lines = [
        "",
        "[[FILE EVIDENCE]]" if full else "[[FILE RECEIPT]]",
        f"Read ID: {unit.unit_id}",
        f"File: {unit.span.path}",
        "Document sections: " + ("; ".join(sections) if sections else "none"),
    ]
    if not full:
        lines.append("Retained behavior evidence:")
        for evidence in unit.behavior_evidence:
            lines.append(_behavior_focus_line(evidence))
            lines.extend(
                f"    {text}"
                for text in evidence.text.rstrip("\n").splitlines()
            )
        lines.extend(
            [
                "Continuous source was removed from older context; these exact source "
                "lines remain valid for finish.",
                "[[END FILE RECEIPT]]",
            ]
        )
        return lines

    regions = unit.source_regions or (
        SourceRegion(
            symbols=(unit.span.symbol,),
            start=unit.span.start,
            end=unit.span.end,
            source=unit.source,
        ),
    )
    for region in regions:
        lines.extend(
            [
                "",
                "[[SOURCE REGION]]",
                "Symbols: " + "; ".join(region.symbols),
                f"Lines: {region.start}-{region.end}",
            ]
        )
        lines.append("Source:")
        lines.extend(
            [
                region.source.rstrip("\n"),
                "[[END SOURCE REGION]]",
            ]
        )
    if unit.related:
        lines.extend(["", "Related files:"])
        for relation in unit.related:
            suffix = f" via {relation.via}" if relation.via else ""
            lines.append(
                f"- {relation.label}: {relation.unit_id} | {relation.name}{suffix}"
            )
    lines.append("[[END FILE EVIDENCE]]")
    return lines


def _render_silentswap_file_evidence(
    unit: EvidenceUnit, *, full: bool
) -> list[str]:
    """Keep the original contract adjacent to compact, continuous current source."""

    title = "FILE EVIDENCE" if full else "FILE RECEIPT"
    lines = [
        "",
        f"[[{title}]]",
        f"Read ID: {unit.unit_id}",
        f"File: {unit.span.path}",
    ]
    documents = tuple(
        document for document in unit.documents if document.excerpt.strip()
    )
    if documents:
        lines.extend(["", "[[BEFORE: ORIGINAL DOCUMENT]]"])
        for index, document in enumerate(documents):
            if index:
                lines.append("")
            lines.extend(
                [
                    f"Section: {document.section}",
                    document.excerpt.strip(),
                ]
            )
        lines.append("[[END BEFORE]]")

    if full:
        regions = unit.source_regions or (
            SourceRegion(
                symbols=(unit.span.symbol,),
                start=unit.span.start,
                end=unit.span.end,
                source=unit.source,
            ),
        )
        lines.extend(["", "[[AFTER: CURRENT SOURCE]]"])
        for index, region in enumerate(regions):
            if index:
                lines.append("")
            focus = "; ".join(region.symbols)
            lines.append(
                f"Focus: {focus} | Lines: {region.start}-{region.end}"
            )
            operation_lines = _operation_lines_in_region(
                unit.behavior_evidence, region
            )
            if operation_lines:
                lines.append(
                    "Read first: lines "
                    + ", ".join(str(value) for value in operation_lines)
                    + " (navigation only; inspect the continuous source)."
                )
            lines.append(region.source.rstrip("\n"))
        lines.append("[[END AFTER]]")
    else:
        retained = _retained_source_lines(unit.behavior_evidence)
        if retained:
            lines.extend(
                [
                    "",
                    "[[RETAINED CURRENT SOURCE LINES]]",
                    *retained,
                    "[[END RETAINED CURRENT SOURCE LINES]]",
                ]
            )
        lines.append(
            "Continuous source was removed from older context; the exact lines above "
            "remain valid for finish."
        )

    if unit.related:
        lines.extend(["", "[[RELATED FILES]]"])
        for relation in unit.related:
            suffix = f" via {relation.via}" if relation.via else ""
            lines.append(
                f"- {relation.label}: {relation.unit_id} | {relation.name}{suffix}"
            )
        lines.append("[[END RELATED FILES]]")
    lines.append(f"[[END {title}]]")
    return lines


def _operation_lines_in_region(
    evidence_items: tuple[BehaviorEvidence, ...], region: SourceRegion
) -> tuple[int, ...]:
    values: set[int] = set()
    for evidence in evidence_items:
        for label in evidence.labels:
            match = re.search(r"@(\d+)", label)
            if match is None:
                continue
            value = int(match.group(1))
            if region.start <= value <= region.end:
                values.add(value)
    return tuple(sorted(values))


def _retained_source_lines(
    evidence_items: tuple[BehaviorEvidence, ...],
) -> list[str]:
    by_number: dict[int, str] = {}
    for evidence in evidence_items:
        for text in evidence.text.rstrip("\n").splitlines():
            match = re.match(r"(\d+) \|", text)
            if match is not None:
                by_number.setdefault(int(match.group(1)), text)
    return [by_number[number] for number in sorted(by_number)]


def _behavior_focus_line(evidence: BehaviorEvidence) -> str:
    label = evidence.labels[0] if evidence.labels else "result"
    kind, _, position = label.partition("@")
    descriptions = {
        "configure": "Configures",
        "execute": "Executes",
        "external_call": "Makes an external call",
        "output": "Produces output",
        "raise": "Raises an exception",
        "render": "Renders output",
        "return": "Returns",
        "state_write": "Updates state",
        "yield": "Yields",
    }
    description = descriptions.get(kind, kind.replace("_", " ").capitalize())
    result = f"{description} at line {position}" if position else description
    ranges = evidence.ranges or ((evidence.start, evidence.end),)
    support = ", ".join(
        str(start) if start == end else f"{start}-{end}"
        for start, end in ranges
    )
    owner = evidence.owner or "module code"
    return f"- {owner}: {result} (supporting lines: {support})."


def _render_package_evidence(
    unit: EvidenceUnit,
    shown_documents: set[tuple[str, str, str]],
    shown_behaviors: set[BehaviorRenderKey],
    *,
    full: bool,
) -> list[str]:
    lines = _render_documents(unit, shown_documents)
    if unit.document_total > len(unit.documents):
        lines.extend(
            [
                "",
                (
                    f"Additional linked document sections: "
                    f"{unit.document_total - len(unit.documents)}; consult the complete "
                    "TASK DOCUMENT already in context."
                ),
            ]
        )
    code_title = unit.code_title or "CURRENT CODE"
    lines.extend(
        [
            "",
            "[[SOURCE PACKAGE]]",
            f"Read ID: {unit.unit_id}",
            f"File: {unit.span.path}",
            f"Lines: {unit.span.start}-{unit.span.end}",
            "Role: neutral continuous source selected and grouped by the behavior graph",
            (
                "Symbols shown in this package:"
                if not unit.member_total or unit.member_total == len(unit.members)
                else f"Symbols shown: {len(unit.members)} of {unit.member_total}"
            ),
        ]
    )
    for member in unit.members:
        role = "focus" if member.focus else "context"
        lines.append(
            f"- {member.symbol} | lines {member.start}-{member.end} | {role}"
        )

    unseen = [
        evidence
        for evidence in unit.behavior_evidence
        if _behavior_key(unit, evidence) not in shown_behaviors
    ]
    if unseen:
        shown_behaviors.update(_behavior_key(unit, evidence) for evidence in unseen)
        lines.extend(
            [
                "",
                (
                    "Behavior evidence from the graph (original source lines, not conclusions):"
                    if not unit.behavior_total or unit.behavior_total == len(unseen)
                    else (
                        f"Behavior evidence anchors shown: {len(unseen)} of "
                        f"{unit.behavior_total}; use the continuous source as authoritative."
                    )
                ),
            ]
        )
        current_owner: str | None = None
        for evidence in unseen:
            if evidence.owner != current_owner:
                lines.append(f"Owner symbol: {evidence.owner or unit.span.symbol}")
                current_owner = evidence.owner
            lines.append(f"- Behavior anchor: {', '.join(evidence.labels)}")
            path = evidence.ranges or ((evidence.start, evidence.end),)
            lines.append(
                "  Evidence path: "
                + " -> ".join(
                    str(start) if start == end else f"{start}-{end}"
                    for start, end in path
                )
            )
            lines.append("  Key original lines:")
            lines.extend(
                f"    {line}" for line in evidence.text.rstrip("\n").splitlines()
            )
    else:
        lines.append("Behavior evidence was already shown elsewhere in this result.")

    if full:
        lines.extend(
            [
                "",
                f"[[{code_title}]]",
                "Continuous original source:",
                unit.source.rstrip("\n"),
                f"[[END {code_title}]]",
            ]
        )
    else:
        lines.append(
            "Continuous source was removed from older context; the exact behavior lines remain valid."
        )
    lines.append("[[END SOURCE PACKAGE]]")
    return lines


def _render_documents(
    unit: EvidenceUnit,
    shown_documents: set[tuple[str, str, str]],
) -> list[str]:
    documents = list(unit.documents)
    if (
        unit.document_title
        and unit.document_section
        and unit.document_excerpt
    ):
        documents.append(
            DocumentEvidence(
                title=unit.document_title,
                section=unit.document_section,
                excerpt=unit.document_excerpt,
            )
        )
    lines: list[str] = []
    for document in documents:
        key = (document.title, document.section, document.excerpt)
        if key in shown_documents:
            continue
        shown_documents.add(key)
        lines.extend(
            [
                "",
                f"[[{document.title}]]",
                f"Role: {_document_role(document.title)}",
                f"Section: {document.section}",
                document.excerpt.rstrip("\n"),
                f"[[END {document.title}]]",
            ]
        )
    return lines


def _source_owners(
    units: tuple[EvidenceUnit, ...], full_ids: set[str]
) -> tuple[set[str], dict[str, str]]:
    full = [unit for unit in units if unit.unit_id in full_ids]
    source_ids = {unit.unit_id for unit in full}
    for unit in full:
        containers = [
            other
            for other in full
            if other.unit_kind == unit.unit_kind
            and other.span.path == unit.span.path
            and other.span.start <= unit.span.start
            and unit.span.end <= other.span.end
            and (
                other.span.start < unit.span.start
                or unit.span.end < other.span.end
                or other.unit_id < unit.unit_id
            )
        ]
        if not containers:
            continue
        source_ids.discard(unit.unit_id)
    covered_by: dict[str, str] = {}
    source_units = [unit for unit in full if unit.unit_id in source_ids]
    for unit in full:
        if unit.unit_id in source_ids:
            continue
        containers = [
            other
            for other in source_units
            if other.unit_kind == unit.unit_kind
            and other.span.path == unit.span.path
            and other.span.start <= unit.span.start
            and unit.span.end <= other.span.end
        ]
        if not containers:
            source_ids.add(unit.unit_id)
            continue
        owner = min(
            containers,
            key=lambda other: (
                other.span.end - other.span.start,
                other.span.start,
                other.unit_id,
            ),
        )
        covered_by[unit.unit_id] = owner.unit_id
    return source_ids, covered_by


def _render_symbol_evidence(
    unit: EvidenceUnit,
    shown_documents: set[tuple[str, str, str]],
    shown_behaviors: set[BehaviorRenderKey],
    *,
    full: bool,
    detailed: bool,
) -> list[str]:
    lines: list[str] = []
    if unit.document_title and unit.document_section and unit.document_excerpt:
        document_key = (
            unit.document_title,
            unit.document_section,
            unit.document_excerpt,
        )
        if document_key not in shown_documents:
            shown_documents.add(document_key)
            lines.extend(
                [
                    "",
                    f"[[{unit.document_title}]]",
                    f"Role: {_document_role(unit.document_title)}",
                    f"Section: {unit.document_section}",
                    unit.document_excerpt.rstrip("\n"),
                    f"[[END {unit.document_title}]]",
                ]
            )
    code_title = unit.code_title or "CURRENT CODE"
    lines.extend(
        [
            "",
            f"[[{code_title}]]",
            f"Read ID: {unit.unit_id}",
            f"File: {unit.span.path}",
            f"Symbol: {unit.span.symbol}",
            f"Lines: {unit.span.start}-{unit.span.end}",
            f"Role: {_code_role(code_title)}",
        ]
    )
    if full:
        lines.extend(["", "Full symbol source:", unit.source.rstrip("\n")])

    unseen = [
        evidence
        for evidence in unit.behavior_evidence
        if _behavior_key(unit, evidence) not in shown_behaviors
    ]
    if not unseen:
        lines.append("Behavior graph coverage was already shown elsewhere in this result.")
        lines.append(f"[[END {code_title}]]")
        return lines

    if not detailed:
        owners = {evidence.owner for evidence in unseen if evidence.owner is not None}
        owner_text = (
            f" across {len(owners)} source symbol{'s' if len(owners) != 1 else ''}"
            if owners
            else ""
        )
        lines.append(
            f"Behavior graph coverage: {len(unseen)} extracted control-flow "
            f"path{'s' if len(unseen) != 1 else ''}{owner_text}."
        )
        lines.append(
            "Coverage counts are navigation metadata, not a semantic summary. "
            "Interpret conditions, defaults, assignments, and results from the full source."
        )
        lines.append("Behavior anchors (navigation only):")
        anchored = _evenly_spaced_anchors(unseen, _MAX_FULL_SOURCE_ANCHORS)
        shown_behaviors.update(_behavior_key(unit, evidence) for evidence in anchored)
        for evidence in anchored:
            owner = f"{evidence.owner}: " if evidence.owner else ""
            ranges = evidence.ranges or ((evidence.start, evidence.end),)
            key_ranges = ranges[-2:]
            lines.append(
                f"- {owner}{', '.join(evidence.labels)} | key lines "
                + ", ".join(
                    str(start) if start == end else f"{start}-{end}"
                    for start, end in key_ranges
                )
            )
        if len(anchored) < len(unseen):
            lines.append(
                f"- {len(unseen) - len(anchored)} additional paths remain in the full source."
            )
        lines.append(f"[[END {code_title}]]")
        return lines

    shown_behaviors.update(_behavior_key(unit, evidence) for evidence in unseen)
    lines.append("Exact behavior evidence retained after full-source compression:")
    current_owner: str | None = None
    for evidence in unseen:
        if evidence.owner is not None:
            if evidence.owner != current_owner:
                lines.append(f"Owner symbol: {evidence.owner}")
                current_owner = evidence.owner
            lines.append(f"- Behavior: {', '.join(evidence.labels)}")
            path = evidence.ranges or ((evidence.start, evidence.end),)
            lines.append(
                "  Evidence path: "
                + " -> ".join(
                    str(start) if start == end else f"{start}-{end}"
                    for start, end in path
                )
            )
            lines.append("  Key source lines:")
            lines.extend(
                f"    {line}" for line in evidence.text.rstrip("\n").splitlines()
            )
        else:
            lines.append(
                f"- {', '.join(evidence.labels)} | Evidence lines: "
                f"{evidence.start}-{evidence.end}"
            )
            lines.extend(
                f"  {line}" for line in evidence.text.rstrip("\n").splitlines()
            )
    lines.append(f"[[END {code_title}]]")
    return lines


def _behavior_key(unit: EvidenceUnit, evidence: BehaviorEvidence) -> BehaviorRenderKey:
    return (
        unit.span.path,
        evidence.owner,
        evidence.labels,
        evidence.ranges or ((evidence.start, evidence.end),),
        evidence.text,
    )


def _evenly_spaced_anchors(
    evidence: list[BehaviorEvidence], limit: int
) -> list[BehaviorEvidence]:
    ordered = sorted(
        evidence,
        key=lambda item: (
            item.end,
            item.start,
            item.owner or "",
            item.labels,
        ),
    )
    if len(ordered) <= limit:
        return ordered
    indexes = [round(index * (len(ordered) - 1) / (limit - 1)) for index in range(limit)]
    return [ordered[index] for index in indexes]


def _document_role(title: str) -> str:
    if title == "CURRENT DOCUMENT EXCERPT":
        return "current post-deletion document; it may omit a few implemented behaviors"
    if title == "BEFORE: ORIGINAL DOCUMENT":
        return "original behavior contract before the silent swap"
    return "task document excerpt related to this symbol"


def _code_role(title: str) -> str:
    if title == "AFTER: CURRENT CODE":
        return "current implementation after the possible silent swap"
    return "current implementation to compare with the task document"


def _render_unit_receipt(unit: EvidenceUnit) -> list[str]:
    return [
        "",
        "[[READ RECEIPT]]",
        f"Unit ID: {unit.unit_id}",
        f"Unit name: {unit.name}",
        f"Location: {unit.span.path} :: {unit.span.symbol} :: {unit.span.start}-{unit.span.end}",
        "Content removed from working context; read this ID again if needed.",
        "[[END READ RECEIPT]]",
    ]
