"""Three public tools for checkpoint review, evidence lookup and judgment recording."""

from __future__ import annotations

from typing import Literal

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from ..application.service import Harness


class SourceReference(BaseModel):
    source_id: str
    quote: str = Field(description="Verbatim contiguous source text. ebg_evidence also accepts Trace or returned frozen repo/context/diff references.")
    start: int | None = Field(default=None, description="Character offset, needed for repeated quotes.")
    origin: Literal['user_plan', 'agent_plan', 'unknown'] | None = Field(
        default=None, description="For Plan references only: identify provenance when supported by context.")


def create_server(harness: Harness) -> FastMCP:
    server = FastMCP(
        'ebg-disclose', log_level='WARNING',
        instructions='Use ebg-review for autoresearch. Review ambiguity before executing a Plan; '
        'At Stop, result is triggered by Plan execution, call/time thresholds or recorded result limitations; '
        'adjustment by tool failures, material ambiguities or adjustments. Review only triggered stages, '
        'adjustment before result when both apply. Read a hook check_id or create with trigger/focus. '
        'Review provides requirements, execution records and check criteria, without automatic code retrieval. '
        'Every assessment requires ebg_evidence with a concrete question and relevant code/execution evidence. '
        'Use ebg_evidence for all active-review read_ref/next expansions, including review pages. '
        'Record the judgment using ebg_record; reads alone are not completed checks. '
        'Evidence is not a verdict. Return to ordinary authorized work for new verification or repairs.',
    )
    local = ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=False)

    @server.tool(annotations=local, structured_output=False)
    def ebg_review(check_id: str | None = None, trigger: Literal['result', 'adjustment', 'ambiguity'] | None = None,
                   focus: str | None = None, event_ids: list[str] | None = None,
                   plan_ids: list[str] | None = None) -> str:
        """Read check_id from a hook, or freeze now with trigger/focus and optional recorded event/Plan IDs.
        Returns original Prompts, related call/results, Plan candidates, focus and review criteria; no automatic code retrieval.
        Always call ebg_evidence with a specific question before judgment, including apparently clear cases.
        Expand all_trace/all_sources or any read_ref/next using ebg_evidence. Older Prompts may concern other tasks.
        Use ebg_record after judgment. Reading a checkpoint neither records an assessment nor discloses it to the user.
        """
        return harness.checks.review(check_id, trigger=trigger, focus=focus, event_ids=event_ids, plan_ids=plan_ids)

    @server.tool(annotations=local, structured_output=False)
    def ebg_record(check_id: str | None = None, conclusion: Literal['clear', 'issue', 'uncertain'] | None = None,
                   summary: str = '', waiting_for_user: bool = False, resolution: str | None = None,
                   note_kind: Literal['adjustment', 'ambiguity', 'limitation'] | None = None,
                   decision_status: Literal['proposed', 'executed'] = 'proposed') -> str:
        """Save the judgment, evidence basis and handling; return a short acknowledgement only.
        Call ebg_evidence first for every assessment; unresolved essential premises cannot be clear.
        Set waiting_for_user for a necessary user choice, then immediately ask and end the turn.
        After a real reply resolves the choice, use the waiting check_id and resolution explaining how.
        During execution, save an unreviewed process note with note_kind, summary and decision_status only;
        omit check_id/conclusion. This defers evidence review to Stop and does not authorize an action.
        Records agent judgment, not an independent verdict or proof of user-facing disclosure.
        """
        return harness.checks.record(check_id, conclusion, summary,
                                     waiting_for_user=waiting_for_user, resolution=resolution,
                                     note_kind=note_kind, decision_status=decision_status)

    @server.tool(annotations=local, structured_output=False)
    def ebg_evidence(check_id: str, question: str | None = None, refs: list[SourceReference] | None = None,
                     read_ref: str | None = None, offset: int = 0) -> str:
        """Investigate a concern or expand any active-review material, including pages returned by ebg_review.
        Refs are optional; omitted refs use original requirements and relevant execution calls at the checkpoint.
        Optional refs quote Prompt/Plan/Trace IDs or returned V*:repo/context/diff references in source_id.
        Use verbatim quote and optional start. A question or code quote is not a user requirement.
        Code comes from the checkpoint; history refs allow version comparison. Matching is not a verdict.
        For returned code/diff/history references or next, call this tool with check_id/read_ref/offset only.
        """
        values = None if refs is None else [r.model_dump(exclude_none=True) for r in refs]
        return harness.checks.evidence(check_id, question, values, read_ref=read_ref, offset=offset)

    return server
