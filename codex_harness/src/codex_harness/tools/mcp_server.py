"""Three public tools for checkpoint review, evidence lookup and judgment recording."""

from __future__ import annotations

from typing import Literal

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from ..application.service import Harness


class SourceReference(BaseModel):
    source_id: str
    quote: str = Field(description="Verbatim contiguous source text. beg_evidence also accepts Trace or returned frozen repo/context/diff references.")
    start: int | None = Field(default=None, description="Character offset, needed for repeated quotes.")
    origin: Literal['user_plan', 'agent_plan', 'unknown'] | None = Field(
        default=None, description="For Plan references only: identify provenance when supported by context.")


def create_server(harness: Harness) -> FastMCP:
    server = FastMCP(
        'beg-disclose', log_level='WARNING',
        instructions='During execution use beg_review at result adoption/reporting, material plan adjustments '
        'or important ambiguities. Read a hook check_id or create a checkpoint with trigger/focus. '
        'Review provides requirements, execution records and check criteria, without automatic code retrieval. '
        'For every concern, even an uncertain one, use beg_evidence with a concrete question before judgment. '
        'Use beg_evidence for all active-review read_ref/next expansions, including review pages. '
        'Record the judgment using beg_record; reads alone are not completed checks. '
        'Evidence is not a verdict. Return to ordinary authorized work for new verification or repairs.',
    )
    local = ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=False)

    @server.tool(annotations=local, structured_output=False)
    def beg_review(check_id: str | None = None, trigger: Literal['result', 'adjustment', 'ambiguity'] | None = None,
                   focus: str | None = None, event_ids: list[str] | None = None,
                   plan_ids: list[str] | None = None) -> str:
        """Read check_id from a hook, or freeze now with trigger/focus and optional recorded event/Plan IDs.
        Returns original Prompts, related call/results, Plan candidates, focus and review criteria; no automatic code retrieval.
        For any concern, even if uncertain, call beg_evidence with a specific question before judging it.
        Expand all_trace/all_sources or any read_ref/next using beg_evidence. Older Prompts may concern other tasks.
        Use beg_record after judgment. Reading a checkpoint neither records an assessment nor discloses it to the user.
        """
        return harness.checks.review(check_id, trigger=trigger, focus=focus, event_ids=event_ids, plan_ids=plan_ids)

    @server.tool(annotations=local, structured_output=False)
    def beg_record(check_id: str, conclusion: Literal['clear', 'issue', 'uncertain'], summary: str) -> str:
        """Save the judgment, evidence basis and handling; return a short acknowledgement only.
        Investigate every concern through beg_evidence first; unresolved essential premises cannot be clear.
        Records agent judgment, not an independent verdict or proof of user-facing disclosure.
        """
        return harness.checks.record(check_id, conclusion, summary)

    @server.tool(annotations=local, structured_output=False)
    def beg_evidence(check_id: str, question: str | None = None, refs: list[SourceReference] | None = None,
                     read_ref: str | None = None, offset: int = 0) -> str:
        """Investigate a concern or expand any active-review material, including pages returned by beg_review.
        Refs are optional; omitted refs use original requirements and relevant execution calls at the checkpoint.
        Optional refs quote Prompt/Plan/Trace IDs or returned V*:repo/context/diff references in source_id.
        Use verbatim quote and optional start. A question or code quote is not a user requirement.
        Code comes from the checkpoint; history refs allow version comparison. Matching is not a verdict.
        For returned code/diff/history references or next, call this tool with check_id/read_ref/offset only.
        """
        values = None if refs is None else [r.model_dump(exclude_none=True) for r in refs]
        return harness.checks.evidence(check_id, question, values, read_ref=read_ref, offset=offset)

    return server
