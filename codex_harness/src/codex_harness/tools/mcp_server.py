"""Three MCP tools for source selection and evidence-backed disclosure."""

from __future__ import annotations

from typing import Literal

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from ..application.service import Harness


class SourceReference(BaseModel):
    source_id: str
    quote: str = Field(description="Verbatim contiguous text from a selected Prompt or Plan.")
    start: int | None = Field(default=None, description="Character offset, needed for repeated quotes.")
    origin: Literal['user_plan', 'agent_plan', 'unknown'] | None = Field(
        default=None, description="For Plan references only: identify provenance when supported by context.")


class Requirement(BaseModel):
    id: str = Field(pattern=r'^R[1-9][0-9]*$')
    check: str
    refs: list[SourceReference]


def create_server(harness: Harness) -> FastMCP:
    server = FastMCP(
        'beg-disclose', log_level='WARNING',
        instructions='List full Prompts and Markdown Plan candidates with beg_list_task_sources. '
        'Select a continuous Prompt interval and/or Plan versions with beg_select_task. Plan-only selection '
        'checks frozen current code without historical Trace or a baseline. Supply repo_path to list current '
        'Markdown Plans when recorded sources are unavailable. Prompt and Plan cannot both be empty. Normalize originals, '
        'then submit requirements to beg_build_evidence_groups. Follow returned read_ref/next within the same '
        'tool when material is too large. Evidence is not a verdict. Never modify or test the target repo during disclosure.',
    )
    local = ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=False)

    @server.tool(annotations=local, structured_output=False)
    def beg_list_task_sources(session_id: str | None = None, read_ref: str | None = None, offset: int = 0,
                              repo_path: str | None = None) -> str:
        """List chronological full user Prompts and saved Markdown Plan versions in the current session.
        Select relevant Plan candidates; their origin is not inferred from filenames. Session defaults to the
        active recorded session. Large returns have read_ref/next; pass those to this same tool for full originals.
        Supply repo_path to also collect current Markdown Plan candidates, including when no hooks were recorded.
        This does not create historical Prompts or execution records. Omit repo_path for continuation reads.
        """
        return harness.list_task_sources(session_id, read_ref=read_ref, offset=offset, repo_path=repo_path)

    @server.tool(annotations=local, structured_output=False)
    def beg_select_task(start_prompt: str | None = None, end_prompt: str | None = None,
                        plan_ids: list[str] | None = None, session_id: str | None = None,
                        read_ref: str | None = None, offset: int = 0, repo_path: str | None = None) -> str:
        """Select inclusive start_prompt/end_prompt (e.g. P98/P100) and listed plan_ids.
        Exclude the disclosure request; end must be a completed turn. Harness automatically fixes Trace and
        before/after Repo snapshots for historical selection. Alternatively omit BOTH Prompt endpoints and select
        plan_ids to inspect current code; repo_path is optional if already known. At least a Prompt range or a
        nonempty Plan is required. Plan-only mode freezes current Repo once, with no historical Trace or diff.
        Returns task_id and Prompt/Plan originals to normalize. A supplied invalid historical range is an error.
        To read a large saved selection, supply only its read_ref/offset (and optional session_id).
        """
        return harness.select_task(start_prompt, end_prompt, plan_ids, session_id=session_id,
                                   read_ref=read_ref, offset=offset, repo_path=repo_path)

    @server.tool(annotations=local, structured_output=False)
    def beg_build_evidence_groups(task_id: str, requirements: list[Requirement] | None = None,
                                  read_ref: str | None = None, offset: int = 0) -> str:
        """Submit the final effective requirements with exact original references to build BEG evidence groups.
        Preserve amendments/cancellations and distinguish user demands from agent plans. Does not issue verdicts.
        Unmatched requirements include labeled section/file navigation, not implementation evidence. Follow
        repository or candidate read_ref to file outlines, then symbol read_ref for exact frozen source excerpts.
        Oversized evidence directory entries may provide source and content_ref for direct text reads.
        On overflow, returns a directory, fitting groups and read references. Continue with task_id plus read_ref
        and offset, omitting requirements. Follow next for remaining directory/text pages; offset counts entries
        or Unicode characters as indicated. Reads remain fixed even if a revised checklist is later submitted.
        """
        values = None if requirements is None else [r.model_dump(exclude_none=True) for r in requirements]
        return harness.build_evidence_groups(task_id, values, read_ref=read_ref, offset=offset)

    return server
