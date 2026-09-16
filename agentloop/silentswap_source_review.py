"""Independent source recovery after an ordinary BEG localization run."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import re
from pathlib import Path
from typing import Any

from agentloop.config import DEFAULT_CONFIG
from agentloop.context import FastTokenCounter
from agentloop.errors import AgentLoopError, ContextUnfitError, RetryableModelError
from agentloop.provider import validated_model_profile
from agentloop.storage import RunStore
from evaluation_core.contracts import (
    EvidenceSpan, PredictionError, RepoPredictionContract,
    load_prediction_schema, model_prediction_schema,
)

ROOT = Path(__file__).resolve().parents[1]
PROMPT_PATH = ROOT / "prompts/BEG/silentswap2.txt"
CONFIG = replace(DEFAULT_CONFIG, format_repair_attempts=1)
WORKFLOW = "beg_independent_source_review_v5"
SELECTION_POLICY = "directory_top6_union_read_files"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def freeze_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        if read_json(path) != value:
            raise AgentLoopError(f"saved experiment input changed: {path}")
    else:
        RunStore(path.parent)._write_json(path, value)


def load_stage1(run: Path) -> tuple[dict, dict, Path]:
    """Use the reading history that produced the final saved first-stage answer."""
    prediction = read_json(run / "prediction.json")
    batch = read_json(run / "batch_result.json") if (run / "batch_result.json").exists() else {}
    source = None
    if batch.get("repair_source_attempt") is not None:
        source = run / f"attempt_{batch['repair_source_attempt']}" / "state.json"
    elif batch.get("local_format_repair"):
        repairs = list(run.glob("*/repair.json"))
        if len(repairs) == 1:
            source = (repairs[0].parent / read_json(repairs[0])["source"]).resolve()
        elif batch.get("manual_retry_attempt") is not None:
            source = run / f"attempt_{batch['manual_retry_attempt']}" / "state.json"
    if source is None:
        matches = []
        for path in [run / "state.json", *run.glob("*/state.json")]:
            saved = path.parent / "prediction.json"
            if path.exists() and saved.exists() and read_json(saved) == prediction:
                if read_json(path).get("status") == "complete":
                    matches.append(path)
        if len(matches) != 1:
            raise AgentLoopError(f"cannot select a unique completed BEG history: {run}")
        source = matches[0]
    state = read_json(source)
    if state.get("input_id") != prediction.get("input_id"):
        raise AgentLoopError("first-stage history and prediction have different input IDs")
    return prediction, state, source


@dataclass
class ReviewInput:
    input_id: str
    messages: list[dict[str, str]]
    spans: list[EvidenceSpan]
    sources: dict[str, str]
    schema: dict[str, Any]
    selection: dict[str, Any]


def build_review(bundle, prediction: dict, directory: dict, *, state: dict, prompt: str | None = None,
                 schema_root: Path = ROOT / "schemas") -> ReviewInput:
    if bundle.benchmark != "silentswap" or bundle.task_document is None:
        raise AgentLoopError("source review requires a SilentSwap document and repository")
    if prediction.get("input_id") != bundle.input_id or directory.get("input_id") != bundle.input_id:
        raise AgentLoopError("source review inputs belong to different samples")
    sources = {item.path: item.content for item in bundle.repo_artifacts}
    if state.get("input_id") != bundle.input_id:
        raise AgentLoopError("first-stage reading history belongs to a different sample")
    read_files = {
        unit["span"]["path"]
        for record in state["records"]
        if (record.get("tool_result") or {}).get("action") == "read"
        for unit in record["tool_result"].get("units", [])
        if unit.get("span") and unit["span"].get("path")
    }
    top_files = [entry["path"] for entry in directory["entries"][:6]]
    full_files = read_files | set(top_files)

    blocks, spans, selected = [], [], []
    # Selection provenance stays outside model messages. Always render canonical
    # full files in path order, without answer text, rankings, or graph annotations.
    for path in sorted(full_files):
        if path not in sources:
            raise AgentLoopError(f"selected file is absent from visible source: {path}")
        lines = sources[path].splitlines()
        rendered = "\n".join(f"{n} | {line}" for n, line in enumerate(lines, 1))
        blocks.append(f"FILE {path}\n{rendered}\nEND FILE")
        if lines:
            spans.append(EvidenceSpan(path, "<file>", 1, len(lines)))
        selected.append({"path": path, "ranges": [{"start": 1, "end": len(lines)}] if lines else []})
    schema = load_prediction_schema(schema_root, "silentswap")
    schema["properties"]["swaps"].update(minItems=5, maxItems=5)
    body = (
        "TASK DOCUMENT\n" + bundle.task_document.content + "\nEND TASK DOCUMENT\n\n"
        + "CURRENT SOURCE\n" + "\n\n".join(blocks) + "\nEND CURRENT SOURCE\n\n"
        + "PREDICTION SCHEMA\n" + json.dumps(model_prediction_schema(schema), ensure_ascii=False)
    )
    return ReviewInput(
        bundle.input_id,
        [{"role": "system", "content": prompt if prompt is not None else PROMPT_PATH.read_text(encoding="utf-8")},
         {"role": "user", "content": body}],
        spans, sources, schema,
        {"selection_policy": SELECTION_POLICY, "directory_top_files": top_files,
         "read_files": sorted(read_files), "full_files": sorted(full_files),
         "source_ranges": selected},
    )


def save_request(store: RunStore, messages: list[dict[str, str]], *, retry: int = 0,
                 format_retry: int = 0) -> int:
    count = FastTokenCounter().count_messages(messages)
    if count > CONFIG.physical_hard_limit:
        raise ContextUnfitError("source review exceeds context limit; source was not truncated")
    store.save_request_if_unchanged(
        1, messages=messages, token_count=count, compression={"mode": WORKFLOW},
        provider_retry=retry, format_retry=format_retry,
    )
    return count


def _complete(messages, client, store, format_retry):
    save_request(store, messages, format_retry=format_retry)
    saved = store.load_response(1, format_retry=format_retry)
    if saved is not None:
        return saved["content"]
    output_limit = CONFIG.max_output_tokens
    for previous_format in range(format_retry):
        previous_response = store.load_response(1, format_retry=previous_format) or {}
        if any(choice.get("finish_reason") == "length"
               for choice in previous_response.get("raw_response", {}).get("choices", [])):
            output_limit = CONFIG.max_output_tokens * 2
            break
    stem = store._turn_stem(1, format_retry)
    previous = (store.root / "provider_failures").glob(f"{stem}_retry_*.json")
    first_retry = max((read_json(path)["provider_retry"] for path in previous), default=-1) + 1
    # A resumed invocation gets a fresh bounded retry budget; old failures remain.
    for retry in range(first_retry, first_retry + CONFIG.network_retries + 1):
        save_request(store, messages, retry=retry, format_retry=format_retry)
        try:
            response = client.complete(messages, max_output_tokens=output_limit)
        except RetryableModelError as error:
            store.save_provider_failure(1, provider_retry=retry, format_retry=format_retry,
                error=str(error), raw_response=error.raw_response, usage=error.usage)
            continue
        store.save_response(1, content=response.content, raw_response=response.raw_response,
            usage=response.usage, provider_retry=retry, format_retry=format_retry)
        return response.content
    raise AgentLoopError("source-review network retries exhausted")


def parse_review_prediction(content: str) -> dict[str, Any]:
    """Accept plain JSON or one unambiguous JSON fence with surrounding prose."""
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        blocks = re.findall(r"^```([^\n]*)\n(.*?)^```[ \t]*$", content,
                            flags=re.MULTILINE | re.DOTALL)
        if not blocks:
            raise
        if len(blocks) != 1 or blocks[0][0].strip().lower() not in {"", "json"}:
            raise ValueError("expected plain JSON or one JSON code block")
        value = json.loads(blocks[0][1])
    if not isinstance(value, dict):
        raise ValueError("source-review prediction must be a JSON object")
    return value


def run_review(review: ReviewInput, client, store: RunStore) -> dict[str, Any]:
    freeze_json(store.root / "review_manifest.json", {
        "workflow": WORKFLOW, "input_id": review.input_id,
        "model_profile": validated_model_profile(client), "config": CONFIG.public_dict(),
        "schema": review.schema,
    })
    contract = RepoPredictionContract("silentswap", review.schema, source_texts=review.sources)
    messages = review.messages
    if (store.root / "prediction.json").exists():
        save_request(store, messages)
        saved = read_json(store.root / "prediction.json")
        if saved.pop("input_id", None) != review.input_id or saved.pop("benchmark", None) != "silentswap":
            raise AgentLoopError("saved prediction belongs to a different sample")
        return contract.validate(saved,
                                 input_id=review.input_id, observed_spans=review.spans)
    for correction in range(CONFIG.format_repair_attempts + 1):
        content = _complete(messages, client, store, correction)
        try:
            prediction = contract.validate(parse_review_prediction(content), input_id=review.input_id,
                                           observed_spans=review.spans)
        except (ValueError, PredictionError) as error:
            store._write_json(store.root / "validation_errors" / f"correction_{correction}.json", {
                "format_retry": correction, "error_type": type(error).__name__, "error": str(error),
            })
            if correction == CONFIG.format_repair_attempts:
                raise AgentLoopError(f"source-review prediction is invalid: {error}") from error
            messages = [*review.messages, {"role": "assistant", "content": content},
                {"role": "user", "content": "Correct this output validation error using only the supplied "
                 f"document and source. Return the complete prediction JSON.\n{error}"}]
            continue
        store._write_json(store.root / "prediction.json", prediction)
        store.save_state({"workflow": WORKFLOW, "input_id": review.input_id, "status": "complete",
                          "format_corrections": correction})
        return prediction
    raise AssertionError("unreachable source review state")


def combine_judgments(stage1: dict, stage2: dict) -> dict:
    if stage1["input_id"] != stage2["input_id"] or any(
        stage["benchmark"] != "silentswap" for stage in (stage1, stage2)
    ):
        raise ValueError("two-stage judgments must belong to the same SilentSwap sample")
    return {
        "input_id": stage1["input_id"], "benchmark": "silentswap", "workflow": WORKFLOW,
        "scores": {
            "localization_score": stage1["rule_based"]["localization_score"],
            "location_correct": stage1["llm_judge"]["scores"]["location_correct"],
            "code_change_correct": stage2["llm_judge"]["scores"]["code_change_correct"],
        },
        "metric_sources": {"localization_score": "stage1", "location_correct": "stage1",
                           "code_change_correct": "stage2"},
    }
