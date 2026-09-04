"""Run one frozen BEG evaluation sample."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlsplit


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agentloop import (
    DEFAULT_CONFIG as AGENTLOOP_CONFIG,
    AgentLoop,
    OpenAICompatibleJsonClient,
    RawBackend,
    RunStore,
    prepare_initial_request,
)
from agentloop.errors import AgentLoopError
from agentloop.benchmark_configs import (
    PredictionRequestConfig,
    repo_benchmark_config,
)
from beg.behavior_directory import DIRECTORY_ENCODING
from beg.evidence_intake import load_visible_bundle
from tracereview import (
    DEFAULT_CONFIG as TRACE_REVIEW_CONFIG,
    TraceReview,
    prepare_trace_request,
    render_raw_trace_payload,
    render_trace_view,
)
from evaluation_core.contracts import load_prediction_schema
from evaluation_core.messages import (
    build_baseline_messages,
    build_trace_review_messages,
    load_task_prompt,
)
from scripts.directory_pipeline import (
    token_counter,
    validate_directory_artifact,
)

def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        result = _run(args)
    except (AgentLoopError, OSError, json.JSONDecodeError, ValueError) as error:
        result = {
            "status": "failed",
            "input_id": args.input_id,
            "benchmark": args.benchmark,
            "arm": args.arm,
            "failure": str(error),
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") in {"complete", "prepared"} else 1


def _run(args: argparse.Namespace) -> dict[str, Any]:
    artifact_root = (
        Path(args.artifact_root)
        if args.artifact_root
        else PROJECT_ROOT / "evaluation" / args.benchmark / "artifacts"
    )
    store = RunStore(args.output)

    if args.benchmark == "feedbacktrace":
        if args.arm == "graph":
            graph = _load_graph(artifact_root, args.input_id)
            view = render_trace_view(graph)
            messages = build_trace_review_messages(
                input_id=view.input_id,
                task_prompt=load_task_prompt("BEG", "feedbacktrace"),
                trace_view=view.text,
            )
            prompt_variant = "BEG"
        else:
            payload = _load_raw_trace_payload(artifact_root, args.input_id)
            view = render_raw_trace_payload(payload)
            messages = build_baseline_messages(
                "feedbacktrace",
                trace_payload=payload,
            )
            prompt_variant = "baseline"
        schema = load_prediction_schema(args.schema_root, "feedbacktrace")
        token_count, request_policy = prepare_trace_request(
            messages,
            config=TRACE_REVIEW_CONFIG,
        )
        store.save_trace_view(view.text)
        store.save_request_if_unchanged(
            1,
            messages=messages,
            token_count=token_count,
            compression=request_policy,
            provider_retry=0,
        )
        if args.prepare_only:
            return {
                "status": "prepared",
                "input_id": args.input_id,
                "benchmark": args.benchmark,
                "arm": args.arm,
                "prompt_variant": prompt_variant,
                "estimated_input_tokens": token_count,
            }
        client = _client(
            args,
            prediction_request=_trace_prediction_request(args.model),
        )
        prediction = TraceReview(
            view=view,
            messages=messages,
            prediction_schema=schema,
            client=client,
            store=store,
            config=TRACE_REVIEW_CONFIG,
        ).run()
        return {
            "status": "complete",
            "input_id": args.input_id,
            "benchmark": args.benchmark,
            "arm": args.arm,
            "prompt_variant": prompt_variant,
            "turns": 1,
            "prediction": prediction,
        }

    bundle = load_visible_bundle(
        artifact_root / "visible_bundles" / args.input_id
    )
    if bundle.task_document is None:
        raise AgentLoopError("Repo benchmark bundle lacks its task document")
    benchmark_config = repo_benchmark_config(args.benchmark)
    benchmark_config.validate_for(bundle.benchmark)
    count_directory_tokens = token_counter(DIRECTORY_ENCODING)
    prompt_variant = "baseline" if args.arm == "raw" else "BEG"
    if args.arm == "raw":
        backend = RawBackend(
            bundle,
            index_budget=AGENTLOOP_CONFIG.index_budget,
            count_tokens=count_directory_tokens,
        )
    else:
        from agentloop.graph_backend import GraphBackend

        graph = _load_graph(artifact_root, args.input_id)
        directory_root = (
            artifact_root
            / "behavior_directories"
            / args.input_id
        )
        graph_root = artifact_root / "behavior_graphs" / args.input_id
        directory_summary = validate_directory_artifact(
            bundle.root,
            graph_root,
            directory_root,
            encoding_name=DIRECTORY_ENCODING,
        )
        directory = _load_json(directory_root / "ranked_directory.json")
        backend = GraphBackend(
            bundle,
            graph,
            directory,
            count_tokens=count_directory_tokens,
        )
        if backend.initial_index_tokens != int(directory_summary["token_count"]):
            raise AgentLoopError("Ranked Directory token count changed during loading")
    module7 = benchmark_config.bind_module7(
        args.schema_root,
        input_id=bundle.input_id,
        task_document_name=bundle.task_document.path,
        task_document=bundle.task_document.content,
        initial_index=backend.initial_index,
        prompt_variant=prompt_variant,
    )
    prepared = prepare_initial_request(
        module7.initial_user_prompt,
        module7.max_rounds,
        config=AGENTLOOP_CONFIG,
        priority_groups=backend.priority_groups,
    )
    store.save_request_if_unchanged(
        1,
        messages=list(prepared.messages),
        token_count=prepared.token_count,
        compression=prepared.manifest.to_dict(),
        provider_retry=0,
    )
    if args.prepare_only:
        return {
            "status": "prepared",
            "input_id": args.input_id,
            "benchmark": args.benchmark,
            "arm": args.arm,
            "prompt_variant": prompt_variant,
            "estimated_input_tokens": prepared.token_count,
        }
    client = _client(
        args,
        prediction_request=benchmark_config.prediction_request,
    )
    outcome = AgentLoop(
        backend=backend,
        initial_user_prompt=module7.initial_user_prompt,
        max_rounds=module7.max_rounds,
        finish_contract=module7.finish_contract,
        client=client,
        store=store,
        config=AGENTLOOP_CONFIG,
    ).run()
    return {
        "status": outcome.status,
        "input_id": args.input_id,
        "benchmark": args.benchmark,
        "arm": args.arm,
        "prompt_variant": prompt_variant,
        "turns": outcome.turns,
        "prediction": outcome.prediction,
        "failure": outcome.failure,
    }


def _client(
    args: argparse.Namespace,
    *,
    prediction_request: PredictionRequestConfig | None = None,
) -> OpenAICompatibleJsonClient:
    base_url = args.base_url or os.environ.get("BEG_API_BASE_URL", "")
    model = args.model or os.environ.get("BEG_MODEL", "")
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key and urlsplit(base_url).hostname == "127.0.0.1":
        api_key = "unused-placeholder"
    if not base_url or not model or not api_key:
        raise AgentLoopError(
            "set --base-url/BEG_API_BASE_URL, --model/BEG_MODEL, and "
            f"the {args.api_key_env} environment variable"
        )
    request_config = prediction_request or PredictionRequestConfig(
        thinking="disabled",
        reasoning_effort=None,
    )
    return OpenAICompatibleJsonClient(
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout=args.timeout,
        disable_thinking=request_config.thinking == "disabled",
        reasoning_effort=request_config.reasoning_effort,
    )


def _trace_prediction_request(model: str | None) -> PredictionRequestConfig:
    """Use the provider's supported non-thinking control for TraceReview."""

    if isinstance(model, str) and model.casefold().startswith("deepseek-"):
        return PredictionRequestConfig(
            thinking="disabled",
            reasoning_effort=None,
        )
    return PredictionRequestConfig(
        thinking=None,
        reasoning_effort="none",
    )


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AgentLoopError(f"expected a JSON object: {path}")
    return value


def _load_graph(artifact_root: Path, input_id: str) -> dict[str, Any]:
    return _load_json(
        artifact_root / "behavior_graphs" / input_id / "behavior_graph.json"
    )


def _load_raw_trace_payload(
    artifact_root: Path, input_id: str
) -> dict[str, Any]:
    bundle_root = artifact_root / "visible_bundles" / input_id
    load_visible_bundle(bundle_root)
    payload = _load_json(bundle_root / "trace" / "model_input.json")
    events = payload.get("events")
    if payload.get("input_id") != input_id or not isinstance(events, list):
        raise AgentLoopError("FeedbackTrace visible input is invalid")
    selectable = sorted(
        str(event["evidence_id"])
        for event in events
        if isinstance(event, dict) and event.get("evidence_id") is not None
    )
    return {
        "input_id": input_id,
        "track": "long",
        "selectable_evidence_ids": selectable,
        "events": events,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one Raw, Graph, or FeedbackTrace BEG prediction."
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        choices=("specgap", "silentswap", "feedbacktrace"),
    )
    parser.add_argument("--input-id", required=True)
    parser.add_argument("--arm", choices=("raw", "graph"), default="graph")
    parser.add_argument(
        "--artifact-root",
        help="Defaults to evaluation/<benchmark>/artifacts.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--schema-root",
        type=Path,
        default=PROJECT_ROOT / "schemas",
    )
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--api-key-env", default="BEG_API_KEY")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Validate and persist turn one without constructing an API client.",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
