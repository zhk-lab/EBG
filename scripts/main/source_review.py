"""Prepare/run independent source recovery after saved BEG, or combine stage scores."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import fields
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from dotenv import load_dotenv
from agentloop.silentswap_source_review import (
    CONFIG, PROMPT_PATH, WORKFLOW, SELECTION_POLICY, build_review, combine_judgments,
    freeze_json, load_stage1, read_json, run_review, save_request,
)
from agentloop.storage import RunStore
from beg.evidence_intake import load_visible_bundle
from evaluation_core.contracts import load_prediction_schema
from scripts.main.layout import update_summary
from scripts.main.predict import _client, collect_response_usage
from scripts.main.judge import JudgeBatchConfig, run_batch_judges


def run_sample(args, manifest, input_id):
    store = RunStore(args.output / "runs" / input_id)
    try:
        prediction, state, source_state = load_stage1(args.source / "runs" / input_id)
        bundle = load_visible_bundle(args.artifact_root / "visible_bundles" / input_id)
        directory_path = args.artifact_root / "behavior_directories" / input_id / "ranked_directory.json"
        review = build_review(bundle, prediction, read_json(directory_path), prompt=manifest["source_review_prompt"],
                              state=state, schema_root=args.output / "schemas")
        # This record is private provenance; it is never appended to model messages.
        freeze_json(store.root / "stage1_selection.json", {
            "source_state": str(source_state.resolve()), "stage1_prediction": prediction,
            "directory_path": str(directory_path.resolve()),
            **review.selection,
        })
        estimated_tokens = save_request(store, review.messages)
        if args.prepare_only:
            return {"input_id": input_id, "status": "prepared", "estimated_input_tokens": estimated_tokens}
        client = _client(argparse.Namespace(
            base_url=manifest["base_url"], model=manifest["model"],
            api_key_env=manifest["api_key_env"], timeout=args.timeout,
            request_options=manifest["prediction_requests"]["silentswap"],
        ))
        run_review(review, client, store)
        record = {"input_id": input_id, "status": "complete", "estimated_input_tokens": estimated_tokens,
                  "usage": collect_response_usage(store.root)}
    except Exception as error:
        record = {"input_id": input_id, "status": "failed", "failure": f"{type(error).__name__}: {error}",
                  "usage": collect_response_usage(store.root)}
    store._write_json(store.root / "batch_result.json", record)
    return record


def aggregate(args, manifest):
    records, missing = [], []
    for input_id in manifest["selected_ids"]["silentswap"]:
        first_path = args.source / "judges" / args.judge_model / input_id / "result.json"
        second_path = args.output / "judges" / args.judge_model / input_id / "result.json"
        if not first_path.exists() or not second_path.exists():
            missing.append(input_id)
            continue
        first, second = read_json(first_path), read_json(second_path)
        result = combine_judgments(first, second)
        result["judge_model"] = args.judge_model
        result["stage1_result"] = str(first_path.resolve())
        result["stage2_result"] = str(second_path.resolve())
        result["stage1_code_change_correct"] = first["llm_judge"]["scores"]["code_change_correct"]
        records.append(result)
        path = args.output / "combined" / args.judge_model / input_id / "result.json"
        RunStore(path.parent)._write_json(path, result)
    complete = not missing
    # Do not turn absent judgments into zero scores or present a partial average as final.
    means = {
        key: sum(r["scores"][key] for r in records) / len(records)
        for key in ("localization_score", "location_correct", "code_change_correct")
    } if complete and records else None
    summary = {"workflow": WORKFLOW, "judge_model": args.judge_model,
               "selected_samples": len(manifest["selected_ids"]["silentswap"]),
               "completed_samples": len(records), "missing_samples": missing,
               "status": "complete" if complete else "incomplete", "score_means": means,
               "samples": records}
    path = args.output / "combined" / args.judge_model / "summary.json"
    RunStore(path.parent)._write_json(path, summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("terra", "sol", "gpt5.6", "kimi-k3", "deepseek_flash", "deepseek_pro", "claude"),
                        help="Use the model's formal SilentSwap BEG stage1/stage2 directories")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--artifact-root", type=Path, default=ROOT / "evaluation/silentswap/artifacts")
    parser.add_argument("--ids", nargs="+")
    parser.add_argument("--workers", type=int, help="Concurrency; resume saved value or use 25")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--model", help="Second-stage model ID; defaults to the saved first-stage model")
    phase = parser.add_mutually_exclusive_group()
    phase.add_argument("--prepare-only", action="store_true")
    phase.add_argument("--aggregate", action="store_true")
    phase.add_argument("--judge", action="store_true", help="Use the first-stage judge settings, then combine scores")
    phase.add_argument("--then-judge", action="store_true", help="Predict, then judge only if all predictions succeed")
    parser.add_argument("--judge-model", default="qwen3.7-max-2026-06-08")
    args = parser.parse_args(argv)
    if args.experiment:
        if args.source or args.output:
            parser.error("--experiment cannot be combined with --source or --output")
        experiment = ROOT / "experiments/silentswap" / args.experiment / "BEG"
        args.source, args.output = experiment / "stage1", experiment / "stage2"
        if args.experiment == "deepseek_flash" and not (args.judge or args.aggregate) and args.model is None:
            args.model = "deepseek-flash"
    if args.source is None:
        args.source = ROOT / "experiments/silentswap/terra/BEG/stage1"
    if args.output is None:
        parser.error("provide --experiment or --output")
    if args.workers is None:
        saved_manifest = (args.output / "judges" / args.judge_model / "manifest.json"
                          if args.judge else args.output / "manifest.json")
        args.workers = read_json(saved_manifest).get("workers", 25) if saved_manifest.exists() else 25
    args.source, args.output, args.artifact_root = (
        path.resolve() for path in (args.source, args.output, args.artifact_root)
    )
    if args.output == args.source or args.source in args.output.parents or args.output in args.source.parents:
        parser.error("output must be separate from the source experiment")
    if args.workers < 1 or args.timeout <= 0:
        parser.error("invalid workers, timeout, or phase combination")
    load_dotenv(ROOT / ".env", override=False)
    if args.aggregate or args.judge:
        if args.model:
            parser.error("--model applies only to second-stage prediction")
        manifest = read_json(args.output / "manifest.json")
        if manifest.get("source_experiment") != str(args.source) or manifest.get("workflow") != WORKFLOW:
            parser.error("aggregate source differs from the saved first-stage experiment")
        if args.judge:
            source_judge = read_json(args.source / "judges" / args.judge_model / "manifest.json")
            settings = {field.name: source_judge[field.name]
                        for field in fields(JudgeBatchConfig) if field.name in source_judge}
            policy = source_judge.get("retry_policy", {})
            for saved, setting in (("network_retries", "network_retries"),
                                   ("json_format_repairs", "format_repairs"),
                                   ("complete_sample_reruns", "deferred_retries")):
                if saved in policy:
                    settings[setting] = policy[saved]
            settings.update(experiment_name=args.output.name, experiment_root=args.output.parent,
                            benchmarks=("silentswap",), arms=("graph",),
                            artifact_root=args.artifact_root.parents[1], workers=args.workers)
            run_batch_judges(JudgeBatchConfig(**settings))
        result = aggregate(args, manifest)
        print(json.dumps({k: v for k, v in result.items() if k != "samples"}, ensure_ascii=False))
        return 0 if result["status"] == "complete" else 1

    source = read_json(args.source / "manifest.json")
    if source["benchmarks"] != ["silentswap"] or source["arms"] != ["graph"]:
        parser.error("source must be an ordinary single-arm SilentSwap BEG experiment")
    source_ids = source["selected_ids"]["silentswap"]
    ids = args.ids if args.ids is not None else source_ids
    if not ids or len(ids) != len(set(ids)) or any(i not in source_ids for i in ids):
        parser.error("IDs must be distinct members of the first-stage experiment")
    schema = load_prediction_schema(ROOT / "schemas", "silentswap")
    schema["properties"]["swaps"].update(minItems=5, maxItems=5)
    freeze_json(args.output / "schemas/silentswap_prediction.schema.json", schema)
    manifest = {
        **source, "experiment_name": args.output.name, "phase": "full",
        "selected_ids": {"silentswap": ids}, "split_id": "explicit_ids" if args.ids else source["split_id"],
        "prepare_only": False, "source_experiment": str(args.source), "workflow": WORKFLOW,
        "source_review_prompt": PROMPT_PATH.read_text(encoding="utf-8"),
        "source_review_config": CONFIG.public_dict(),
        "source_selection_policy": SELECTION_POLICY,
        "schema_root": str(args.output / "schemas"), "artifact_root": str(args.artifact_root.parents[1]),
        "workers": args.workers, "timeout": args.timeout,
        "single_sample_runner": "scripts.main.source_review.run_sample",
        "metric_sources": {"localization_score": "stage1", "location_correct": "stage1",
                           "code_change_correct": "stage2"},
    }
    manifest.pop("silentswap_retry_policy", None)
    if args.model:
        manifest["source_model"] = source["model"]
        manifest["model"] = args.model
    freeze_json(args.output / "manifest.json", manifest)
    records = []
    with ThreadPoolExecutor(max_workers=min(args.workers, len(ids))) as pool:
        jobs = [pool.submit(run_sample, args, manifest, input_id) for input_id in ids]
        for future in as_completed(jobs):
            record = future.result()
            records.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
    summary = {"workflow": WORKFLOW, "samples": sorted(records, key=lambda r: r["input_id"]),
               "totals": {"selected_samples": len(ids),
                          "completed_samples": sum(r["status"] == "complete" for r in records),
                          "prepared_samples": sum(r["status"] == "prepared" for r in records),
                          "failed_samples": sum(r["status"] == "failed" for r in records),
                          "stage2_usage": {
                              key: sum(r.get("usage", {}).get(key, 0) for r in records)
                              for key in ("calls", "input_tokens", "output_tokens", "total_tokens")
                          }}}
    update_summary(args.output, prediction=summary)
    print(json.dumps(summary["totals"]))
    if summary["totals"]["failed_samples"]:
        return 1
    if args.then_judge:
        saved_judge = args.output / "judges" / args.judge_model / "manifest.json"
        workers = read_json(saved_judge).get("workers", args.workers) if saved_judge.exists() else args.workers
        return main(["--source", str(args.source), "--output", str(args.output),
                     "--artifact-root", str(args.artifact_root), "--judge",
                     "--judge-model", args.judge_model, "--workers", str(workers)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
