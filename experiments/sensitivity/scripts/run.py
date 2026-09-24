"""Prepare, predict and judge Repo sensitivity experiments; resume per sample."""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from dotenv import load_dotenv
from agentloop import RunStore
from agentloop.config import DEFAULT_CONFIG
from agentloop.context import FastTokenCounter
from agentloop.evidence import render_tool_result
from agentloop.graph_backend import GraphBackend
from agentloop.silentswap_source_review import (
    PROMPT_PATH, build_review, freeze_json, load_stage1, run_review, save_request,
)
from ebg.evidence_intake import load_visible_bundle
from evaluation_core.messages import load_task_prompt
from scripts.main import judge, predict

BENCHMARKS = ("specgap", "silentswap")
EXPERIMENTS = ("expansion_hops", "max_rounds")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    RunStore(Path(path).parent)._write_json(Path(path), value)


@dataclass(frozen=True)
class Condition:
    benchmark: str
    arm: str
    hops: int
    rounds: int

    @property
    def key(self):
        return f"{self.arm}_h{self.hops}_r{self.rounds}"

    def root(self, results, model, stage="stage1"):
        return Path(results) / "conditions" / model / self.benchmark / self.key / stage

    @property
    def has_stage2(self):
        return self.benchmark == "silentswap" and self.arm == "graph"


def comparisons(config, benchmark, experiments=EXPERIMENTS):
    """Logical contrasts share canonical default conditions on disk."""
    default_rounds = config["defaults"]["max_rounds"][benchmark]
    for experiment in experiments:
        values = config[experiment] if experiment == "expansion_hops" else config[experiment][benchmark]
        for value in values:
            hops = value if experiment == "expansion_hops" else 1
            rounds = default_rounds if experiment == "expansion_hops" else value
            yield (experiment, value, Condition(benchmark, "raw", 1, rounds),
                   Condition(benchmark, "graph", hops, rounds))


def conditions(config, benchmark, experiments=EXPERIMENTS):
    return list(dict.fromkeys(c for _, _, raw, graph in comparisons(config, benchmark, experiments)
                              for c in (raw, graph)))


class MeasuredGraphBackend(GraphBackend):
    """Save diagnostics separately from the text returned to the model."""

    def __init__(self, *args, diagnostics_path, **kwargs):
        super().__init__(*args, **kwargs)
        self.diagnostics_path = diagnostics_path
        self.diagnostics = read_json(diagnostics_path) if diagnostics_path.exists() else {}

    def local_graph(self, read_id, **kwargs):
        value = super().local_graph(read_id, **kwargs)
        self.diagnostics[read_id] = dict(self._retriever.last_read_stats)
        write_json(self.diagnostics_path, self.diagnostics)
        return value


def run_one(args, condition, tool_result_budget=None):
    def factory(*pos, **kwargs):
        return MeasuredGraphBackend(*pos, **kwargs, expansion_hops=condition.hops,
                                    diagnostics_path=args.output / "read_diagnostics.json")
    return predict._run(args, max_rounds=condition.rounds,
                        tool_result_budget=tool_result_budget,
                        backend_factory=factory if condition.arm == "graph" else None)


def manifest(output, condition, ids, profile, stage="stage1", tool_result_budget=None):
    limits = replace(DEFAULT_CONFIG, format_repair_attempts=1) if condition.benchmark == "silentswap" else DEFAULT_CONFIG
    if stage == "stage1" and tool_result_budget is not None:
        limits = replace(limits, tool_result_budget=tool_result_budget)
    prompt = (PROMPT_PATH.read_text(encoding="utf-8") if stage == "stage2" else
              load_task_prompt("EBG" if condition.arm == "graph" else "baseline", condition.benchmark))
    value = {
        "experiment_name": output.name, "split_id": "hyperparameter_main_key", "phase": "full",
        "benchmarks": [condition.benchmark], "arms": [condition.arm],
        "selected_ids": {condition.benchmark: ids}, "prepare_only": False,
        "implementation_version": 1, "condition": condition.__dict__, "stage": stage,
        "task_prompt": prompt, "runner_limits": limits.public_dict(), **profile,
    }
    freeze_json(output / "manifest.json", value)


def predict_sample(config, model, condition, input_id, output):
    path = output / "runs" / input_id / "batch_result.json"
    if path.exists() and (path.parent / "prediction.json").exists():
        saved = read_json(path)
        if saved["status"] == "complete":
            return {**saved, "resumed": True}
    profile = config["models"][model][condition.benchmark]
    batch = predict.BatchConfig(
        experiment_name=output.name, experiment_root=output.parent, split_file=None, phase="full",
        benchmarks=(condition.benchmark,), arms=(condition.arm,),
        model=profile["model"], base_url=profile["base_url"], api_key_env=profile["api_key_env"],
        request_options=profile["request_options"], timeout=config["timeout"],
    )
    job = (condition.benchmark, condition.arm, input_id)
    runner = lambda args: run_one(args, condition, config.get("tool_result_budget"))
    result = predict._run_job(batch, job, runner, lambda _: None)
    if result["status"] == "failed" and not result.get("auto_retried"):
        result = predict._run_job(batch, job, runner, lambda _: None, attempt=2)
    return result


def stage2_sample(config, model, input_id, first, second):
    first_result = first / "runs" / input_id / "batch_result.json"
    if not first_result.exists() or read_json(first_result)["status"] != "complete":
        return {"input_id": input_id, "status": "blocked", "failure": "stage1 incomplete"}
    store = RunStore(second / "runs" / input_id)
    prediction, state, _ = load_stage1(first / "runs" / input_id)
    # Invalidate reuse when the upstream prediction or reading history changes.
    freeze_json(store.root / "stage1_input.json", {"prediction": prediction, "records": state["records"]})
    result_path = store.root / "batch_result.json"
    if result_path.exists() and (store.root / "prediction.json").exists():
        saved = read_json(result_path)
        if saved["status"] == "complete":
            return {**saved, "resumed": True}
    artifacts = ROOT / "data/prepared/silentswap/artifacts"
    bundle = load_visible_bundle(artifacts / "visible_bundles" / input_id)
    directory = read_json(artifacts / "behavior_directories" / input_id / "ranked_directory.json")
    review = build_review(bundle, prediction, directory, state=state)
    write_json(store.root / "selection.json", review.selection)
    save_request(store, review.messages)
    profile = config["models"][model]["silentswap"]
    try:
        run_review(review, predict._client(argparse.Namespace(**profile, timeout=config["timeout"])), store)
        result = {"input_id": input_id, "status": "complete"}
    except Exception as error:
        result = {"input_id": input_id, "status": "failed", "failure": f"{type(error).__name__}: {error}"}
    result["usage"] = predict.collect_response_usage(store.root)
    write_json(result_path, result)
    return result


def prepare_sample(condition, input_id, output):
    """Offline initial request plus one representative read; no API clients."""
    captured = []

    def factory(*pos, **kwargs):
        backend = MeasuredGraphBackend(*pos, **kwargs, expansion_hops=condition.hops,
                                       diagnostics_path=output / "read_diagnostics.json")
        captured.append(backend)
        return backend

    args = argparse.Namespace(benchmark=condition.benchmark, input_id=input_id, arm=condition.arm,
                              artifact_root=None, output=output, schema_root=ROOT / "schemas", prepare_only=True)
    result = predict._run(args, max_rounds=condition.rounds,
                          backend_factory=factory if condition.arm == "graph" else None)
    if captured:
        backend = captured[0]
        ids = [e["read_id"] for e in backend._retriever.directory["entries"]
               if e["read_id"] in backend.initial_ids][:DEFAULT_CONFIG.max_read_ids]
        counter = FastTokenCounter(DEFAULT_CONFIG.token_estimator).count_text
        read = backend.read(ids, token_budget=DEFAULT_CONFIG.tool_result_budget,
                            max_atomic_unit_tokens=DEFAULT_CONFIG.max_atomic_unit_tokens, count_tokens=counter)
        (output / "read.txt").write_text(render_tool_result(read), encoding="utf-8")
        result["returned_ids"] = [u.unit_id for u in read.units]
    write_json(output / "preview.json", {**result, "condition": condition.__dict__})
    return result


def judge_sample(batch, benchmark, arm, input_id, module, factory):
    result = judge._run_job(batch, benchmark, arm, input_id, module, factory)
    if batch.deferred_retries and judge._needs_deferred_retry(batch, result):
        result = judge._run_deferred_retry(batch, result, module, factory)
    return result


def parallel(function, ids, workers, label):
    failures = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        tasks = {pool.submit(function, i): i for i in ids}
        for future in as_completed(tasks):
            try:
                result = future.result()
            except Exception as error:
                result = {"input_id": tasks[future], "status": "failed", "failure": f"{type(error).__name__}: {error}"}
            print(json.dumps({**label, **result}, ensure_ascii=False), flush=True)
            failures += result["status"] not in ("complete", "prepared")
    return failures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "prepare", "predict", "judge"))
    parser.add_argument("--config", type=Path, default=HERE / "configs/config.json")
    parser.add_argument("--samples", type=Path, default=HERE / "configs/samples.json")
    parser.add_argument("--results", type=Path, default=ROOT / "outputs/analysis/sensitivity/results")
    parser.add_argument("--experiment", nargs="+", choices=EXPERIMENTS)
    parser.add_argument("--model", nargs="+", help="Model keys from the selected config")
    parser.add_argument("--benchmark", nargs="+", choices=BENCHMARKS)
    parser.add_argument("--arm", nargs="+", choices=("raw", "graph"))
    parser.add_argument("--ids", nargs="+")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--reuse-existing-manifest",
        action="store_true",
        help="judge saved predictions without rewriting their frozen prediction manifest",
    )
    args = parser.parse_args(argv)
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("workers and limit must be positive")
    if args.reuse_existing_manifest and args.command != "judge":
        parser.error("--reuse-existing-manifest is only valid with the judge command")
    config, samples = read_json(args.config), read_json(args.samples)
    models = args.model or list(config["models"])
    if any(model not in config["models"] for model in models):
        parser.error("unknown model; choose from: " + ", ".join(config["models"]))
    experiments = args.experiment or EXPERIMENTS
    plan = []
    for benchmark in args.benchmark or BENCHMARKS:
        for experiment, value, raw, graph in comparisons(config, benchmark, experiments):
            plan.append({"experiment": experiment, "value": value, "benchmark": benchmark,
                         "baseline": raw.key, "EBG": graph.key})
    if args.command == "plan":
        write_json(HERE / "configs/plan.json", {"models": models, "comparisons": plan})
        print(json.dumps(plan, indent=2))
        return 0
    if args.command in ("predict", "judge"):
        load_dotenv(ROOT / ".env", override=False)
    failures = 0
    for benchmark in args.benchmark or BENCHMARKS:
        all_ids = samples[benchmark]
        ids = args.ids or all_ids
        if len(ids) != len(set(ids)) or any(i not in all_ids for i in ids):
            parser.error(f"IDs must be unique members of {benchmark}")
        ids = ids[:args.limit] if args.limit else ids
        for condition in conditions(config, benchmark, experiments):
            if args.arm and condition.arm not in args.arm:
                continue
            if args.command == "prepare":
                failures += parallel(lambda i: prepare_sample(condition, i, args.results / "previews" /
                                     benchmark / condition.key / i), ids, args.workers, {"condition": condition.key})
                continue
            for model in models:
                first = condition.root(args.results, model)
                profile = config["models"][model][benchmark]
                stages = [first]
                if condition.has_stage2:
                    stages.append(condition.root(args.results, model, "stage2"))
                for stage in stages:
                    if args.command == "judge" and not (stage / "manifest.json").exists():
                        parser.error(f"run predictions first: {stage}")
                    if args.command != "judge" or not args.reuse_existing_manifest:
                        manifest(stage, condition, all_ids, profile, stage.name, config.get("tool_result_budget"))
                label = {"benchmark": benchmark, "model": model, "condition": condition.key}
                if args.command == "predict":
                    failures += parallel(lambda i: predict_sample(config, model, condition, i, first),
                                         ids, args.workers, {**label, "stage": "stage1"})
                    if condition.has_stage2:
                        failures += parallel(lambda i: stage2_sample(config, model, i, first, stages[1]),
                                             ids, args.workers, {**label, "stage": "stage2"})
                else:
                    for stage in stages:
                        available = [i for i in ids if (stage / "runs" / i / "prediction.json").exists()
                                     and (stage / "runs" / i / "batch_result.json").exists()
                                     and read_json(stage / "runs" / i / "batch_result.json")["status"] == "complete"]
                        failures += len(ids) - len(available)
                        profile_j = config["judges"][benchmark]
                        batch = judge.JudgeBatchConfig(
                            experiment_name=stage.name, experiment_root=stage.parent, phase="full",
                            benchmarks=(benchmark,), arms=(condition.arm,), judge_model=profile_j["model"],
                            base_url=profile_j["base_url"], api_key_env=profile_j["api_key_env"],
                            request_options=profile_j["request_options"], workers=args.workers, timeout=config["timeout"])
                        freeze_json(stage / "judges" / batch.judge_model / "profile.json", profile_j)
                        module, factory = judge._load_judge_module(benchmark), judge._default_client_factory(batch)
                        failures += parallel(lambda i: judge_sample(batch, benchmark, condition.arm, i, module, factory),
                                             available, args.workers, {**label, "stage": stage.name})
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
