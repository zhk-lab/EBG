"""Summarize paired scores, rounds, depth and tokens; no plotting dependencies."""

import argparse
import csv
import itertools
import random
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

from run import (
    BENCHMARKS, EXPERIMENTS, HERE, RunStore, comparisons, conditions,
    load_stage1, read_json, write_json,
)

METRICS = {
    "specgap": ("f1", "question_quality", "location_f1"),
    "silentswap": ("localization_score", "location_correct", "code_change_correct"),
}
TOKEN_KEYS = ("input_tokens", "output_tokens", "total_tokens")


def usage_sum(usages):
    totals = dict.fromkeys(TOKEN_KEYS, 0)
    for usage in usages:
        incoming = usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0
        outgoing = usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0
        totals["input_tokens"] += incoming
        totals["output_tokens"] += outgoing
        totals["total_tokens"] += usage.get("total_tokens", incoming + outgoing) or 0
    return totals


def load_sample(results, config, model, condition, input_id):
    first = condition.root(results, model)
    roots = [first] + ([condition.root(results, model, "stage2")] if condition.has_stage2 else [])
    judged, batches, accepted_usage = [], [], []
    diagnostics, state = {}, None
    for root in roots:
        case = root / "runs" / input_id
        prediction_path = case / "batch_result.json"
        judge_path = root / "judges" / config["judges"][condition.benchmark]["model"] / input_id / "status.json"
        if not prediction_path.exists() or not judge_path.exists():
            return None
        batch, judgment = read_json(prediction_path), read_json(judge_path)
        assigned_zero = root != first and judgment.get("score_source") == "user_assigned_zero"
        if (batch["status"] != "complete" and not assigned_zero) or judgment["status"] != "complete":
            return None
        batches.append(batch)
        judged.append(judgment)
        if root == first:
            _, state, state_path = load_stage1(case)
            if state["max_rounds"] != condition.rounds:
                raise ValueError(f"round configuration mismatch: {state_path}")
            accepted_usage.extend(r.get("usage", {}) for r in [*state["records"], state["terminal_record"]])
            path = state_path.parent / "read_diagnostics.json"
            diagnostics = read_json(path) if path.exists() else {}
        elif not assigned_zero:
            second_state = read_json(case / "state.json")
            response = RunStore(case).load_response(1, format_retry=second_state["format_corrections"])
            accepted_usage.append(response.get("usage", {}))
    metrics = {k: judged[0]["metrics"][k] for k in METRICS[condition.benchmark]}
    if condition.has_stage2:
        metrics["code_change_correct"] = judged[1]["metrics"]["code_change_correct"]
    reads = [diagnostics[u["unit_id"]] for r in state["records"]
             if r["tool_result"]["action"] == "read" for u in r["tool_result"].get("units", [])
             if u["unit_id"] in diagnostics]
    return {"metrics": metrics, "rounds": state["turns"],
            "usage": usage_sum(accepted_usage),
            "actual_usage": usage_sum(b.get("actual_usage", b["usage"]) for b in batches),
            "actual_depth": max((r["actual_depth"] for r in reads), default=0) if condition.arm == "graph" else None,
            "read_diagnostics": reads}


def percentile(values, q):
    values = sorted(values)
    p = (len(values) - 1) * q
    left = int(p)
    return values[left] + (values[min(left + 1, len(values) - 1)] - values[left]) * (p - left)


def paired_statistics(pairs, repeats=2000, permutations=9999, seed=20260913):
    """Cluster bootstrap CI and one-sided paired cluster sign-flip test."""
    groups = defaultdict(list)
    for raw, graph, cluster in pairs:
        groups[cluster].append(graph - raw)
    sums = [sum(g) for g in groups.values()]
    counts = [len(g) for g in groups.values()]
    n, k = len(pairs), len(groups)
    result = {"n": n, "clusters": k, "baseline_mean": statistics.mean(p[0] for p in pairs),
              "ebg_mean": statistics.mean(p[1] for p in pairs),
              "delta": sum(sums) / n, "ci95": None, "p_value": None}
    if k < 2:
        return result
    rng = random.Random(seed)
    differences = []
    for _ in range(repeats):
        indices = rng.choices(range(k), k=k)
        differences.append(sum(sums[i] for i in indices) / sum(counts[i] for i in indices))
    result["ci95"] = [percentile(differences, .025), percentile(differences, .975)]
    observed = sum(sums)
    if k <= 16:
        flips = itertools.product((-1, 1), repeat=k)
        extreme = sum(sum(a * b for a, b in zip(sums, signs)) >= observed - 1e-12 for signs in flips)
        result["p_value"] = extreme / (2 ** k)
    else:
        extreme = sum(sum(v if rng.getrandbits(1) else -v for v in sums) >= observed - 1e-12
                      for _ in range(permutations))
        result["p_value"] = (extreme + 1) / (permutations + 1)
    return result


def holm(rows, family_size):
    valid = sorted((r for r in rows if r.get("p_value") is not None), key=lambda r: r["p_value"])
    previous = 0
    for index, row in enumerate(valid):
        previous = max(previous, min(1, (family_size - index) * row["p_value"]))
        row["p_adjusted"] = previous
        row["significantly_better"] = row["delta"] > 0 and previous < .05


def summarize(results, config, samples, *, models=None, benchmarks=BENCHMARKS,
              clusters=None, independent_samples=False, repeats=2000, permutations=9999):
    summaries, contrasts, sample_rows = [], [], []
    for benchmark in benchmarks:
        ids = samples[benchmark]
        if clusters is not None and any(i not in clusters.get(benchmark, {}) for i in ids):
            raise ValueError(f"cluster mapping must cover every {benchmark} sample")
        for model in models or config["models"]:
            loaded = {}
            for condition in conditions(config, benchmark):
                rows, progress = {}, {}
                finished_usage = []
                stages = ["stage1", "stage2"] if condition.has_stage2 else ["stage1"]
                for stage in stages:
                    progress[stage] = {"complete": 0, "failed": 0, "pending": 0}
                for input_id in ids:
                    for stage in stages:
                        path = condition.root(results, model, stage) / "runs" / input_id / "batch_result.json"
                        batch = read_json(path) if path.exists() else {}
                        status = batch.get("status", "pending")
                        progress[stage][status if status in progress[stage] else "pending"] += 1
                        finished_usage.append(batch.get("actual_usage", batch.get("usage", {})))
                    row = load_sample(results, config, model, condition, input_id)
                    if row is not None:
                        rows[input_id] = row
                        sample_rows.append({"benchmark": benchmark, "model": model, "condition": condition.key,
                                            "input_id": input_id, **row})
                loaded[condition] = rows
                complete = len(rows) == len(ids)
                turns = [r["rounds"] for r in rows.values()]
                summaries.append({"benchmark": benchmark, "model": model, "condition": condition.key,
                                  "status": "complete" if complete else "incomplete", "prediction_status": progress,
                                  "expected_samples": len(ids), "scored_samples": len(rows),
                                  "missing_ids": [i for i in ids if i not in rows],
                                  "metric_means": {k: statistics.mean(r["metrics"][k] for r in rows.values())
                                                   for k in METRICS[benchmark]} if complete else None,
                                  "rounds": {"mean": statistics.mean(turns), "min": min(turns), "max": max(turns),
                                             "at_cap_fraction": turns.count(condition.rounds) / len(turns)} if turns else None,
                                  "prediction_tokens": usage_sum(r["usage"] for r in rows.values()),
                                  "actual_prediction_tokens": usage_sum(r["actual_usage"] for r in rows.values()),
                                  "all_finished_prediction_tokens": usage_sum(finished_usage),
                                  "mean_actual_depth": statistics.mean(r["actual_depth"] for r in rows.values())
                                                       if rows and condition.arm == "graph" else None})
            for experiment, value, raw, graph in comparisons(config, benchmark):
                paired_ids = [i for i in ids if i in loaded[raw] and i in loaded[graph]]
                complete = len(paired_ids) == len(ids)
                for metric in METRICS[benchmark]:
                    row = {"experiment": experiment, "value": value, "benchmark": benchmark, "model": model,
                           "metric": metric, "paired_samples": len(paired_ids), "expected_samples": len(ids),
                           "status": "complete" if complete else "incomplete", "baseline_mean": None,
                           "ebg_mean": None, "delta": None, "ci95": None, "p_value": None,
                           "p_adjusted": None, "significantly_better": None}
                    if complete:
                        pairs = [(loaded[raw][i]["metrics"][metric], loaded[graph][i]["metrics"][metric],
                                  clusters[benchmark][i] if clusters else i) for i in ids]
                        if clusters is not None or independent_samples:
                            row.update(paired_statistics(pairs, repeats, permutations))
                        else:
                            row.update(baseline_mean=statistics.mean(p[0] for p in pairs),
                                       ebg_mean=statistics.mean(p[1] for p in pairs),
                                       delta=statistics.mean(p[1] - p[0] for p in pairs),
                                       inference_status="repository grouping not specified")
                    contrasts.append(row)
    for experiment in EXPERIMENTS:
        # Keep the full planned family even when CLI filters select fewer models/benchmarks.
        family_size = sum(len([x for x in comparisons(config, b, [experiment])]) * len(METRICS[b])
                          for b in BENCHMARKS) * len(config["models"])
        holm([r for r in contrasts if r["experiment"] == experiment], family_size)
    return {"delta_definition": "EBG minus baseline", "conditions": summaries, "comparisons": contrasts,
            "samples": sample_rows, "inference": "cluster percentile bootstrap CI; one-sided cluster sign-flip p; Holm per experiment",
            "token_scope": "accepted calls in successful trajectories; excludes judge and graph construction; EBG SilentSwap includes stage2",
            "grouping": "repository" if clusters is not None else "independent samples" if independent_samples else "unspecified; descriptive only"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "outputs/analysis/sensitivity/results")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", type=Path, default=HERE / "configs/config.json")
    parser.add_argument("--samples", type=Path, default=HERE / "configs/samples.json")
    parser.add_argument("--model", nargs="+")
    parser.add_argument("--benchmark", nargs="+", choices=BENCHMARKS)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--clusters", type=Path, help="benchmark -> input_id -> repository ID")
    group.add_argument("--independent-samples", action="store_true", help="Use only when samples are independent repositories")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--permutations", type=int, default=9999)
    args = parser.parse_args(argv)
    if min(args.bootstrap, args.permutations) < 1:
        parser.error("resampling counts must be positive")
    config = read_json(args.config)
    if args.model and any(m not in config["models"] for m in args.model):
        parser.error("unknown model")
    result = summarize(args.results, config, read_json(args.samples), models=args.model,
                       benchmarks=args.benchmark or BENCHMARKS,
                       clusters=read_json(args.clusters) if args.clusters else None,
                       independent_samples=args.independent_samples, repeats=args.bootstrap, permutations=args.permutations)
    out = args.output or ROOT / "outputs/analysis/sensitivity/data/summary"
    write_json(out / "summary.json", result)
    rows = [{k: v for k, v in row.items() if k != "ci95"} |
            {"ci_low": (row["ci95"] or [None, None])[0], "ci_high": (row["ci95"] or [None, None])[1]}
            for row in result["comparisons"]]
    with (out / "comparisons.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved JSON/CSV to {out}; conclusions belong in {ROOT / "outputs/analysis/sensitivity/REPORT.md"}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
