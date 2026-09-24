"""Summarize saved ablation scores and token usage, with paired bootstrap CIs; no plots."""

import argparse
import csv
import random
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

from run import HERE, VARIANTS, read_json, stage_root, supported, write_json

METRICS = {
    "specgap": {"f1": "Macro F1", "question_quality": "Question Quality", "location_f1": "Location F1"},
    "silentswap": {"localization_score": "Localization Score", "location_correct": "Location Correct",
                   "code_change_correct": "Code Correct"},
    "feedbacktrace": {"verification_point_alignment": "Verification Point Alignment",
                      "evidence_location_score": "Evidence Location Score", "evidence_hit_rate": "Evidence Hit Rate"},
}
TOKEN_KEYS = ("input_tokens", "output_tokens", "total_tokens")


def percentile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    left = int(position)
    right = min(left + 1, len(ordered) - 1)
    return ordered[left] + (ordered[right] - ordered[left]) * (position - left)


def paired_summary(pairs, *, repeats=2000, seed=20260913):
    """Pairs are (full, ablated, cluster_id); resample clusters with shared indices."""
    if not pairs:
        return None
    groups = defaultdict(list)
    for full, ablated, cluster in pairs:
        groups[cluster].append((full, ablated))
    values = list(groups.values())
    rng = random.Random(seed)
    differences = []
    for _ in range(repeats):
        sampled = [pair for group in rng.choices(values, k=len(values)) for pair in group]
        differences.append(statistics.mean(b - a for a, b in sampled))
    # A single independent unit cannot support an inferential confidence interval.
    ci = [percentile(differences, .025), percentile(differences, .975)] if len(groups) > 1 else None
    return {"n": len(pairs), "clusters": len(groups),
            "full_mean": statistics.mean(a for a, _, _ in pairs),
            "ablated_mean": statistics.mean(b for _, b, _ in pairs),
            "delta": statistics.mean(b - a for a, b, _ in pairs), "ci95": ci}


def load_sample(results, config, model, benchmark, variant, input_id):
    first = stage_root(results, model, benchmark, variant)
    roots = [first]
    if benchmark == "silentswap":
        roots.append(stage_root(results, model, benchmark, variant, "stage2"))
    judge_model = config["judges"][benchmark]["model"]
    predictions, judgments = [], []
    for root in roots:
        prediction_path = root / "runs" / input_id / "batch_result.json"
        judge_path = root / "judges" / judge_model / input_id / "status.json"
        if not prediction_path.exists() or not judge_path.exists():
            return None
        p, j = read_json(prediction_path), read_json(judge_path)
        failure_zero = (benchmark == "silentswap" and root.name == "stage2"
                        and p["status"] == "failed"
                        and j.get("score_origin") == "user_requested_prediction_failure_zero"
                        and j.get("metrics", {}).get("code_change_correct") == 0)
        if (p["status"] != "complete" and not failure_zero) or j["status"] != "complete":
            return None
        predictions.append(p)
        judgments.append(j)
    metrics = {k: judgments[0]["metrics"][k] for k in METRICS[benchmark]}
    if benchmark == "silentswap":
        metrics["code_change_correct"] = judgments[1]["metrics"]["code_change_correct"]
    usage = {k: sum(p["usage"].get(k, 0) for p in predictions) for k in TOKEN_KEYS}
    actual = {k: sum(p.get("actual_usage", p["usage"]).get(k, 0) for p in predictions) for k in TOKEN_KEYS}
    return {"metrics": metrics, "usage": usage, "actual_usage": actual}


def prediction_progress(results, model, benchmark, variant, ids):
    """Count failed/incomplete predictions and their actual usage separately from scored rows."""
    stages = ("stage1", "stage2") if benchmark == "silentswap" else ("stage1",)
    counts = {stage: {"complete": 0, "failed": 0, "pending": 0} for stage in stages}
    tokens = {key: 0 for key in TOKEN_KEYS}
    failures = []
    for stage in stages:
        root = stage_root(results, model, benchmark, variant, stage)
        for input_id in ids:
            path = root / "runs" / input_id / "batch_result.json"
            if not path.exists():
                counts[stage]["pending"] += 1
                continue
            record = read_json(path)
            status = record["status"] if record["status"] in ("complete", "failed") else "pending"
            counts[stage][status] += 1
            usage = record.get("actual_usage", record.get("usage", {}))
            for key in tokens:
                tokens[key] += usage.get(key, 0)
            if status == "failed":
                failures.append({"input_id": input_id, "stage": stage, "failure": record.get("failure")})
    return {"prediction_status": counts, "all_finished_prediction_tokens": tokens,
            "prediction_failures": failures}


def analyze(results, config, samples, *, repeats=2000, clusters=None, variants=VARIANTS):
    summaries, comparisons, records = [], [], []
    clusters = clusters or {}
    for model in config["models"]:
        for benchmark, ids in samples.items():
            loaded = {}
            for variant in variants:
                if not supported(benchmark, variant):
                    continue
                rows = {}
                for input_id in ids:
                    row = load_sample(results, config, model, benchmark, variant, input_id)
                    if row is not None:
                        rows[input_id] = row
                        records.append({"model": model, "benchmark": benchmark, "variant": variant,
                                        "input_id": input_id, **row["metrics"], **row["usage"]})
                loaded[variant] = rows
                complete = len(rows) == len(ids)
                summaries.append({"model": model, "benchmark": benchmark, "variant": variant,
                                  **prediction_progress(results, model, benchmark, variant, ids),
                                  "status": "complete" if complete else "incomplete",
                                  "expected_samples": len(ids), "completed_samples": len(rows),
                                  "missing_samples": [i for i in ids if i not in rows],
                                  "metric_means": {k: statistics.mean(r["metrics"][k] for r in rows.values())
                                                   for k in METRICS[benchmark]} if complete else None,
                                  "completed_sample_means": {k: statistics.mean(r["metrics"][k] for r in rows.values())
                                                             for k in METRICS[benchmark]} if rows else None,
                                  "completed_prediction_tokens": {k: sum(r["usage"][k] for r in rows.values()) for k in TOKEN_KEYS},
                                  "completed_actual_prediction_tokens": {k: sum(r["actual_usage"][k] for r in rows.values()) for k in TOKEN_KEYS}})
            for variant, rows in loaded.items():
                if variant == "full" or "full" not in loaded:
                    continue
                full = loaded["full"]
                paired_ids = [i for i in ids if i in full and i in rows]
                for metric, title in METRICS[benchmark].items():
                    pairs = [(full[i]["metrics"][metric], rows[i]["metrics"][metric],
                              clusters.get(benchmark, {}).get(i, i)) for i in paired_ids]
                    comparisons.append({"model": model, "benchmark": benchmark, "variant": variant,
                                        "metric": metric, "metric_name": title,
                                        "status": "complete" if len(paired_ids) == len(ids) else "incomplete",
                                        "expected_samples": len(ids), "paired_samples": len(paired_ids),
                                        "statistics": paired_summary(pairs, repeats=repeats) if len(paired_ids) == len(ids) else None})
    return {"delta_definition": "ablated minus full EBG; negative means the ablation scored lower",
            "ci_method": "paired percentile bootstrap; sample units unless a repository cluster mapping is supplied",
            "token_scope": "prediction input/output, including SilentSwap stage1+stage2; excludes judging and graph construction",
            "conditions": summaries, "comparisons": comparisons, "samples": records}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "outputs/analysis/ablation/results")
    parser.add_argument("--config", type=Path, default=HERE / "configs/config.json")
    parser.add_argument("--samples", type=Path, default=HERE / "configs/samples.json")
    parser.add_argument("--clusters", type=Path, help="Optional benchmark -> input_id -> repository ID mapping")
    parser.add_argument("--model", nargs="+", choices=("kimi-k3", "luna", "glm-5-3"))
    parser.add_argument("--benchmark", nargs="+", choices=tuple(METRICS))
    parser.add_argument("--variant", nargs="+", choices=VARIANTS)
    parser.add_argument("--output-dir", type=Path, help="Defaults to the input results directory")
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args(argv)
    if args.bootstrap < 1:
        parser.error("bootstrap must be positive")
    config, samples = read_json(args.config), read_json(args.samples)
    if args.model:
        config["models"] = {m: config["models"][m] for m in args.model}
    if args.benchmark:
        samples = {b: samples[b] for b in args.benchmark}
    result = analyze(args.results, config, samples, repeats=args.bootstrap,
                     clusters=read_json(args.clusters) if args.clusters else None,
                     variants=args.variant or VARIANTS)
    output_dir = args.output_dir or args.results
    write_json(output_dir / "summary.json", result)
    rows = []
    for item in result["comparisons"]:
        stats = item["statistics"] or {}
        ci = stats.get("ci95") or [None, None]
        rows.append({k: v for k, v in item.items() if k != "statistics"} |
                    {"full_mean": stats.get("full_mean"), "ablated_mean": stats.get("ablated_mean"),
                     "delta": stats.get("delta"), "ci_low": ci[0], "ci_high": ci[1]})
    if rows:
        with (output_dir / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    complete = sum(c["status"] == "complete" for c in result["conditions"])
    print(f"Completed conditions: {complete}/{len(result['conditions'])}; saved summary.json"
          + (" and comparison.csv" if rows else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
