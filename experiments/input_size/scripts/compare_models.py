"""Compare saved scores across evidence-search groups; no model or judge calls."""

import argparse
import csv
import json
import math
import statistics

import numpy as np

from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
JUDGES = ("qwen3.7-max-2026-06-08", "glm-5-2")
METRICS = {
    "specgap": {"f1": "Macro F1", "question_quality": "Question Quality", "location_f1": "Location F1"},
    "silentswap": {"localization_score": "Localization Score", "location_correct": "Location Correct", "code_change_correct": "Code Correct"},
    "feedbacktrace": {"verification_point_alignment": "Verification Point Alignment", "evidence_location_score": "Evidence Location Score", "evidence_hit_rate": "Evidence Hit Rate"},
}

MODELS = {
    "luna": "Luna", "deepseek_flash": "Flash", "deepseek_pro": "Pro",
    "terra": "Terra", "sol": "Sol", "kimi-k3": "K3",
    "claude": "Sonnet 5", "glm-5-3": "GLM-5-3 (low)",
}
PRIMARY = {component: next(iter(metrics)) for component, metrics in METRICS.items()}
METRIC_PAIRS = [(component, metric) for component, metrics in METRICS.items() for metric in metrics]
OUTPUT = ROOT / "outputs/analysis/input_size/results"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_scores(component, method, judge, input_id, *, model):
    directory = "gpt5.6" if model == "luna" and component != "feedbacktrace" else model
    base = ROOT / "outputs/main" / method / component / directory
    if component == "feedbacktrace" and judge == "glm-5-2":
        judge = "glm-5.2"  # Existing FeedbackTrace directory uses this spelling.
    if component == "silentswap" and method == "EBG":
        path = base / "stage2" / "combined" / judge / input_id / "result.json"
        data = read(path)
        if data["input_id"] != input_id:
            raise ValueError(f"Sample mismatch: {path}")
        scores = data["scores"]
    else:
        path = base / "judges" / judge / input_id / "status.json"
        data = read(path)
        if data["input_id"] != input_id or data["status"] != "complete":
            raise ValueError(f"Incomplete or mismatched score: {path}")
        scores = data["metrics"]
    selected = {key: float(scores[key]) for key in METRICS[component]}
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in selected.values()):
        raise ValueError(f"Invalid score: {path}")
    return selected, path.relative_to(ROOT).as_posix()


def aggregate(rows):
    output = []
    for judge in JUDGES:
        for component, metrics in METRICS.items():
            for label in ("Low", "Medium", "High", "All"):
                group = [r for r in rows if r["judge"] == judge and r["component"] == component
                         and (label == "All" or r["bin"] == label)]
                for metric in metrics:
                    pairs = {}
                    for row in group:
                        pair = pairs.setdefault(row["input_id"], {})
                        if row["method"] in pair:
                            raise ValueError("Duplicate score")
                        pair[row["method"]] = row["metrics"][metric]
                    if not pairs or any(set(p) != {"EBG", "baseline"} for p in pairs.values()):
                        raise ValueError("Missing paired scores")
                    ebg = statistics.mean(p["EBG"] for p in pairs.values())
                    baseline = statistics.mean(p["baseline"] for p in pairs.values())
                    output.append({"judge": judge, "component": component, "bin": label,
                                   "metric": metric, "n_pairs": len(pairs), "EBG": ebg,
                                   "baseline": baseline, "difference": ebg - baseline})
    return output


def group_medians(rows, repeats=2000):
    """Paired bootstrap of group medians and mean EBG-minus-baseline differences."""
    groups = {}
    for row in rows:
        key = (row["judge"], row["component"], row["bin"])
        pair = groups.setdefault(key, {}).setdefault(row["input_id"], {})
        if row["method"] in pair:
            raise ValueError("Duplicate score")
        pair[row["method"]] = row["metrics"]
    output = []
    for (judge, component, label), pairs in groups.items():
        if any(set(pair) != {"EBG", "baseline"} for pair in pairs.values()):
            raise ValueError("Missing paired scores")
        rng = np.random.default_rng(20260914)
        indices = rng.integers(0, len(pairs), size=(repeats, len(pairs)))
        for metric in next(iter(pairs.values()))["EBG"]:
            values = {method: np.array([pair[method][metric] for pair in pairs.values()])
                      for method in ("EBG", "baseline")}
            record = {"model": rows[0]["model"], "judge": judge, "component": component,
                      "bin": label, "metric": metric, "n_pairs": len(pairs)}
            for method, scores in values.items():
                record[f"{method}_median"] = float(np.median(scores))
                record[f"{method}_ci95"] = np.quantile(
                    np.median(scores[indices], axis=1), [0.025, 0.975]).tolist()
            difference = values["EBG"] - values["baseline"]
            record["mean_difference"] = float(difference.mean())
            record["mean_difference_ci95"] = np.quantile(
                difference[indices].mean(axis=1), [0.025, 0.975]).tolist()
            output.append(record)
    return output


def slope(x, y):
    """OLS slope with an intercept on the component-specific search scale."""
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("Expected at least two paired observations")
    center = statistics.mean(x)
    denominator = sum((v - center) ** 2 for v in x)
    if denominator == 0:
        raise ValueError("Search count has no variation")
    mean_y = statistics.mean(y)
    return sum((a - center) * (b - mean_y) for a, b in zip(x, y)) / denominator


def slope_interval(x, difference, repeats=2000):
    """Paired sample bootstrap on the same sample indices for both methods."""
    x, difference = np.asarray(x), np.asarray(difference)
    rng = np.random.default_rng(20260914)
    indices = rng.integers(0, len(x), size=(repeats, len(x)))
    bx, by = x[indices], difference[indices]
    bx = bx - bx.mean(axis=1, keepdims=True)
    by = by - by.mean(axis=1, keepdims=True)
    denominator = (bx ** 2).sum(axis=1)
    valid = denominator > 0
    if not valid.any():
        raise ValueError("Search count has no variation")
    values = (bx * by).sum(axis=1)[valid] / denominator[valid]
    return np.quantile(values, [0.025, 0.975]).tolist()


def search_axis(component, count):
    return math.log2(count)


def summarize_trends(rows, groups):
    output = []
    for judge in JUDGES:
        for component, metric in METRIC_PAIRS:
            pairs = {}
            for row in rows:
                if row["judge"] == judge and row["component"] == component:
                    pairs.setdefault(row["input_id"], {})[row["method"]] = row
            x, ebg, baseline = [], [], []
            for pair in pairs.values():
                b, a = pair["EBG"], pair["baseline"]
                if b["search_count"] != a["search_count"]:
                    raise ValueError("Mismatched paired workload")
                x.append(search_axis(component, b["search_count"]))
                ebg.append(b["metrics"][metric])
                baseline.append(a["metrics"][metric])
            difference = [b - a for b, a in zip(ebg, baseline)]
            selected = {g["bin"]: g for g in groups if g["judge"] == judge
                        and g["component"] == component and g["metric"] == metric}
            output.append({
                "model": rows[0]["model"], "judge": judge, "component": component,
                "metric": metric, "n_pairs": len(x),
                "slope_unit": "per_workload_token_doubling",
                "EBG_slope": slope(x, ebg), "baseline_slope": slope(x, baseline),
                "slope_difference": slope(x, difference),
                "slope_difference_ci95": slope_interval(x, difference),
                "mean_difference": statistics.mean(difference),
                "bin_differences": {label: selected[label]["difference"]
                                    for label in ("Low", "Medium", "High")},
            })
    return output


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def export_metric_table(trends):
    records = []
    for trend in trends:
        records.append({
            "model": MODELS[trend["model"]], "judge": trend["judge"],
            "benchmark": trend["component"], "metric": METRICS[trend["component"]][trend["metric"]],
            "n_pairs": trend["n_pairs"], "slope_unit": trend["slope_unit"], "EBG_slope": trend["EBG_slope"],
            "baseline_slope": trend["baseline_slope"], "slope_difference": trend["slope_difference"],
            "ci95_low": trend["slope_difference_ci95"][0], "ci95_high": trend["slope_difference_ci95"][1],
            "mean_difference": trend["mean_difference"], **trend["bin_differences"],
        })
    with (OUTPUT / "metric_comparison.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def render_report(trends):
    available = {t['model'] for t in trends}
    lines = ['# Visible input size and method performance', '', 'Group by original visible-input tokens, not Gold files or evidence counts.', '', '- SpecGAP and SilentSwap: task-document tokens plus readable repository content.', '- FeedbackTrace: assistant_response and tool_exchange content in the visible trace, excluding user and system messages.', '- Count segments separately with o200k_base and sum them. These are input sizes, not cumulative API usage.', '- Sort the 100 inputs in each benchmark by token count then input_id; use Low/Medium/High groups of 33/33/34. Ties are ordered by input_id.', '- Models: ' + '、'.join((MODELS[m] for m in MODELS if m in available)) + '; judges are reported separately.', '', '| Benchmark | Group | n | Min tokens | Median tokens | Max tokens |', '|---|---|---:|---:|---:|---:|']
    for g in read(OUTPUT / 'bin_summary.json'):
        lines.append(f'| {g['component']} | {g['bin']} | {g['n']} | {g['min_count']} | {g['median_count']} | {g['max_count']} |')
    lines += ['', '## Relative gain', '', 'Relative gain = (EBG group mean - Base group mean) / Base group mean * 100%.', 'See [relative gains](RELATIVE_GAIN.md) for percentages. Original means and absolute differences are retained. A zero Base mean produces NA.', 'Group percentage changes are descriptive; their trend significance was not tested. Confidence intervals for absolute-difference slopes do not apply.', '', '## Supporting analysis', '', '- Group plots show score medians with 95% intervals from 2,000 paired sample bootstrap replicates. Group means are in results/<model>/group_scores.json.', '- Absolute score differences use all samples and regress on log2(tokens). The slope difference is the EBG-minus-Base change per doubling; intervals are in results/metric_comparison.csv. No multiple-comparison correction is applied.', '- All methods share the same inputs and reuse saved per-sample scores. SilentSwap stage and revision distinctions remain applicable; see docs/benchmark_results.md in the repository.', '- Natural associations between input size and performance do not establish a causal effect of difficulty.', '', '## Reproduction', '', '```powershell', 'python experiments/input_size/scripts/count_workload.py', 'python experiments/input_size/scripts/compare_models.py --models kimi-k3 glm-5-3 luna --resume', 'python experiments/input_size/scripts/relative_gain.py', 'python experiments/input_size/scripts/plot_results.py', '```', '', 'Counts resume per input. Use --recount when source inputs change. Group changes invalidate summary caches; omit --resume when score sources change.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=tuple(MODELS), default=list(MODELS),
                        help="Models to summarize; preserves the same frozen groups")
    parser.add_argument("--resume", action="store_true",
                        help="Reuse completed model outputs from this run; omit after source changes")
    args = parser.parse_args()
    workloads = read(ROOT / "outputs/analysis/input_size/data" / "search_samples.json")
    for component in PRIMARY:
        samples = [s for s in workloads if s["component"] == component]
        if len(samples) != 100 or len({s["input_id"] for s in samples}) != 100:
            raise ValueError(f"Expected 100 unique samples: {component}")
    all_trends = []
    for model in dict.fromkeys(args.models):
        completed = OUTPUT / model / "trends.json"
        manifest = OUTPUT / model / "analysis_inputs.json"
        reusable = (args.resume and manifest.exists() and read(manifest) == workloads)
        if reusable and completed.exists():
            cached = read(completed)
            expected = {(judge, component, metric) for judge in JUDGES
                        for component, metric in METRIC_PAIRS}
            if len(cached) == len(expected) and {
                    (t["judge"], t["component"], t["metric"]) for t in cached} == expected:
                all_trends.extend(cached)
                print(f"{MODELS[model]}: reused completed output", flush=True)
                continue
        samples_path = OUTPUT / model / "sample_scores.json"
        if reusable and samples_path.exists():
            rows = read(samples_path)
        else:
            rows = []
            for judge in JUDGES:
                for sample in workloads:
                    for method in ("EBG", "baseline"):
                        scores, source = load_scores(sample["component"], method, judge,
                                                     sample["input_id"], model=model)
                        rows.append({"model": model, "judge": judge,
                                     "component": sample["component"], "input_id": sample["input_id"],
                                     "bin": sample["bin"], "bin_label": sample["bin_label"],
                                     "count_metric": sample["count_metric"], "search_count": sample["search_count"],
                                     "method": method, "metrics": scores, "source": source})
        groups = aggregate(rows)
        trends = summarize_trends(rows, groups)
        save(OUTPUT / model / "sample_scores.json", rows)
        save(OUTPUT / model / "group_scores.json", groups)
        save(OUTPUT / model / "group_medians.json", group_medians(rows))
        save(manifest, workloads)
        save(OUTPUT / model / "trends.json", trends)
        all_trends.extend(trends)
        print(f"{MODELS[model]}: {len(rows)} scores", flush=True)
    save(OUTPUT / "trends.json", all_trends)
    medians = []
    for model in dict.fromkeys(args.models):
        path = OUTPUT / model / "group_medians.json"
        if path.exists():
            records = read(path)
        else:
            records = group_medians(read(OUTPUT / model / "sample_scores.json"))
            save(path, records)
        medians.extend(records)
        print(f"{MODELS[model]}: grouped medians ready", flush=True)
    save(OUTPUT / "group_medians.json", medians)
    export_metric_table(all_trends)
    (ROOT / "outputs/analysis/input_size/REPORT.md").write_text(render_report(all_trends), encoding="utf-8")


if __name__ == "__main__":
    main()
