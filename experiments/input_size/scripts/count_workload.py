"""Count visible input tokens and restore the original within-benchmark tertiles."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import tiktoken
import statistics

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parents[1]
COMPONENTS = ("specgap", "silentswap", "feedbacktrace")
ANALYSIS_VERSION = "token_workload_v2"
ENCODING = "o200k_base"
sys.path.insert(0, str(ROOT / "src"))
from ebg.evidence_intake import load_visible_bundle


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def measure_bundle(bundle, count):
    document = count(bundle.task_document.content) if bundle.task_document else 0
    workspace = sum(count(artifact.content) for artifact in bundle.repo_artifacts)
    trace = sum(count(event.content) for event in bundle.trace_events
                if event.event_type in {"assistant_response", "tool_exchange"})
    return {"search_count": trace if bundle.benchmark == "feedbacktrace" else document + workspace,
            "count_metric": "workload_tokens", "document_tokens": document,
            "workspace_tokens": workspace, "trace_output_tokens": trace,
            "tokenizer": ENCODING}


def measure(component, input_id):
    source = ROOT / "data" / "prepared" / component / "artifacts" / "visible_bundles" / input_id
    bundle = load_visible_bundle(source)
    encoding = tiktoken.get_encoding(ENCODING)
    return {"analysis_version": ANALYSIS_VERSION, "component": component, "input_id": input_id,
            "count_source": source.relative_to(ROOT).as_posix(),
            **measure_bundle(bundle, lambda text: len(encoding.encode_ordinary(text)))}


def assign_bins(rows):
    if not rows or len({r["component"] for r in rows}) != 1:
        raise ValueError("Expected one nonempty component")
    if any(not isinstance(r["search_count"], int) or r["search_count"] <= 0 for r in rows):
        raise ValueError("Counts must be positive integers")
    ordered = sorted(rows, key=lambda r: (r["search_count"], r["input_id"]))
    n = len(ordered)
    for i, row in enumerate(ordered):
        label = "Low" if i < n // 3 else "Medium" if i < 2 * n // 3 else "High"
        row.update(rank=i + 1, bin=label, bin_label=label)
    return ordered


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recount", action="store_true", help="Refresh counts after source changes")
    args = parser.parse_args()
    all_rows, summaries, distributions = [], [], {}
    for component in COMPONENTS:
        bundle_root = ROOT / "data" / "prepared" / component / "artifacts" / "visible_bundles"
        ids = sorted(path.name for path in bundle_root.iterdir() if path.is_dir())
        if len(ids) != 100:
            raise ValueError(f"Expected 100 main samples: {component}")
        rows = []
        for input_id in ids:
            cache = ROOT / "outputs/analysis/input_size/data" / "samples" / ANALYSIS_VERSION / component / f"{input_id}.json"
            if cache.exists() and not args.recount:
                row = read(cache)
                if (row["analysis_version"], row["input_id"], row["component"]) != (
                        ANALYSIS_VERSION, input_id, component):
                    raise ValueError(f"Cache mismatch: {cache}; use --recount")
            else:
                row = measure(component, input_id)
                write_json(cache, row)
            rows.append(row)
        rows = assign_bins(rows)
        all_rows.extend(rows)
        distributions[component] = dict(sorted(Counter(r["search_count"] for r in rows).items()))
        for label in ("Low", "Medium", "High"):
            group = [r for r in rows if r["bin"] == label]
            if not group:
                raise ValueError(f"Empty group: {component}/{label}")
            values = [r["search_count"] for r in group]
            summaries.append({"component": component, "bin": label, "bin_label": group[0]["bin_label"],
                              "count_metric": group[0]["count_metric"], "n": len(group),
                              "min_count": min(values), "median_count": statistics.median(values),
                              "max_count": max(values)})
        print(f"{component}: counted {len(rows)} samples", flush=True)
    write_json(ROOT / "outputs/analysis/input_size/data" / "search_samples.json", all_rows)
    write_json(ROOT / "outputs/analysis/input_size/data" / "count_distributions.json", distributions)
    write_json(ROOT / "outputs/analysis/input_size/results" / "bin_summary.json", summaries)


if __name__ == "__main__":
    main()
