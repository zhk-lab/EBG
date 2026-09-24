"""Overlap one variant's judging with the next variant's prediction."""

import os
import subprocess
import sys
import types
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
from threading import Lock


def load_trace_runtime():
    """Load the published TraceReview package from this checkout."""
    root = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(root), str(root / "src")]
    import tracereview


def execute(predict, judge, benchmarks, variants, supported):
    """Keep benchmark barriers and a single judging batch active at a time."""
    with ThreadPoolExecutor(max_workers=1) as scoring:
        for benchmark in benchmarks:
            pending = []
            for variant in variants:
                if not supported(benchmark, variant):
                    continue
                for future in pending:
                    if future.done():
                        future.result()
                predict(benchmark, variant)
                pending.append(scoring.submit(judge, benchmark, variant))
            for future in pending:
                future.result()


def main():
    load_trace_runtime()
    import run
    import analyze

    path = run.ROOT / "outputs/analysis/ablation/results/workflow_status.json"
    state = run.read_json(path) if path.exists() else {}
    state.update(status="running", evaluation_pid=os.getpid(), workers=32,
                 prediction_workers=32, judge_workers=32,
                 execution_order="sequential predictions with overlapping sequential judgments",
                 active_prediction=None, active_judge=None, completed_phases=[])
    state["started_at"] = datetime.now().astimezone().isoformat()
    for key in ("active_benchmark", "active_variant", "active_phase", "failure"):
        state.pop(key, None)
    lock = Lock()

    def save():
        state["updated_at"] = datetime.now().astimezone().isoformat()
        run.write_json(path, state)

    def phase(command, benchmark, variant):
        slot = "active_prediction" if command == "predict" else "active_judge"
        item = dict(benchmark=benchmark, variant=variant, phase=command)
        with lock:
            state[slot] = item
            save()
        result = run.main([command, "--model", "luna", "--benchmark", benchmark,
                           "--variant", variant, "--workers", "32"])
        if result:
            raise RuntimeError(f"{benchmark}/{variant}/{command}: unfinished samples")
        with lock:
            state[slot] = None
            state["completed_phases"].append(item)
            save()

    save()
    try:
        execute(lambda b, v: phase("predict", b, v),
                lambda b, v: phase("judge", b, v),
                ["silentswap", "feedbacktrace"],
                ["no_behavior", "no_graph", "no_task"], run.supported)
        analyze.main(["--model", "kimi-k3", "luna", "glm-5-3",
                      "--variant", "no_behavior", "no_graph", "no_task"])
    except BaseException as error:
        state.update(status="needs_attention", failure=f"{type(error).__name__}: {error}")
        save()
        raise
    state.update(status="complete", completed_benchmarks=["silentswap", "feedbacktrace"])
    save()


if __name__ == "__main__":
    main()
