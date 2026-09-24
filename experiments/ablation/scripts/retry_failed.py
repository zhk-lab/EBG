"""Retry incomplete experiment-one batches while retaining failed attempts."""

import json
import io
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
from threading import Lock

import pipeline

pipeline.load_trace_runtime()
import run


def stream_opener(request, timeout):
    """Receive long predictions incrementally while preserving the provider response."""
    body = json.loads(request.data.decode("utf-8"))
    body.update(stream=True, stream_options={"include_usage": True})
    streamed = urllib.request.Request(request.full_url, data=json.dumps(body).encode("utf-8"),
                                      headers=dict(request.header_items()), method="POST")
    content, reasoning, usage = [], [], {}
    finish = model = response_id = None
    with urllib.request.urlopen(streamed, timeout=timeout) as response:
        if "text/event-stream" not in response.headers.get("Content-Type", ""):
            return io.BytesIO(response.read())
        for line in response:
            value = line.decode("utf-8").strip()
            if not value.startswith("data:"):
                continue
            value = value[5:].strip()
            if value == "[DONE]":
                break
            chunk = json.loads(value)
            if chunk.get("error"):
                raise OSError(str(chunk["error"]))
            model = chunk.get("model") or model
            response_id = chunk.get("id") or response_id
            usage = chunk.get("usage") or usage
            for choice in chunk.get("choices", []):
                if choice.get("index", 0) != 0:
                    continue
                delta = choice.get("delta", {})
                content.append(delta.get("content") or "")
                reasoning.append(delta.get("reasoning_content") or "")
                finish = choice.get("finish_reason") or finish
    if not finish:
        raise OSError("Stream ended without a finish reason")
    result = {"id": response_id, "model": model, "usage": usage, "_transport": "streaming",
              "choices": [{"index": 0, "finish_reason": finish,
                           "message": {"role": "assistant", "content": "".join(content),
                                       "reasoning_content": "".join(reasoning)}}]}
    return io.BytesIO(json.dumps(result).encode("utf-8"))


def archive_failure(case, filename, results):
    record_path = case / filename
    if not record_path.exists():
        return None
    record = run.read_json(record_path)
    if record.get("status") != "failed":
        return None
    source = case.resolve()
    history = results / "retry_history" / source.relative_to(results.resolve())
    number = 1
    while (history / str(number)).exists():
        number += 1
    target = (history / str(number)).resolve()
    if not source.is_relative_to(results.resolve()) or not target.is_relative_to(results.resolve()):
        raise ValueError("Retry history must remain inside results")
    target.parent.mkdir(parents=True, exist_ok=True)
    source.rename(target)
    return record.get("actual_usage", record.get("usage", {}))


def reconcile_prediction_usage(case, results):
    """Recover attempt totals from saved calls, including interrupted retry batches."""
    record_path = case / "batch_result.json"
    history = results / "retry_history" / case.resolve().relative_to(results.resolve())
    if not record_path.exists() or not history.exists():
        return
    call_roots = {folder.parent for base in (case, history)
                  for name in ("responses", "provider_failures")
                  for folder in base.rglob(name) if folder.is_dir()}
    if not call_roots:
        return
    totals = dict(calls=0, input_tokens=0, output_tokens=0, total_tokens=0)
    for root in call_roots:
        usage = run.predict.collect_response_usage(root)
        for key in totals:
            totals[key] += usage[key]
    record = run.read_json(record_path)
    record["actual_usage"] = totals
    run.write_json(record_path, record)


def main():
    original_client = run.predict._client
    def streaming_client(args):
        client = original_client(args)
        client.opener = stream_opener
        return client
    run.predict._client = streaming_client
    results = run.ROOT / "outputs/analysis/ablation/results"
    config = run.read_json(run.HERE / "configs/config.json")
    samples = run.read_json(run.HERE / "configs/samples.json")
    state_path = results / "experiment_one_only_status.json"
    state = {"status": "running", "pid": os.getpid(), "variant": "no_behavior",
             "stop_after": "no_behavior", "models": ["kimi-k3", "glm-5-3"],
             "started_at": datetime.now().astimezone().isoformat(), "finished_batches": []}
    lock = Lock()
    run.write_json(state_path, state)

    def process(model):
        for benchmark in run.BENCHMARKS:
            roots = [run.stage_root(results, model, benchmark, "no_behavior", stage)
                     for stage in (("stage1", "stage2") if benchmark == "silentswap" else ("stage1",))]
            for recovery in (1, 2):
                ids = set()
                carried = {}
                for root in roots:
                    for ident in samples[benchmark]:
                        case = root / "runs" / ident
                        path = case / "batch_result.json"
                        if (path.exists() and run.read_json(path).get("status") == "complete"
                                and (case / "prediction.json").exists()):
                            continue
                        ids.add(ident)
                        usage = archive_failure(case, "batch_result.json", results)
                        if usage is not None:
                            carried[path] = usage
                if not ids:
                    break
                print(model, benchmark, "retry", recovery, "samples", len(ids), flush=True)
                run.main(["predict", "--model", model, "--benchmark", benchmark,
                          "--variant", "no_behavior", "--workers", "32", "--ids", *sorted(ids)])
                for path, usage in carried.items():
                    if path.exists():
                        record = run.read_json(path)
                        actual = record.get("actual_usage", record.get("usage", {}))
                        record["actual_usage"] = {key: actual.get(key, 0) + usage.get(key, 0)
                                                  for key in set(actual) | set(usage)}
                        run.write_json(path, record)
            judge_model = config["judges"][benchmark]["model"]
            carried = {}
            for root in roots:
                for ident in samples[benchmark]:
                    case = root / "judges" / judge_model / ident
                    usage = archive_failure(case, "status.json", results)
                    if usage is not None:
                        carried[case / "status.json"] = usage
            code = run.main(["judge", "--model", model, "--benchmark", benchmark,
                             "--variant", "no_behavior", "--workers", "32"])
            for path, usage in carried.items():
                if path.exists():
                    record = run.read_json(path)
                    record["previous_attempt_usage"] = usage
                    run.write_json(path, record)
            for root in roots:
                for ident in samples[benchmark]:
                    reconcile_prediction_usage(root / "runs" / ident, results)
            with lock:
                state["finished_batches"].append({"model": model, "benchmark": benchmark,
                                                 "status": "complete" if not code else "needs_attention"})
                state["updated_at"] = datetime.now().astimezone().isoformat()
                run.write_json(state_path, state)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(process, state["models"]))
    state["status"] = "paused_after_experiment_one"
    state["updated_at"] = datetime.now().astimezone().isoformat()
    run.write_json(state_path, state)
    print(json.dumps(state, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
