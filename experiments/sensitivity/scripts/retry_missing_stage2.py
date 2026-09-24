"""Retry the two missing stage2 predictions and judge successful outputs."""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
from experiments.sensitivity.scripts import run


def retry_one(condition, input_id):
    base = ROOT / "experiments/sensitivity"
    config = run.read_json(base / "configs/qwen_resume.json")
    stage = base / "results/conditions/glm-5-3/silentswap" / condition
    sample = stage / "stage2/runs" / input_id
    if not (sample / "prediction.json").exists() and sample.exists():
        saved = run.read_json(sample / "batch_result.json")
        if saved["status"] != "failed":
            raise RuntimeError(f"Expected a failed prediction: {sample}")
        target = base / "results/prediction_retry_backups" / datetime.now().strftime("%Y%m%d_%H%M%S") / condition / input_id
        source = sample.resolve()
        target = target.resolve()
        if not source.is_relative_to((base / "results/conditions").resolve()) or not target.is_relative_to((base / "results/prediction_retry_backups").resolve()):
            raise RuntimeError("Invalid backup path")
        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)
    print(json.dumps({"event": "prediction_started", "condition": condition, "input_id": input_id}), flush=True)
    result = run.stage2_sample(config, "glm-5-3", input_id, stage / "stage1", stage / "stage2")
    print(json.dumps({"event": "prediction_finished", "condition": condition, **result}), flush=True)
    if result["status"] != "complete":
        return False
    profile = config["judges"]["silentswap"]
    batch = run.judge.JudgeBatchConfig(experiment_name="stage2", experiment_root=stage, phase="full", benchmarks=("silentswap",), arms=("graph",), judge_model=profile["model"], base_url=profile["base_url"], api_key_env=profile["api_key_env"], request_options=profile["request_options"], workers=1, timeout=config["timeout"])
    result = run.judge_sample(batch, "silentswap", "graph", input_id, run.judge._load_judge_module("silentswap"), run.judge._default_client_factory(batch))
    print(json.dumps({"event": "judge_finished", "condition": condition, **result}), flush=True)
    return result["status"] == "complete"


if __name__ == "__main__":
    load_dotenv(ROOT / ".env", override=False)
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(retry_one, condition, sample) for condition, sample in (("graph_h3_r6", "ss_039"), ("graph_h1_r5", "ss_089"))]
        success = [job.result() for job in jobs]
    raise SystemExit(0 if all(success) else 1)
