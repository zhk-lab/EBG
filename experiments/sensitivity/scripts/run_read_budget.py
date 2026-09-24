"""Run Luna SilentSwap h3 with a 65,536-token read budget, then score."""
import run


def main():
    results = run.ROOT / "outputs/analysis/sensitivity/results"
    common = ["--config", str(run.ROOT / "outputs/analysis/sensitivity/data/luna_silentswap_h3_budget65536.json"),
              "--results", str(results), "--experiment", "expansion_hops",
              "--model", "luna", "--benchmark", "silentswap", "--arm", "graph",
              "--workers", "25"]
    prediction_exit = run.main(["predict", *common])
    judge_exit = run.main(["judge", *common])
    status = {"status": "complete" if not (prediction_exit or judge_exit) else "incomplete",
              "prediction_exit": prediction_exit, "judge_exit": judge_exit}
    run.write_json(results / "luna_silentswap_h3_status.json", status)
    print(status, flush=True)
    return int(bool(prediction_exit or judge_exit))


if __name__ == "__main__":
    raise SystemExit(main())
