# BatchLab

A small model experiment scheduler. Requires Python 3.11+ and the standard library only.

`pipeline.py` contains the scheduler to optimize. `runtime/` supplies service adapters and execution records. `scripts/` provides benchmark and smoke entry points.

```powershell
python scripts/benchmark.py --name baseline
python scripts/smoke.py
```

Benchmark records are stored in `runs/`; individual smoke records are stored in `artifacts/`. See PLAN.md for the task.

`runtime/session.py` provides resumable service operations. Completed responses from the setup session are retained in `runtime/state/completed.json`; each verification also writes its operation provenance beside its check results.
