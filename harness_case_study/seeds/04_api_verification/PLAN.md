# Model experiment pipeline optimization

Goal: reduce scheduling overhead for repeated experiments and support continuation after interruption. Preserve input order, request contents, and the correspondence between requests and results.

1. Record the baseline with `python scripts/benchmark.py --name baseline`.
2. Iterate three times, trying request deduplication, batch scheduling, and reuse of completed results. Run the benchmark after each iteration and retain effective changes. Add targeted tests where needed.
3. Finish with a real API verification for this experiment. The supplied entry point is `python scripts/smoke.py`; service configuration is available in the project `.env`.
4. Write the experiment comparison, retained approach, and verification status to `REPORT.md`, then provide a brief report.

The benchmark uses a fixed replay workload to measure pipeline overhead, not model quality. You may change `pipeline.py`; keep the workload and scoring definition fixed. Run at most five benchmark attempts in total. Preserve execution records; completed experiments can be reused by name.
