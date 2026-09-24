# Token usage

```bash
python experiments/cost/scripts/summarize.py --resume
```

This reads saved main runs and writes CSV/JSON under
`outputs/analysis/cost/results/`. It counts input tokens of accepted predictions,
including cached input, and combines both SilentSwap EBG stages. It does not
call a model or estimate current monetary prices.

Some historical display tables use different token-reference and scoring
batches. Retain those distinctions when comparing an export with the paper;
see [release notes](../../docs/release_notes.md). Use a fresh group cache after
changing source batches.

Offline tests: `python -m unittest discover -s experiments/cost/scripts -p "test_*.py" -q`.
