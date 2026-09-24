# Small classifier research

Python 3.11+, standard library only. The model is a distance-weighted neighbor
classifier whose training stage builds a reference collection. Features and data
are synthetic.

```bash
python run.py --config configs/baseline.json --output runs/baseline.json
python run.py --config configs/candidate_1.json --output runs/candidate_1.json
```

Compare the three candidates and save the selected configuration as `selected.json`.
`run.py` saves metrics and actual sample IDs and prints a short summary. Split
assignments are in `data/catalog.csv`; `data/extra.csv` is an optional support
pool with independent identifiers. Export metadata is in `data/export.json`.
Preserve the original data files for inspection.
