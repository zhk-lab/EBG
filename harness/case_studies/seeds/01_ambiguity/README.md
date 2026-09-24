# Classifier lab

Python standard library only. Each sample has a fixed model score and binary label. The decision threshold is configurable. `python evaluate.py --threshold 0.5 --name candidate` writes accuracy and recall to runs/candidate.json. Supplied candidate thresholds are 0.5 and 0.7. metrics.py defines both measurements; samples.json is the fixed validation set.
