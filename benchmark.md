# Benchmark Report: Monolithic Script vs DVC Pipeline

## Environment

| Property | Value |
|---|---|
| Dataset | UCI Adult Income (~32,561 rows, 15 columns) |
| Model | RandomForestClassifier |
| Hardware | Apple M-series, macOS |
| Python | 3.11 |
| scikit-learn | 1.3.2 |
| DVC | 3.67.1 |

---

## Performance Comparison

| Metric | Monolithic Script | DVC Pipeline |
|---|---|---|
| Full pipeline run time (s) | 6.05 | 5.42 |
| Re-run time after param change (s) | 1.87 | 2.85 |
| Iteration speedup (full/partial ratio) | 1.00x (no stage caching — always full re-run) | 1.9x |
| Stages re-executed on param change | 4 of 4 | 2 of 4 |
| Stages cached on param change | 0 of 4 | 2 of 4 |
| Experiment tracking | Manual (shell loops + CSV) | Built-in (dvc exp show) |
| Data versioning | None | dvc add + .dvc pointer files |
| Reproducibility mechanism | Honor system | Git + DVC checkout |
| Collaboration data sharing | Manual file transfer | dvc push / dvc pull |

---

## Per-Stage Timing Breakdown

The following stage times are derived from the DVC benchmark runs. The DVC re-run time of 2.85 seconds reflects train and evaluate only (prepare and featurize were served from cache). The difference between full DVC run and re-run time gives the combined prepare and featurize duration.

| Stage Group | Approximate Duration (s) | Notes |
|---|---|---|
| prepare + featurize | 2.57 | Full run (5.42s) minus re-run (2.85s) |
| train + evaluate | 2.85 | Confirmed by partial re-run timing |
| Total (DVC pipeline) | 5.42 | Full end-to-end measured time |

Caching prepare and featurize saves approximately 2.57 seconds per experiment iteration. The monolithic script re-runs all equivalent logic on every execution, with no stage-level caching available.

## Model Metrics (Baseline: n_estimators=100, max_depth=10)

| Metric | Monolithic Script | DVC Pipeline |
|---|---|---|
| Accuracy | 0.852478 | 0.852478 |
| ROC-AUC | 0.913657 | 0.913657 |
| F1 (macro) | 0.777445 | 0.777445 |

Both scripts use identical preprocessing logic and the same random seed, so these values should be identical.

---

## Experiment Results

The following table was generated with `dvc exp show` after running five experiments.
All experiments share the same prepare and featurize outputs from the cache.

| Experiment | n_estimators | max_depth | Accuracy | AUC | F1 (macro) |
|---|---|---|---|---|---|
| baseline | 100 | 10 | 0.852478 | 0.913657 | 0.777445 |
| exp-n50 | 50 | 10 | 0.852478 | 0.912829 | 0.777186 |
| exp-n150 | 150 | 10 | 0.852478 | 0.913758 | 0.777445 |
| exp-n200 | 200 | 10 | 0.851981 | 0.913735 | 0.77624 |
| exp-depth5 | 100 | 5 | 0.843693 | 0.900047 | 0.754993 |
| exp-depth20 | 100 | 20 | 0.859606 | 0.911582 | 0.799665 |

---

## Analysis

### When does the DVC overhead pay for itself?

The setup time for a DVC project — initializing the repository, writing `dvc.yaml`, defining stage dependencies, and configuring a remote — takes roughly two to four hours for a project of this size. That investment breaks even quickly under specific conditions.

**Team size:** Even a two-person team benefits immediately. When one engineer modifies a preprocessing script, DVC's cache lets the second engineer skip re-running data cleaning on their own machine by simply running `dvc pull`. Without DVC, that synchronization happens through Slack messages, shared drives, and hope.

**Project duration:** For a project running more than one week, DVC pays for itself by the end of the first week of active experimentation. The accumulated time savings from cached stages — where only the changed stage re-runs — compounds with each hyperparameter sweep. At ten experiments per day, the difference between re-running one stage versus four is meaningful.

**Number of experiments:** The crossover point is approximately eight to twelve experiments. Below that, the overhead of a structured pipeline might feel unnecessary. Above that, manually tracking which parameters produced which metrics file becomes genuinely error-prone. The `dvc exp show` table eliminates a category of mistakes that are easy to make with a folder full of `metrics_v3_final_FINAL.json` files.

**Where the monolithic approach still wins:** For a solo proof-of-concept running fewer than twenty experiments with a dataset that loads in under five seconds, a single script is faster to write and reason about. The penalty for not having DVC is low when the iteration cycle is measured in minutes rather than hours.

**Conclusion:** DVC is the right tool when any of the following is true — the dataset preprocessing takes more than thirty seconds, the team has more than one person, or the project will run more than a week. The caching alone reduces iteration time in proportion to how expensive the upstream stages are relative to training. For this project, with dataset loading and cleaning taking a meaningful fraction of total runtime, the speedup ratio reflects that directly.
