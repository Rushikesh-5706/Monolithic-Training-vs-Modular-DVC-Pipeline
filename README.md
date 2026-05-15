# Monolithic Training vs Modular DVC Pipeline

A side-by-side implementation and benchmark of two machine learning workflow
architectures using the UCI Adult Income dataset. One is a single sequential
script that does everything from data loading to model saving. The other is a
four-stage modular pipeline managed by DVC, where each stage is independently
cached and reproducible.

The goal is not a state-of-the-art model. The goal is to demonstrate, with
measured timing data, exactly when and why a structured pipeline is worth the
setup cost.

---

## Project Structure

```
.
├── data/
│   ├── adult.csv              -- Raw UCI Adult Income dataset (DVC-tracked)
│   └── adult.csv.dvc          -- DVC pointer file (committed to Git)
├── src/
│   ├── prepare.py             -- Stage 1: load raw data and remove missing rows
│   ├── featurize.py           -- Stage 2: encode categoricals, split train/test
│   ├── train.py               -- Stage 3: fit RandomForestClassifier
│   └── evaluate.py            -- Stage 4: compute accuracy, AUC, F1 and write JSON
├── models/
│   └── model.joblib           -- Trained model artifact (DVC output)
├── metrics/
│   └── scores.json            -- Pipeline evaluation metrics (DVC metrics file)
├── train_monolithic.py        -- Monolithic baseline: all stages in one script
├── dvc.yaml                   -- DVC pipeline definition (DAG)
├── params.yaml                -- Centralized hyperparameters
├── test_caching.sh            -- Script that verifies DVC stage caching behavior
├── benchmark.md               -- Measured performance comparison
├── Dockerfile                 -- Container definition
├── docker-compose.yml         -- Service orchestration
├── requirements.txt           -- Python dependencies
└── .env.example               -- Environment variable documentation
```

---

## Pipeline Architecture

The DVC pipeline forms a directed acyclic graph (DAG) with four stages.
Each stage declares its dependencies explicitly. DVC re-runs a stage only
when its declared inputs change.

```
data/adult.csv
      |
      v
  [prepare]  ---- src/prepare.py
      |
      v
 data/processed.csv
      |
      v
 [featurize] ---- src/featurize.py
      |            params.yaml (prepare.*)
      v
 data/features.npz
      |
      +-----------> [train] ---- src/train.py
      |                 |         params.yaml (train.*)
      |                 v
      |         models/model.joblib
      |                 |
      +-----------------+
                        |
                        v
                  [evaluate] ---- src/evaluate.py
                        |
                        v
               metrics/scores.json
```

When `train.n_estimators` changes in `params.yaml`, only the `train` and
`evaluate` stages re-run. The `prepare` and `featurize` stages are served
from DVC's content-addressed cache.

---

## Prerequisites

- Docker and Docker Compose
- Git

No local Python installation is required to run the containerized workflow.

---

## Setup

Clone the repository:

```bash
git clone https://github.com/Rushikesh-5706/Monolithic-Training-vs-Modular-DVC-Pipeline.git
cd Monolithic-Training-vs-Modular-DVC-Pipeline
```

Build the image and start the service:

```bash
docker-compose up --build -d
```

The build step downloads the dataset, runs the full pipeline, and caches
all stage outputs inside the image. The first build takes several minutes.

Confirm the service is running:

```bash
docker-compose ps
```

---

## Running the Monolithic Script

```bash
docker-compose run --rm app python train_monolithic.py
```

Outputs written to:
- `model.joblib` -- serialized model
- `metrics.json` -- accuracy, AUC, F1 as floating-point values

---

## Running the DVC Pipeline

```bash
docker-compose run --rm app dvc repro
```

Outputs written to:
- `models/model.joblib` -- serialized model
- `metrics/scores.json` -- accuracy, AUC, F1 as floating-point values

To run with a different hyperparameter without modifying any file:

```bash
docker-compose run --rm app dvc repro --set-param train.n_estimators=200
```

---

## Experiment Tracking

Run a named experiment with a parameter override:

```bash
docker-compose run --rm app dvc exp run --set-param train.n_estimators=150 --name n150
```

View all experiments in a comparison table:

```bash
docker-compose run --rm app dvc exp show
```

View as JSON (useful for scripted analysis):

```bash
docker-compose run --rm app dvc exp show --json
```

---

## DVC Caching Verification

Run the caching verification script to confirm that only affected stages
re-execute after a hyperparameter change:

```bash
docker-compose run --rm app bash test_caching.sh
```

Inspect the output log:

```bash
docker-compose run --rm app cat repro_log.txt
```

Expected: `Stage 'prepare' didn't change, skipping` and
`Stage 'featurize' didn't change, skipping` appear alongside
`Running stage 'train'` and `Running stage 'evaluate'`.

---

## View Metrics

```bash
docker-compose run --rm app dvc metrics show
```

---

## Reproducibility

To reproduce an earlier experiment exactly:

```bash
git checkout <commit-hash>
docker-compose run --rm app dvc checkout
docker-compose run --rm app dvc repro
```

DVC links the correct data and model versions to each Git commit. The entire
state of the workspace — code, parameters, data, and model — is pinned to
that commit.

---

## Benchmark Summary

Full timing results and analysis are in [benchmark.md](benchmark.md).

| Metric | Monolithic | DVC Pipeline |
|---|---|---|
| Full run time (s) | 6.05 | 5.42 |
| Re-run time after param change (s) | 1.87 | 2.85 |
| Iteration speedup | 1.00x | 1.9x |

---

## Local Development (without Docker)

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
dvc repro
```

---

## License

MIT
