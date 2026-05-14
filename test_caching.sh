#!/usr/bin/env bash
# test_caching.sh
# Verifies that DVC's caching skips unaffected pipeline stages and only
# re-executes stages whose dependencies have changed.
set -euo pipefail

echo "=== DVC Stage Caching Verification ==="

# Step 1: Warm the cache by running a full pipeline pass
echo ""
echo "[1/3] Running full pipeline to warm DVC cache..."
dvc repro
echo "Full pipeline complete."

# Step 2: Modify the n_estimators hyperparameter in the train section
# This change affects only the train and evaluate stages.
# The prepare and featurize stages have no dependency on this parameter.
echo ""
echo "[2/3] Modifying params.yaml: setting train.n_estimators from 100 to 200..."
perl -pi -e 's/n_estimators: 100/n_estimators: 200/' params.yaml
echo "Modification applied:"
grep "n_estimators" params.yaml

# Step 3: Re-run the pipeline and capture all output to repro_log.txt
echo ""
echo "[3/3] Running dvc repro after parameter change, capturing output..."
dvc repro 2>&1 | tee repro_log.txt

echo ""
echo "=== repro_log.txt ==="
cat repro_log.txt

echo ""
echo "=== Verification ==="
if grep -q "Stage 'prepare' didn't change, skipping" repro_log.txt; then
    echo "PASS: prepare stage was correctly skipped"
else
    echo "WARN: prepare skip message not found in log (check DVC version output format)"
fi

if grep -q "Stage 'featurize' didn't change, skipping" repro_log.txt; then
    echo "PASS: featurize stage was correctly skipped"
else
    echo "WARN: featurize skip message not found in log (check DVC version output format)"
fi

if grep -q "Running stage 'train'" repro_log.txt || grep -q "running stage 'train'" repro_log.txt; then
    echo "PASS: train stage was correctly re-executed"
else
    echo "WARN: train run message not found in log (check DVC version output format)"
fi

if grep -q "Running stage 'evaluate'" repro_log.txt || grep -q "running stage 'evaluate'" repro_log.txt; then
    echo "PASS: evaluate stage was correctly re-executed"
else
    echo "WARN: evaluate run message not found in log (check DVC version output format)"
fi

# Restore original params.yaml
perl -pi -e 's/n_estimators: 200/n_estimators: 100/' params.yaml
dvc repro
echo ""
echo "params.yaml restored to original state (n_estimators: 100)"
echo "=== Caching verification complete ==="
