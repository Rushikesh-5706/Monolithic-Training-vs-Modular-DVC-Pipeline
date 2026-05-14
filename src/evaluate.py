"""
DVC Stage 4 — evaluate

Loads the trained model and test data, computes accuracy, ROC-AUC, and
macro-averaged F1 score, and writes the results to a JSON metrics file.

Usage:
    python src/evaluate.py \
        --model models/model.joblib \
        --data data/features.npz \
        --metrics metrics/scores.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Model evaluation stage")
    parser.add_argument("--model", required=True, help="Path to trained model .joblib file")
    parser.add_argument("--data", required=True, help="Path to features .npz file")
    parser.add_argument("--metrics", required=True, help="Path to write metrics JSON file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_path = Path(args.model)
    data_path = Path(args.data)

    if not model_path.exists():
        logger.error("Model file not found: %s", model_path)
        sys.exit(1)
    if not data_path.exists():
        logger.error("Features file not found: %s", data_path)
        sys.exit(1)

    logger.info("Loading model from %s", model_path)
    model = joblib.load(model_path)

    logger.info("Loading test data from %s", data_path)
    data = np.load(data_path)
    X_test = data["X_test"]
    y_test = data["y_test"]
    logger.info("Test set shape: %s", X_test.shape)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(round(accuracy_score(y_test, y_pred), 6)),
        "auc": float(round(roc_auc_score(y_test, y_prob), 6)),
        "f1_macro": float(round(f1_score(y_test, y_pred, average="macro"), 6)),
    }

    logger.info(
        "accuracy=%.4f  auc=%.4f  f1_macro=%.4f",
        metrics["accuracy"],
        metrics["auc"],
        metrics["f1_macro"],
    )

    metrics_path = Path(args.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("Metrics written to %s", metrics_path)


if __name__ == "__main__":
    main()
