"""
DVC Stage 3 — train

Loads training data from the features .npz file, reads hyperparameters from
the train section of params.yaml, trains a RandomForestClassifier, and saves
the serialized model to the output path.

Usage:
    python src/train.py --input data/features.npz --output models/model.joblib
"""

import argparse
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PARAMS_FILE = Path("params.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Model training stage")
    parser.add_argument("--input", required=True, help="Path to features .npz file")
    parser.add_argument("--output", required=True, help="Path to write model .joblib file")
    return parser.parse_args()


def load_params() -> dict:
    if not PARAMS_FILE.exists():
        logger.error("params.yaml not found at %s", PARAMS_FILE)
        sys.exit(1)
    with open(PARAMS_FILE) as fh:
        all_params = yaml.safe_load(fh)
    return all_params.get("train", {})


def build_model(params: dict) -> RandomForestClassifier:
    model_type = params.get("model_type", "random_forest")
    n_estimators = int(params.get("n_estimators", 100))
    max_depth_raw = params.get("max_depth", 10)
    max_depth = (
        None
        if max_depth_raw in (None, "null", "None", "none")
        else int(max_depth_raw)
    )
    random_state = int(params.get("random_state", 42))

    if model_type != "random_forest":
        logger.warning(
            "model_type '%s' is not supported; falling back to random_forest", model_type
        )

    logger.info(
        "Building RandomForestClassifier — n_estimators=%d, max_depth=%s, random_state=%d",
        n_estimators,
        max_depth,
        random_state,
    )
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )


def main() -> None:
    args = parse_args()
    params = load_params()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Features file not found: %s", input_path)
        sys.exit(1)

    logger.info("Loading training data from %s", input_path)
    data = np.load(input_path)
    X_train = data["X_train"]
    y_train = data["y_train"]
    logger.info("Training set shape: %s", X_train.shape)

    model = build_model(params)

    logger.info("Fitting model")
    model.fit(X_train, y_train)
    logger.info("Training complete")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    logger.info("Model saved to %s", output_path)


if __name__ == "__main__":
    main()
