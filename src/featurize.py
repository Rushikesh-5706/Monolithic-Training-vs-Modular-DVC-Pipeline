"""
DVC Stage 2 — featurize

Loads the cleaned dataset, label-encodes all categorical columns and the
target column, performs a stratified train/test split using parameters from
params.yaml, and saves all four arrays (X_train, X_test, y_train, y_test)
to a single compressed .npz file.

Usage:
    python src/featurize.py --input data/processed.csv --output data/features.npz
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

CATEGORICAL_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
TARGET_COLUMN = "income"
PARAMS_FILE = Path("params.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Feature engineering: encode and split dataset")
    parser.add_argument("--input", required=True, help="Path to cleaned input CSV file")
    parser.add_argument("--output", required=True, help="Path to write features .npz file")
    return parser.parse_args()


def load_params() -> dict:
    if not PARAMS_FILE.exists():
        logger.error("params.yaml not found at %s", PARAMS_FILE)
        sys.exit(1)
    with open(PARAMS_FILE) as fh:
        all_params = yaml.safe_load(fh)
    return all_params.get("prepare", {})


def encode_and_split(
    df: pd.DataFrame, random_state: int, test_size: float
):
    logger.info(
        "Encoding features and splitting (test_size=%.2f, random_state=%d)",
        test_size,
        random_state,
    )

    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

    target_encoder = LabelEncoder()
    df[TARGET_COLUMN] = target_encoder.fit_transform(df[TARGET_COLUMN])

    X = df.drop(columns=[TARGET_COLUMN]).values.astype(np.float64)
    y = df[TARGET_COLUMN].values.astype(np.int32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logger.info(
        "Split complete — train: %d rows, test: %d rows", len(X_train), len(X_test)
    )
    return X_train, X_test, y_train, y_test


def main() -> None:
    args = parse_args()
    params = load_params()

    random_state = int(params.get("random_state", 42))
    test_size = float(params.get("test_size", 0.2))

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    logger.info("Reading %s", input_path)
    df = pd.read_csv(input_path)

    X_train, X_test, y_train, y_test = encode_and_split(df, random_state, test_size)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )
    logger.info("Features saved to %s", output_path)


if __name__ == "__main__":
    main()
