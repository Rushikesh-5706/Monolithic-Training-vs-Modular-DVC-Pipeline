"""
Monolithic training pipeline for UCI Adult Income binary classification.

Executes all stages in a single sequential script: data loading, cleaning,
categorical encoding, train/test split, model training, evaluation, and
artifact persistence.

Outputs:
    model.joblib   -- serialized RandomForestClassifier
    metrics.json   -- accuracy, auc, f1_macro as float values
"""

import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_PATH = Path("data/adult.csv")
MODEL_PATH = Path("model.joblib")
METRICS_PATH = Path("metrics.json")

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 100
MAX_DEPTH = 10

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


def load_data(path: Path) -> pd.DataFrame:
    """Load CSV dataset from disk and strip whitespace from string columns."""
    logger.info("Loading dataset from %s", path)
    if not path.exists():
        logger.error("Dataset not found at %s", path)
        sys.exit(1)

    df = pd.read_csv(path)
    df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)
    logger.info("Loaded %d rows and %d columns", len(df), len(df.columns))
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Replace '?' sentinels with NaN and drop incomplete rows."""
    logger.info("Cleaning: removing rows with missing values")
    initial_count = len(df)
    df = df.replace("?", np.nan)
    df = df.dropna()
    df = df.reset_index(drop=True)
    removed = initial_count - len(df)
    logger.info("Removed %d rows; %d rows remain", removed, len(df))
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode all categorical columns and the target column."""
    logger.info("Encoding categorical features")
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

    target_encoder = LabelEncoder()
    df[TARGET_COLUMN] = target_encoder.fit_transform(df[TARGET_COLUMN])
    logger.info("Encoding complete")
    return df


def split_data(df: pd.DataFrame):
    """Split into stratified train and test sets."""
    logger.info("Splitting data (test_size=%.2f, random_state=%d)", TEST_SIZE, RANDOM_STATE)
    X = df.drop(columns=[TARGET_COLUMN]).values
    y = df[TARGET_COLUMN].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info("Train: %d rows  Test: %d rows", len(X_train), len(X_test))
    return X_train, X_test, y_train, y_test


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    """Instantiate and fit a RandomForestClassifier."""
    logger.info(
        "Training RandomForestClassifier (n_estimators=%d, max_depth=%s, random_state=%d)",
        N_ESTIMATORS,
        MAX_DEPTH,
        RANDOM_STATE,
    )
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    logger.info("Training complete")
    return model


def evaluate_model(model: RandomForestClassifier, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Compute accuracy, ROC-AUC, and macro F1 on the test set."""
    logger.info("Evaluating model on test set")
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
    return metrics


def save_artifacts(model: RandomForestClassifier, metrics: dict) -> None:
    """Persist the trained model and metrics to disk."""
    joblib.dump(model, MODEL_PATH)
    logger.info("Model saved to %s", MODEL_PATH)

    with open(METRICS_PATH, "w") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("Metrics saved to %s", METRICS_PATH)


def main() -> None:
    logger.info("=== Monolithic Training Pipeline — Start ===")
    df = load_data(DATA_PATH)
    df = clean_data(df)
    df = encode_features(df)
    X_train, X_test, y_train, y_test = split_data(df)
    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    save_artifacts(model, metrics)
    logger.info("=== Monolithic Training Pipeline — Complete ===")


if __name__ == "__main__":
    main()
