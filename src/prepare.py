"""
DVC Stage 1 — prepare

Loads raw data from the input CSV, strips whitespace, replaces '?' sentinel
values with NaN, and drops all rows containing missing values. Writes the
cleaned dataset to the output path.

Usage:
    python src/prepare.py --input data/adult.csv --output data/processed.csv
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Data preparation: load and clean raw dataset")
    parser.add_argument("--input", required=True, help="Path to raw input CSV file")
    parser.add_argument("--output", required=True, help="Path to write cleaned CSV file")
    return parser.parse_args()


def load_and_clean(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    logger.info("Reading %s", input_path)
    df = pd.read_csv(input_path)
    df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)

    initial_count = len(df)
    df = df.replace("?", np.nan)
    df = df.dropna()
    df = df.reset_index(drop=True)

    removed = initial_count - len(df)
    logger.info("Removed %d rows with missing values; %d rows remaining", removed, len(df))
    return df


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    df = load_and_clean(input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Cleaned dataset written to %s", output_path)


if __name__ == "__main__":
    main()
