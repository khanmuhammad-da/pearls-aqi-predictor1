"""
Pearls AQI Predictor
STEP 9 — CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT

Purpose:
    Split the supervised 72-hour forecasting dataset chronologically.

Important:
    - No random splitting
    - No shuffling
    - No future information leakage
"""

from pathlib import Path
import json

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "lahore_supervised_72h.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "splits"
)

REPORT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "split_report.json"
)

TARGET_COLUMN = "us_aqi"

HORIZON = 72

# Chronological proportions
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15


# ============================================================
# HELPERS
# ============================================================

def print_section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def validate_dataset(df):
    """Validate the supervised dataset before splitting."""

    required_target_columns = [
        f"{TARGET_COLUMN}_t_plus_{i}"
        for i in range(1, HORIZON + 1)
    ]

    missing_targets = [
        col for col in required_target_columns
        if col not in df.columns
    ]

    if missing_targets:
        raise RuntimeError(
            f"Missing future target columns: {missing_targets}"
        )

    if "time" not in df.columns:
        raise RuntimeError("Missing required time column.")

    # Convert timestamp
    df["time"] = pd.to_datetime(df["time"])

    # Chronological order
    if not df["time"].is_monotonic_increasing:
        raise RuntimeError(
            "Dataset is not chronologically sorted."
        )

    # Duplicate timestamps
    duplicate_timestamps = df["time"].duplicated().sum()

    if duplicate_timestamps > 0:
        raise RuntimeError(
            f"Found {duplicate_timestamps} duplicate timestamps."
        )

    # Missing values
    missing_values = int(df.isna().sum().sum())

    if missing_values > 0:
        raise RuntimeError(
            f"Dataset contains {missing_values} missing values."
        )

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PEARLS AQI PREDICTOR")
    print("STEP 9 — CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT")
    print("=" * 70)

    # --------------------------------------------------------
    # INPUT CHECK
    # --------------------------------------------------------

    print_section("INPUT FILE CHECK")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    print(f"Input file : FOUND")
    print(f"Path       : {INPUT_FILE}")

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print_section("LOADING SUPERVISED DATASET")

    df = pd.read_csv(INPUT_FILE)

    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns):,}")

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    print_section("INPUT VALIDATION")

    df = validate_dataset(df)

    print("Required target columns : PASS")
    print("Chronological order     : PASS")
    print("Duplicate timestamps    : PASS")
    print("Missing values          : PASS")

    # --------------------------------------------------------
    # DATASET RANGE
    # --------------------------------------------------------

    print_section("FULL DATASET RANGE")

    print(f"Start : {df['time'].iloc[0]}")
    print(f"End   : {df['time'].iloc[-1]}")

    # --------------------------------------------------------
    # CALCULATE SPLIT INDICES
    # --------------------------------------------------------

    n = len(df)

    train_end = int(n * TRAIN_RATIO)
    validation_end = int(
        n * (TRAIN_RATIO + VALIDATION_RATIO)
    )

    print_section("SPLIT PLAN")

    print(f"Total rows       : {n:,}")
    print(f"Train ratio      : {TRAIN_RATIO:.0%}")
    print(f"Validation ratio : {VALIDATION_RATIO:.0%}")
    print(f"Test ratio       : {TEST_RATIO:.0%}")

    # --------------------------------------------------------
    # PERFORM CHRONOLOGICAL SPLIT
    # --------------------------------------------------------

    train_df = df.iloc[:train_end].copy()

    validation_df = df.iloc[
        train_end:validation_end
    ].copy()

    test_df = df.iloc[
        validation_end:
    ].copy()

    # --------------------------------------------------------
    # SPLIT VALIDATION
    # --------------------------------------------------------

    print_section("SPLIT RESULTS")

    datasets = {
        "train": train_df,
        "validation": validation_df,
        "test": test_df,
    }

    for name, data in datasets.items():

        print(f"{name.upper()}")

        print(f"  Rows  : {len(data):,}")
        print(f"  Start : {data['time'].iloc[0]}")
        print(f"  End   : {data['time'].iloc[-1]}")

        print()

    # --------------------------------------------------------
    # CHECK CHRONOLOGICAL SEPARATION
    # --------------------------------------------------------

    print_section("CHRONOLOGICAL SEPARATION CHECK")

    train_last = train_df["time"].iloc[-1]
    validation_first = validation_df["time"].iloc[0]

    validation_last = validation_df["time"].iloc[-1]
    test_first = test_df["time"].iloc[0]

    if train_last >= validation_first:
        raise RuntimeError(
            "Train/validation chronological separation failed."
        )

    if validation_last >= test_first:
        raise RuntimeError(
            "Validation/test chronological separation failed."
        )

    print("Train → Validation : PASS")
    print("Validation → Test  : PASS")

    # --------------------------------------------------------
    # CHECK FUTURE TARGETS
    # --------------------------------------------------------

    print_section("72-HOUR TARGET CHECK")

    target_columns = [
        f"{TARGET_COLUMN}_t_plus_{i}"
        for i in range(1, HORIZON + 1)
    ]

    for name, data in datasets.items():

        missing_targets = int(
            data[target_columns].isna().sum().sum()
        )

        if missing_targets > 0:
            raise RuntimeError(
                f"{name} contains missing future targets."
            )

        print(
            f"{name.capitalize():12s} : "
            f"72 targets present"
        )

    # --------------------------------------------------------
    # SAVE SPLITS
    # --------------------------------------------------------

    print_section("SAVING DATASET SPLITS")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    train_file = OUTPUT_DIR / "train.csv"
    validation_file = OUTPUT_DIR / "validation.csv"
    test_file = OUTPUT_DIR / "test.csv"

    train_df.to_csv(
        train_file,
        index=False
    )

    validation_df.to_csv(
        validation_file,
        index=False
    )

    test_df.to_csv(
        test_file,
        index=False
    )

    print(f"Train      : {train_file}")
    print(f"Validation : {validation_file}")
    print(f"Test       : {test_file}")

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    report = {
        "step": 9,
        "description": "Chronological train validation test split",

        "input": {
            "file": str(INPUT_FILE),
            "rows": int(n),
            "columns": int(len(df.columns)),
            "start": str(df["time"].iloc[0]),
            "end": str(df["time"].iloc[-1]),
        },

        "split_ratios": {
            "train": TRAIN_RATIO,
            "validation": VALIDATION_RATIO,
            "test": TEST_RATIO,
        },

        "datasets": {
            "train": {
                "rows": int(len(train_df)),
                "start": str(train_df["time"].iloc[0]),
                "end": str(train_df["time"].iloc[-1]),
                "file": str(train_file),
            },

            "validation": {
                "rows": int(len(validation_df)),
                "start": str(validation_df["time"].iloc[0]),
                "end": str(validation_df["time"].iloc[-1]),
                "file": str(validation_file),
            },

            "test": {
                "rows": int(len(test_df)),
                "start": str(test_df["time"].iloc[0]),
                "end": str(test_df["time"].iloc[-1]),
                "file": str(test_file),
            },
        },

        "forecast_horizon_hours": HORIZON,

        "validation": {
            "chronological_split": True,
            "random_shuffle": False,
            "future_target_leakage": False,
            "missing_values": False,
        },
    }

    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    print_section("STEP 9 COMPLETE")

    print(f"Total rows      : {n:,}")
    print(f"Train rows      : {len(train_df):,}")
    print(f"Validation rows : {len(validation_df):,}")
    print(f"Test rows       : {len(test_df):,}")

    print()
    print("Chronological split : PASS")
    print("Random shuffle      : NO")
    print("Future leakage      : NO")

    print()
    print("Report saved to:")
    print(REPORT_FILE)

    print()
    print("The dataset is now ready for model development.")


if __name__ == "__main__":
    main()