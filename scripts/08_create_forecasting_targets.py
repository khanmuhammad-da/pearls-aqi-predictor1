"""
PEARLS AQI PREDICTOR
STEP 8 — CREATE 72-HOUR FORECASTING TARGETS

Purpose
-------
Convert the hourly feature dataset into a supervised-learning dataset
for direct 72-hour AQI forecasting.

For each timestamp t:

    Features available at t
                ↓
    Predict:
        t+1, t+2, ..., t+72

No future information is included in the input features.

Output
------
data/processed/lahore_supervised_72h.csv

Report
------
reports/forecast_target_report.json
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd


# ======================================================================
# CONFIGURATION
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "lahore_features_hourly.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "lahore_supervised_72h.csv"
)

REPORT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "forecast_target_report.json"
)

TARGET_COLUMN = "us_aqi"
TIME_COLUMN = "time"
HORIZON = 72


# ======================================================================
# DISPLAY HELPERS
# ======================================================================

WIDTH = 70


def heading(title):
    print()
    print("=" * WIDTH)
    print(title)
    print("=" * WIDTH)


# ======================================================================
# VALIDATION
# ======================================================================

def validate_input(df):
    heading("INPUT VALIDATION")

    required_columns = {TIME_COLUMN, TARGET_COLUMN}

    missing = required_columns - set(df.columns)

    if missing:
        raise RuntimeError(
            f"Missing required columns: {sorted(missing)}"
        )

    print("Required columns : PASS")

    if df[TIME_COLUMN].duplicated().any():
        raise RuntimeError("Duplicate timestamps found.")

    print("Unique timestamps: PASS")

    if not df[TIME_COLUMN].is_monotonic_increasing:
        raise RuntimeError("Timestamps are not chronological.")

    print("Chronological     : PASS")

    time_diff = df[TIME_COLUMN].diff().dropna()

    irregular = time_diff[time_diff != pd.Timedelta(hours=1)]

    print(f"Irregular gaps    : {len(irregular)}")

    if len(irregular) > 0:
        raise RuntimeError(
            "Hourly continuity check failed."
        )

    print("Hourly continuity : PASS")

    if df[TARGET_COLUMN].isna().any():
        raise RuntimeError(
            "Target contains missing values."
        )

    print("Target missing    : 0")


# ======================================================================
# CREATE TARGETS
# ======================================================================

def create_targets(df):
    heading("CREATING FUTURE TARGETS")

    print(f"Forecast horizon : {HORIZON} hours")
    print()
    print("Creating:")
    print("  us_aqi_t_plus_1")
    print("  us_aqi_t_plus_2")
    print("  ...")
    print("  us_aqi_t_plus_72")

    target_columns = []

    for step in range(1, HORIZON + 1):

        column_name = f"{TARGET_COLUMN}_t_plus_{step}"

        # Negative shift moves future AQI values onto the current row.
        df[column_name] = df[TARGET_COLUMN].shift(-step)

        target_columns.append(column_name)

    print()
    print(f"Target columns created: {len(target_columns)}")

    return df, target_columns


# ======================================================================
# REMOVE ROWS WITHOUT COMPLETE FUTURE HORIZON
# ======================================================================

def remove_incomplete_targets(df, target_columns):
    heading("TARGET COMPLETENESS")

    before = len(df)

    complete_mask = df[target_columns].notna().all(axis=1)

    df = df.loc[complete_mask].copy()

    after = len(df)

    removed = before - after

    print(f"Rows before       : {before:,}")
    print(f"Rows after        : {after:,}")
    print(f"Rows removed      : {removed:,}")

    expected_removed = HORIZON

    if removed != expected_removed:
        raise RuntimeError(
            f"Unexpected row loss. "
            f"Expected {expected_removed}, got {removed}."
        )

    print("Expected row loss : PASS")

    return df


# ======================================================================
# VALIDATE FUTURE TARGET CONTINUITY
# ======================================================================

def validate_target_continuity(df, target_columns):
    heading("72-HOUR TARGET VALIDATION")

    print("Checking every row has 72 future hourly targets...")

    invalid_rows = 0

    for _, row in df.iterrows():

        values = row[target_columns].to_numpy(dtype=float)

        if np.isnan(values).any():
            invalid_rows += 1
            continue

    if invalid_rows > 0:
        raise RuntimeError(
            f"{invalid_rows} rows contain incomplete targets."
        )

    print("Complete 72-hour targets : PASS")

    # Verify target columns are ordered correctly.
    expected_columns = [
        f"{TARGET_COLUMN}_t_plus_{i}"
        for i in range(1, HORIZON + 1)
    ]

    if target_columns != expected_columns:
        raise RuntimeError(
            "Target column ordering is incorrect."
        )

    print("Target ordering          : PASS")


# ======================================================================
# LEAKAGE CHECK
# ======================================================================

def leakage_check(df, target_columns):
    heading("TARGET LEAKAGE CHECK")

    feature_columns = [
        column
        for column in df.columns
        if column not in target_columns
    ]

    # The existing us_aqi at time t is allowed because it represents
    # information available at forecast origin t.
    #
    # Future target columns must not be present among features.
    leaked_targets = set(target_columns).intersection(
        feature_columns
    )

    if leaked_targets:
        raise RuntimeError(
            f"Future target leakage detected: {sorted(leaked_targets)}"
        )

    print("Future targets in input features : NONE")
    print("Leakage check                     : PASS")


# ======================================================================
# DATASET SUMMARY
# ======================================================================

def create_report(
    original_rows,
    final_rows,
    feature_columns,
    target_columns,
    df,
):
    report = {
        "project": "Pearls AQI Predictor",
        "step": 8,
        "description": "72-hour direct multi-output forecasting dataset",
        "input_file": str(INPUT_FILE),
        "output_file": str(OUTPUT_FILE),
        "time_column": TIME_COLUMN,
        "source_target_column": TARGET_COLUMN,
        "forecast_horizon_hours": HORIZON,
        "original_rows": int(original_rows),
        "final_rows": int(final_rows),
        "rows_removed": int(original_rows - final_rows),
        "feature_count": int(len(feature_columns)),
        "target_count": int(len(target_columns)),
        "feature_columns": feature_columns,
        "target_columns": target_columns,
        "dataset_start": str(df[TIME_COLUMN].min()),
        "dataset_end": str(df[TIME_COLUMN].max()),
        "first_forecast_target": target_columns[0],
        "last_forecast_target": target_columns[-1],
        "target_missing_values": int(
            df[target_columns].isna().sum().sum()
        ),
        "status": "PASS",
    }

    return report


# ======================================================================
# MAIN
# ======================================================================

def main():

    print("=" * WIDTH)
    print("PEARLS AQI PREDICTOR")
    print("STEP 8 — CREATE 72-HOUR FORECASTING TARGETS")
    print("=" * WIDTH)

    # ------------------------------------------------------------------
    # CHECK INPUT
    # ------------------------------------------------------------------

    heading("INPUT FILE CHECK")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    print(f"Input file : FOUND")
    print(f"Path       : {INPUT_FILE}")

    # ------------------------------------------------------------------
    # LOAD DATA
    # ------------------------------------------------------------------

    heading("LOADING FEATURE DATA")

    df = pd.read_csv(INPUT_FILE)

    df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN])

    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns)}")
    print(f"Start   : {df[TIME_COLUMN].min()}")
    print(f"End     : {df[TIME_COLUMN].max()}")

    original_rows = len(df)

    # ------------------------------------------------------------------
    # VALIDATE
    # ------------------------------------------------------------------

    validate_input(df)

    # ------------------------------------------------------------------
    # CREATE FUTURE TARGETS
    # ------------------------------------------------------------------

    df, target_columns = create_targets(df)

    # ------------------------------------------------------------------
    # REMOVE LAST 72 ROWS
    # ------------------------------------------------------------------

    df = remove_incomplete_targets(
        df,
        target_columns,
    )

    # ------------------------------------------------------------------
    # TARGET VALIDATION
    # ------------------------------------------------------------------

    validate_target_continuity(
        df,
        target_columns,
    )

    # ------------------------------------------------------------------
    # LEAKAGE CHECK
    # ------------------------------------------------------------------

    leakage_check(
        df,
        target_columns,
    )

    # ------------------------------------------------------------------
    # IDENTIFY FEATURES
    # ------------------------------------------------------------------

    feature_columns = [
        column
        for column in df.columns
        if column not in target_columns
    ]

    heading("FINAL DATASET STRUCTURE")

    print(f"Rows             : {len(df):,}")
    print(f"Input features   : {len(feature_columns):,}")
    print(f"Future targets   : {len(target_columns):,}")
    print(f"Total columns    : {len(df.columns):,}")

    print()
    print("Feature dataset:")
    print("  Current information at time t")

    print()
    print("Target dataset:")
    print("  AQI t+1")
    print("  AQI t+2")
    print("  AQI t+3")
    print("  ...")
    print("  AQI t+72")

    # ------------------------------------------------------------------
    # TARGET RANGE
    # ------------------------------------------------------------------

    heading("TARGET STATISTICS")

    all_targets = df[target_columns].to_numpy(dtype=float)

    print(f"Minimum future AQI : {all_targets.min():.2f}")
    print(f"Maximum future AQI : {all_targets.max():.2f}")
    print(f"Mean future AQI    : {all_targets.mean():.2f}")

    # ------------------------------------------------------------------
    # SAVE CSV
    # ------------------------------------------------------------------

    heading("SAVING SUPERVISED DATASET")

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(f"Saved to:")
    print(OUTPUT_FILE)

    # ------------------------------------------------------------------
    # REPORT
    # ------------------------------------------------------------------

    heading("SAVING REPORT")

    report = create_report(
        original_rows=original_rows,
        final_rows=len(df),
        feature_columns=feature_columns,
        target_columns=target_columns,
        df=df,
    )

    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
        )

    print(f"Report saved to:")
    print(REPORT_FILE)

    # ------------------------------------------------------------------
    # SAMPLE
    # ------------------------------------------------------------------

    heading("FIRST FORECASTING EXAMPLE")

    sample = df.iloc[0]

    print(f"Forecast origin : {sample[TIME_COLUMN]}")
    print(f"Current AQI     : {sample[TARGET_COLUMN]}")

    print()
    print("Future AQI:")

    for i in range(1, HORIZON + 1):

        column = f"{TARGET_COLUMN}_t_plus_{i}"

        print(
            f"  t+{i:02d}h : "
            f"{sample[column]:.1f}"
        )

    # ------------------------------------------------------------------
    # COMPLETE
    # ------------------------------------------------------------------

    heading("STEP 8 COMPLETE")

    print(f"Input rows       : {original_rows:,}")
    print(f"Output rows      : {len(df):,}")
    print(f"Features         : {len(feature_columns):,}")
    print(f"Future targets   : {len(target_columns):,}")
    print(f"Forecast horizon : {HORIZON} hours")
    print(f"Target           : {TARGET_COLUMN}")

    print()
    print("Output file:")
    print(OUTPUT_FILE)

    print()
    print("Report:")
    print(REPORT_FILE)

    print()
    print("The dataset is ready for the chronological")
    print("train / validation / test split.")


if __name__ == "__main__":
    main()