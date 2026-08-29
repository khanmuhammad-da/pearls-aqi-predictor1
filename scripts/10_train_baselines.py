"""
Pearls AQI Predictor
STEP 10 — BASELINE FORECAST MODELS

Purpose:
    Establish simple forecasting benchmarks before training
    a machine-learning model.

Baselines:
    1. Persistence
    2. Previous 24-hour mean
    3. Seasonal persistence

Evaluation:
    MAE
    RMSE

Horizons:
    1h ... 72h

Important:
    No model training occurs in this step.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SPLIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "splits"
)

REPORT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "baseline_results.json"
)

TARGET = "us_aqi"
HORIZON = 72


# ============================================================
# HELPERS
# ============================================================

def print_section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def mae(y_true, y_pred):
    """Mean Absolute Error."""

    return float(
        np.mean(
            np.abs(
                np.asarray(y_true)
                - np.asarray(y_pred)
            )
        )
    )


def rmse(y_true, y_pred):
    """Root Mean Squared Error."""

    return float(
        np.sqrt(
            np.mean(
                (
                    np.asarray(y_true)
                    - np.asarray(y_pred)
                ) ** 2
            )
        )
    )


def horizon_target_columns():
    return [
        f"{TARGET}_t_plus_{i}"
        for i in range(1, HORIZON + 1)
    ]


def validate_split(df, name):

    if "time" not in df.columns:
        raise RuntimeError(
            f"{name}: missing time column."
        )

    df["time"] = pd.to_datetime(df["time"])

    if not df["time"].is_monotonic_increasing:
        raise RuntimeError(
            f"{name}: timestamps are not chronological."
        )

    target_columns = horizon_target_columns()

    missing = [
        col
        for col in target_columns
        if col not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"{name}: missing targets: {missing}"
        )

    if df[target_columns].isna().any().any():
        raise RuntimeError(
            f"{name}: missing future target values."
        )


# ============================================================
# BASELINE PREDICTIONS
# ============================================================

def persistence_predictions(df):
    """
    Predict every future hour as the current AQI.

    Example:
        current AQI = 150

        t+1 ... t+72
        = 150
    """

    current_aqi = df[TARGET].to_numpy()

    return np.repeat(
        current_aqi[:, None],
        HORIZON,
        axis=1
    )


def rolling_24h_predictions(df):
    """
    Predict every future hour using the previous
    24-hour AQI mean.

    The rolling mean is based only on historical
    information available at the forecast origin.
    """

    rolling_mean = (
        df[TARGET]
        .shift(1)
        .rolling(24)
        .mean()
    )

    predictions = np.repeat(
        rolling_mean.to_numpy()[:, None],
        HORIZON,
        axis=1
    )

    return predictions


def seasonal_persistence_predictions(df):
    """
    Daily seasonal persistence.

    Predict future values using the AQI from the
    corresponding hour one day earlier.

    For horizon h:
        prediction(t+h) = AQI(t+h-24)
    """

    current_values = df[TARGET].to_numpy()

    predictions = np.full(
        (len(df), HORIZON),
        np.nan,
        dtype=float
    )

    # For each forecast horizon
    for h in range(1, HORIZON + 1):

        # Future timestamp corresponds to:
        # origin + h
        #
        # One day earlier:
        # origin + h - 24
        #
        # Relative to origin this is h - 24.
        offset = h - 24

        if offset <= 0:

            source_indices = (
                np.arange(len(df)) + offset
            )

        else:

            source_indices = (
                np.arange(len(df)) + offset
            )

        valid = (
            (source_indices >= 0)
            &
            (source_indices < len(df))
        )

        predictions[valid, h - 1] = (
            current_values[source_indices[valid]]
        )

    return predictions


# ============================================================
# EVALUATION
# ============================================================

def evaluate_predictions(
    df,
    predictions,
    model_name
):

    target_columns = horizon_target_columns()

    y_true = df[target_columns].to_numpy(
        dtype=float
    )

    results = {
        "model": model_name,
        "overall": {},
        "horizons": {}
    }

    valid_mask = ~np.isnan(predictions)

    overall_true = y_true[valid_mask]
    overall_pred = predictions[valid_mask]

    results["overall"]["mae"] = mae(
        overall_true,
        overall_pred
    )

    results["overall"]["rmse"] = rmse(
        overall_true,
        overall_pred
    )

    for h in range(HORIZON):

        true_h = y_true[:, h]
        pred_h = predictions[:, h]

        valid = (
            ~np.isnan(true_h)
            &
            ~np.isnan(pred_h)
        )

        if valid.sum() == 0:
            continue

        results["horizons"][str(h + 1)] = {
            "mae": mae(
                true_h[valid],
                pred_h[valid]
            ),
            "rmse": rmse(
                true_h[valid],
                pred_h[valid]
            ),
            "samples": int(valid.sum())
        }

    return results


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_model_summary(result):

    print(
        f"{result['model']:25s}"
        f" MAE = {result['overall']['mae']:8.3f}"
        f" | RMSE = {result['overall']['rmse']:8.3f}"
    )


def print_selected_horizons(results):

    selected = [1, 6, 12, 24, 48, 72]

    print()

    print(
        f"{'Model':25s}"
        f"{'1h':>10s}"
        f"{'6h':>10s}"
        f"{'12h':>10s}"
        f"{'24h':>10s}"
        f"{'48h':>10s}"
        f"{'72h':>10s}"
    )

    print("-" * 85)

    for result in results:

        values = []

        for h in selected:

            value = result["horizons"].get(
                str(h),
                {}
            ).get("mae", np.nan)

            values.append(value)

        print(
            f"{result['model']:25s}"
            f"{values[0]:10.3f}"
            f"{values[1]:10.3f}"
            f"{values[2]:10.3f}"
            f"{values[3]:10.3f}"
            f"{values[4]:10.3f}"
            f"{values[5]:10.3f}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PEARLS AQI PREDICTOR")
    print("STEP 10 — BASELINE FORECAST MODELS")
    print("=" * 70)

    # --------------------------------------------------------
    # FILE CHECK
    # --------------------------------------------------------

    print_section("INPUT FILE CHECK")

    train_file = SPLIT_DIR / "train.csv"
    validation_file = SPLIT_DIR / "validation.csv"
    test_file = SPLIT_DIR / "test.csv"

    for file in [
        train_file,
        validation_file,
        test_file
    ]:

        if not file.exists():
            raise FileNotFoundError(
                f"Missing split file:\n{file}"
            )

        print(f"{file.name:15s}: FOUND")

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    print_section("LOADING DATA")

    train = pd.read_csv(train_file)
    validation = pd.read_csv(validation_file)
    test = pd.read_csv(test_file)

    print(f"Train      : {len(train):,} rows")
    print(f"Validation : {len(validation):,} rows")
    print(f"Test       : {len(test):,} rows")

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    print_section("DATA VALIDATION")

    validate_split(train, "Train")
    validate_split(validation, "Validation")
    validate_split(test, "Test")

    print("Train      : PASS")
    print("Validation : PASS")
    print("Test       : PASS")

    # --------------------------------------------------------
    # BASELINES
    # --------------------------------------------------------

    models = [
        (
            "Persistence",
            persistence_predictions
        ),
        (
            "Previous 24h Mean",
            rolling_24h_predictions
        ),
        (
            "Seasonal Persistence",
            seasonal_persistence_predictions
        ),
    ]

    all_results = {}

    # --------------------------------------------------------
    # EVALUATE EACH SPLIT
    # --------------------------------------------------------

    for split_name, df in [
        ("validation", validation),
        ("test", test)
    ]:

        print_section(
            f"{split_name.upper()} BASELINE RESULTS"
        )

        split_results = []

        for model_name, model_function in models:

            predictions = model_function(df)

            result = evaluate_predictions(
                df,
                predictions,
                model_name
            )

            split_results.append(result)

            print_model_summary(result)

        all_results[split_name] = split_results

        print()
        print("Selected horizon MAE:")
        print_selected_horizons(split_results)

    # --------------------------------------------------------
    # DETERMINE BEST BASELINE
    # --------------------------------------------------------

    print_section("BEST BASELINE")

    validation_results = all_results["validation"]

    best = min(
        validation_results,
        key=lambda x: x["overall"]["mae"]
    )

    print(
        f"Best validation baseline : "
        f"{best['model']}"
    )

    print(
        f"Validation MAE           : "
        f"{best['overall']['mae']:.3f}"
    )

    print(
        f"Validation RMSE          : "
        f"{best['overall']['rmse']:.3f}"
    )

    # --------------------------------------------------------
    # SAVE REPORT
    # --------------------------------------------------------

    print_section("SAVING BASELINE REPORT")

    report = {
        "step": 10,
        "target": TARGET,
        "forecast_horizon_hours": HORIZON,

        "models": [
            "Persistence",
            "Previous 24h Mean",
            "Seasonal Persistence"
        ],

        "results": all_results,

        "best_validation_baseline": {
            "model": best["model"],
            "mae": best["overall"]["mae"],
            "rmse": best["overall"]["rmse"]
        }
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

    print(
        f"Report saved to:\n{REPORT_FILE}"
    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print_section("STEP 10 COMPLETE")

    print("Baseline models evaluated : 3")
    print("Forecast horizons         : 1–72 hours")
    print("Validation evaluation     : PASS")
    print("Test evaluation           : PASS")

    print()
    print(
        "Next step: train the first machine-learning "
        "72-hour forecasting model."
    )


if __name__ == "__main__":
    main()