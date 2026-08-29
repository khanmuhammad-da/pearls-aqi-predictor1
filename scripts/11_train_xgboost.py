"""
PEARLS AQI PREDICTOR
STEP 12 — XGBOOST REGRESSOR

Train one XGBoost regression model for each forecast horizon (1–72 hours).

Evaluation protocol:
- Same train / validation / test datasets as Step 11
- Same 101 ML features
- Same 72 forecasting targets
- No random shuffling
- MAE and RMSE on validation and test sets
- Compare against the Step 10 best baseline

Input:
    data/processed/splits/train.csv
    data/processed/splits/validation.csv
    data/processed/splits/test.csv

Outputs:
    models/artifacts/xgboost/xgboost_horizon_01.joblib
    ...
    models/artifacts/xgboost/xgboost_horizon_72.joblib
    reports/xgboost_results.json
"""

from pathlib import Path
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from xgboost import XGBRegressor
except ImportError:
    raise ImportError(
        "\nXGBoost is not installed in the current virtual environment.\n\n"
        "Install it with:\n"
        "    python -m pip install xgboost\n\n"
        "Then verify with:\n"
        "    python -c \"import xgboost; print(xgboost.__version__)\"\n"
    )


# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "splits"
    / "train.csv"
)

VALIDATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "splits"
    / "validation.csv"
)

TEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "splits"
    / "test.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "artifacts"
    / "xgboost"
)

REPORT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "xgboost_results.json"
)

TARGET_COLUMN = "us_aqi"
FORECAST_HORIZON = 72

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# XGBoost configuration
#
# This is deliberately a strong but reasonably conservative first model.
# Hyperparameter tuning is NOT performed in Step 12.
# ---------------------------------------------------------------------------

XGB_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "objective": "reg:squarederror",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "tree_method": "hist",
}


# ============================================================================
# HELPERS
# ============================================================================

def print_header(title):
    print("=" * 70)
    print(title)
    print("=" * 70)


def validate_dataset(df, name):
    """Validate basic dataset integrity."""

    required_columns = {"time", TARGET_COLUMN}

    for horizon in range(1, FORECAST_HORIZON + 1):
        required_columns.add(f"{TARGET_COLUMN}_t_plus_{horizon}")

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"{name} is missing required columns: {sorted(missing)}"
        )

    if df["time"].duplicated().any():
        raise ValueError(f"{name} contains duplicate timestamps.")

    if not df["time"].is_monotonic_increasing:
        raise ValueError(f"{name} timestamps are not chronological.")

    if df.isna().any().any():
        missing_count = int(df.isna().sum().sum())
        raise ValueError(
            f"{name} contains {missing_count} missing values."
        )

    return True


def calculate_metrics(y_true, y_pred):
    """Return MAE and RMSE."""

    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    return float(mae), float(rmse)


def get_feature_columns(df):
    """
    Reproduce the Step 11 feature selection protocol.

    The supervised dataset contains:
      - time
      - 101 ML input features
      - 72 future target columns

    Exclude:
      - time
      - future target columns

    Keep all remaining columns.
    """

    future_target_columns = [
        f"{TARGET_COLUMN}_t_plus_{horizon}"
        for horizon in range(1, FORECAST_HORIZON + 1)
    ]

    excluded_columns = {"time", *future_target_columns}

    feature_columns = [
        column
        for column in df.columns
        if column not in excluded_columns
    ]

    return feature_columns


# ============================================================================
# MAIN
# ============================================================================

def main():

    start_time = time.time()

    print_header(
        "PEARLS AQI PREDICTOR\n"
        "STEP 12 — XGBOOST REGRESSOR"
    )

    # ------------------------------------------------------------------------
    # INPUT FILE CHECK
    # ------------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("INPUT FILE CHECK")
    print("=" * 70)

    files = {
        "train.csv": TRAIN_FILE,
        "validation.csv": VALIDATION_FILE,
        "test.csv": TEST_FILE,
    }

    for name, path in files.items():
        status = "FOUND" if path.exists() else "MISSING"
        print(f"{name:<15}: {status}")

        if not path.exists():
            raise FileNotFoundError(
                f"Required input file not found: {path}"
            )

    # ------------------------------------------------------------------------
    # LOADING DATA
    # ------------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    train = pd.read_csv(TRAIN_FILE)
    validation = pd.read_csv(VALIDATION_FILE)
    test = pd.read_csv(TEST_FILE)

    for df in (train, validation, test):
        df["time"] = pd.to_datetime(df["time"])

    print(f"Train      : {len(train):,} rows")
    print(f"Validation : {len(validation):,} rows")
    print(f"Test       : {len(test):,} rows")

    # ------------------------------------------------------------------------
    # DATA VALIDATION
    # ------------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("DATA VALIDATION")
    print("=" * 70)

    validate_dataset(train, "Train")
    print("Train       : PASS")

    validate_dataset(validation, "Validation")
    print("Validation  : PASS")

    validate_dataset(test, "Test")
    print("Test        : PASS")

    # ------------------------------------------------------------------------
    # FEATURE PREPARATION
    # ------------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("FEATURE PREPARATION")
    print("=" * 70)

    feature_columns = get_feature_columns(train)

    future_target_columns = [
        f"{TARGET_COLUMN}_t_plus_{horizon}"
        for horizon in range(1, FORECAST_HORIZON + 1)
    ]

    print(f"Total input columns : {len(train.columns)}")
    print(f"Future targets      : {len(future_target_columns)}")
    print(f"Usable ML features  : {len(feature_columns)}")

    if len(feature_columns) != 101:
        raise ValueError(
            f"Expected 101 ML features to match Step 11, "
            f"but found {len(feature_columns)}."
        )

    # ------------------------------------------------------------------------
    # CREATE FEATURE MATRICES
    # ------------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("CREATING FEATURE MATRICES")
    print("=" * 70)

    X_train = train[feature_columns].copy()
    X_validation = validation[feature_columns].copy()
    X_test = test[feature_columns].copy()

    print(f"X_train      : {X_train.shape}")
    print(f"X_validation : {X_validation.shape}")
    print(f"X_test       : {X_test.shape}")

    if X_train.isna().any().any():
        raise ValueError("Missing values found in X_train.")

    if X_validation.isna().any().any():
        raise ValueError("Missing values found in X_validation.")

    if X_test.isna().any().any():
        raise ValueError("Missing values found in X_test.")

    print("Feature missing-value check: PASS")

    # ------------------------------------------------------------------------
    # CREATE MODEL DIRECTORY
    # ------------------------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # ------------------------------------------------------------------------
    # MODEL PARAMETERS
    # ------------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("XGBOOST CONFIGURATION")
    print("=" * 70)

    for key, value in XGB_PARAMS.items():
        print(f"{key:<20}: {value}")

    print("\nModels will be trained independently for horizons 1–72h.")

    # ------------------------------------------------------------------------
    # TRAINING
    # ------------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING XGBOOST MODELS")
    print("=" * 70)

    print("Forecast horizons : 1–72 hours")

    validation_results = []
    test_results = []

    all_predictions = {}

    total_training_start = time.time()

    for horizon in range(1, FORECAST_HORIZON + 1):

        horizon_start = time.time()

        target_column = f"{TARGET_COLUMN}_t_plus_{horizon}"

        y_train = train[target_column]
        y_validation = validation[target_column]
        y_test = test[target_column]

        # ------------------------------------------------------------
        # MODEL
        # ------------------------------------------------------------

        model = XGBRegressor(
            **XGB_PARAMS
        )

        # ------------------------------------------------------------
        # TRAIN
        # ------------------------------------------------------------

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            model.fit(
                X_train,
                y_train,
                eval_set=[
                    (X_validation, y_validation)
                ],
                verbose=False,
            )

        # ------------------------------------------------------------
        # VALIDATION PREDICTIONS
        # ------------------------------------------------------------

        validation_prediction = model.predict(
            X_validation
        )

        validation_mae, validation_rmse = calculate_metrics(
            y_validation,
            validation_prediction
        )

        # ------------------------------------------------------------
        # TEST PREDICTIONS
        # ------------------------------------------------------------

        test_prediction = model.predict(
            X_test
        )

        test_mae, test_rmse = calculate_metrics(
            y_test,
            test_prediction
        )

        # ------------------------------------------------------------
        # SAVE MODEL
        # ------------------------------------------------------------

        model_path = (
            MODEL_DIR
            / f"xgboost_horizon_{horizon:02d}.joblib"
        )

        joblib.dump(
            model,
            model_path
        )

        # ------------------------------------------------------------
        # SAVE RESULT
        # ------------------------------------------------------------

        validation_results.append(
            {
                "horizon": horizon,
                "mae": validation_mae,
                "rmse": validation_rmse,
            }
        )

        test_results.append(
            {
                "horizon": horizon,
                "mae": test_mae,
                "rmse": test_rmse,
            }
        )

        all_predictions[horizon] = {
            "validation_mae": validation_mae,
            "validation_rmse": validation_rmse,
            "test_mae": test_mae,
            "test_rmse": test_rmse,
        }

        elapsed = time.time() - horizon_start

        print(
            f"{horizon:02d}h | "
            f"Validation MAE = {validation_mae:8.3f} | "
            f"Validation RMSE = {validation_rmse:8.3f} | "
            f"Test MAE = {test_mae:8.3f} | "
            f"Time = {elapsed:6.1f}s"
        )

    total_training_time = (
        time.time() - total_training_start
    )

    # ------------------------------------------------------------------------
    # SUMMARY STATISTICS
    # ------------------------------------------------------------------------

    validation_mae_values = [
        result["mae"]
        for result in validation_results
    ]

    validation_rmse_values = [
        result["rmse"]
        for result in validation_results
    ]

    test_mae_values = [
        result["mae"]
        for result in test_results
    ]

    test_rmse_values = [
        result["rmse"]
        for result in test_results
    ]

    mean_validation_mae = float(
        np.mean(validation_mae_values)
    )

    mean_validation_rmse = float(
        np.mean(validation_rmse_values)
    )

    mean_test_mae = float(
        np.mean(test_mae_values)
    )

    mean_test_rmse = float(
        np.mean(test_rmse_values)
    )

    best_validation_index = int(
        np.argmin(validation_mae_values)
    )

    best_validation_horizon = (
        best_validation_index + 1
    )

    best_validation_mae = (
        validation_mae_values[
            best_validation_index
        ]
    )

    best_validation_rmse = (
        validation_rmse_values[
            best_validation_index
        ]
    )

    best_test_index = int(
        np.argmin(test_mae_values)
    )

    best_test_horizon = (
        best_test_index + 1
    )

    best_test_mae = (
        test_mae_values[
            best_test_index
        ]
    )

    # ------------------------------------------------------------------------
    # BASELINE COMPARISON
    # ------------------------------------------------------------------------

    # Step 10 selected Seasonal Persistence:
    # validation MAE = 20.231
    # validation RMSE = 26.661

    baseline_validation_mae = 20.231
    baseline_validation_rmse = 26.661

    baseline_comparison = (
        "BETTER"
        if mean_validation_mae < baseline_validation_mae
        else "NOT BETTER"
    )

    # ------------------------------------------------------------------------
    # HORIZON TABLE
    # ------------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("XGBOOST SUMMARY BY HORIZON")
    print("=" * 70)

    print(
        "\n"
        "Model                            "
        "1h        6h       12h       24h       48h       72h"
    )

    print("-" * 85)

    selected_horizons = [
        1,
        6,
        12,
        24,
        48,
        72,
    ]

    for horizon in selected_horizons:

        result = validation_results[horizon - 1]

        print(
            f"Validation MAE              "
            f"{horizon:02d}h: "
            f"{result['mae']:8.3f}"
        )

    print()

    for horizon in selected_horizons:

        result = test_results[horizon - 1]

        print(
            f"Test MAE                    "
            f"{horizon:02d}h: "
            f"{result['mae']:8.3f}"
        )

    # ------------------------------------------------------------------------
    # FINAL SUMMARY
    # ------------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("XGBOOST SUMMARY")
    print("=" * 70)

    print(
        f"Mean validation MAE : "
        f"{mean_validation_mae:.3f}"
    )

    print(
        f"Mean validation RMSE: "
        f"{mean_validation_rmse:.3f}"
    )

    print(
        f"Mean test MAE       : "
        f"{mean_test_mae:.3f}"
    )

    print(
        f"Mean test RMSE      : "
        f"{mean_test_rmse:.3f}"
    )

    print()

    print(
        f"Best validation horizon : "
        f"{best_validation_horizon}h"
    )

    print(
        f"Best validation MAE     : "
        f"{best_validation_mae:.3f}"
    )

    print(
        f"Best validation RMSE    : "
        f"{best_validation_rmse:.3f}"
    )

    print()

    print(
        f"Best test horizon       : "
        f"{best_test_horizon}h"
    )

    print(
        f"Best test MAE           : "
        f"{best_test_mae:.3f}"
    )

    print()

    print(
        f"Baseline validation MAE : "
        f"{baseline_validation_mae:.3f}"
    )

    print(
        f"XGBoost baseline comparison: "
        f"{baseline_comparison}"
    )

    # ------------------------------------------------------------------------
    # SAVE REPORT
    # ------------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("SAVING XGBOOST REPORT")
    print("=" * 70)

    report = {
        "step": 12,
        "model": "XGBoost Regressor",

        "dataset": {
            "train_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "test_rows": int(len(test)),
            "feature_count": int(len(feature_columns)),
            "forecast_horizon": FORECAST_HORIZON,
            "target": TARGET_COLUMN,
        },

        "features": feature_columns,

        "xgboost_parameters": XGB_PARAMS,

        "validation": {
            "mean_mae": mean_validation_mae,
            "mean_rmse": mean_validation_rmse,
            "best_horizon": best_validation_horizon,
            "best_mae": best_validation_mae,
            "best_rmse": best_validation_rmse,
            "by_horizon": validation_results,
        },

        "test": {
            "mean_mae": mean_test_mae,
            "mean_rmse": mean_test_rmse,
            "best_horizon": best_test_horizon,
            "best_mae": best_test_mae,
            "best_rmse": test_results[
                best_test_index
            ]["rmse"],
            "by_horizon": test_results,
        },

        "baseline": {
            "model": "Seasonal Persistence",
            "validation_mae": baseline_validation_mae,
            "validation_rmse": baseline_validation_rmse,
            "comparison": baseline_comparison,
        },

        "training": {
            "models_trained": FORECAST_HORIZON,
            "random_state": RANDOM_STATE,
            "training_time_seconds": total_training_time,
        },

        "model_directory": str(MODEL_DIR),
    }

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            report,
            file,
            indent=2
        )

    print(
        f"Report saved to:\n"
        f"{REPORT_FILE}"
    )

    # ------------------------------------------------------------------------
    # COMPLETE
    # ------------------------------------------------------------------------

    total_time = time.time() - start_time

    print("\n" + "=" * 70)
    print("STEP 12 COMPLETE")
    print("=" * 70)

    print(
        f"Models trained       : "
        f"{FORECAST_HORIZON}"
    )

    print(
        f"Features used        : "
        f"{len(feature_columns)}"
    )

    print(
        f"Forecast horizons    : "
        f"1–{FORECAST_HORIZON} hours"
    )

    print(
        f"Training time        : "
        f"{total_training_time:.1f} seconds"
    )

    print(
        f"Total execution time : "
        f"{total_time:.1f} seconds"
    )

    print()

    print(
        f"Model directory:\n"
        f"{MODEL_DIR}"
    )

    print()

    print(
        f"Report:\n"
        f"{REPORT_FILE}"
    )


if __name__ == "__main__":
    main()