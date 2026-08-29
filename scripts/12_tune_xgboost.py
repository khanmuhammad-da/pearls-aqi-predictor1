"""
PEARLS AQI PREDICTOR
STEP 13 — XGBOOST HYPERPARAMETER TUNING

Purpose
-------
Tune XGBoost hyperparameters using the chronological validation set.

Important methodology
---------------------
- Train data is used for fitting.
- Validation data is used for hyperparameter selection.
- Test data is NOT used to select hyperparameters.
- The same 101 ML features from Step 12 are used.
- Forecast horizons remain independent: t+1 ... t+72.
- MAE is the primary tuning metric.
- A small representative set of horizons is used during tuning to
  keep the search computationally practical.
- After selecting the best configuration, all 72 horizon models
  are trained and evaluated.

Output
------
Models:
    models/artifacts/xgboost_tuned/

Report:
    reports/xgboost_tuning_results.json
"""

import json
import os
import time
import warnings
from itertools import product

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


# ======================================================================
# CONFIGURATION
# ======================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "splits",
    "train.csv",
)

VALIDATION_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "splits",
    "validation.csv",
)

TEST_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "splits",
    "test.csv",
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "artifacts",
    "xgboost_tuned",
)

REPORT_FILE = os.path.join(
    BASE_DIR,
    "reports",
    "xgboost_tuning_results.json",
)

TARGET_COLUMN = "us_aqi"

FORECAST_HORIZON = 72

RANDOM_STATE = 42

# Horizons used during hyperparameter search.
# We deliberately do not tune using all 72 horizons.
TUNING_HORIZONS = [1, 6, 12, 24, 48, 72]

# Number of candidate parameter configurations.
# The parameter combinations below are intentionally controlled.
MAX_CONFIGURATIONS = 16


# ======================================================================
# HEADER
# ======================================================================

print("=" * 70)
print("PEARLS AQI PREDICTOR")
print("STEP 13 — XGBOOST HYPERPARAMETER TUNING")
print("=" * 70)


# ======================================================================
# INPUT FILE CHECK
# ======================================================================

print("\n" + "=" * 70)
print("INPUT FILE CHECK")
print("=" * 70)

for name, path in [
    ("train.csv", TRAIN_FILE),
    ("validation.csv", VALIDATION_FILE),
    ("test.csv", TEST_FILE),
]:
    if os.path.exists(path):
        print(f"{name:<15}: FOUND")
    else:
        print(f"{name:<15}: NOT FOUND")
        raise FileNotFoundError(f"Required file not found: {path}")


# ======================================================================
# LOAD DATA
# ======================================================================

print("\n" + "=" * 70)
print("LOADING DATA")
print("=" * 70)

train_df = pd.read_csv(TRAIN_FILE)
validation_df = pd.read_csv(VALIDATION_FILE)
test_df = pd.read_csv(TEST_FILE)

print(f"Train      : {len(train_df):,} rows")
print(f"Validation : {len(validation_df):,} rows")
print(f"Test       : {len(test_df):,} rows")


# ======================================================================
# TIMESTAMP CONVERSION
# ======================================================================

for df in [train_df, validation_df, test_df]:
    df["time"] = pd.to_datetime(df["time"])


# ======================================================================
# DATA VALIDATION
# ======================================================================

print("\n" + "=" * 70)
print("DATA VALIDATION")
print("=" * 70)

for name, df in [
    ("Train", train_df),
    ("Validation", validation_df),
    ("Test", test_df),
]:

    if not df["time"].is_unique:
        raise ValueError(f"{name}: duplicate timestamps detected.")

    if not df["time"].is_monotonic_increasing:
        raise ValueError(f"{name}: timestamps are not chronological.")

    if df.isna().any().any():
        raise ValueError(f"{name}: missing values detected.")

    print(f"{name:<12}: PASS")


# ======================================================================
# IDENTIFY FUTURE TARGET COLUMNS
# ======================================================================

future_target_columns = [
    f"{TARGET_COLUMN}_t_plus_{h}"
    for h in range(1, FORECAST_HORIZON + 1)
]

missing_targets = [
    col for col in future_target_columns
    if col not in train_df.columns
]

if missing_targets:
    raise ValueError(
        f"Missing future target columns. Example: {missing_targets[:5]}"
    )


# ======================================================================
# FEATURE PREPARATION
# ======================================================================

print("\n" + "=" * 70)
print("FEATURE PREPARATION")
print("=" * 70)

# Exclude:
# - time
# - current target us_aqi
# - all future target columns
#
# This follows the same feature protocol used in Step 12.

excluded_columns = (
    ["time", TARGET_COLUMN]
    + future_target_columns
)

feature_columns = [
    col for col in train_df.columns
    if col not in excluded_columns
]

print(f"Total input columns : {len(train_df.columns)}")
print(f"Future targets      : {len(future_target_columns)}")
print(f"Usable ML features  : {len(feature_columns)}")


# ======================================================================
# CREATE MATRICES
# ======================================================================

print("\n" + "=" * 70)
print("CREATING FEATURE MATRICES")
print("=" * 70)

X_train = train_df[feature_columns].astype(np.float32)
X_validation = validation_df[feature_columns].astype(np.float32)
X_test = test_df[feature_columns].astype(np.float32)

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


# ======================================================================
# HYPERPARAMETER SEARCH SPACE
# ======================================================================

print("\n" + "=" * 70)
print("HYPERPARAMETER SEARCH SPACE")
print("=" * 70)

# The current Step 12 configuration was:
#
# n_estimators     = 500
# learning_rate    = 0.05
# max_depth        = 6
# min_child_weight = 3
# subsample        = 0.8
# colsample_bytree = 0.8
# reg_alpha        = 0.0
# reg_lambda       = 1.0
#
# We search around this configuration.

parameter_grid = {
    "n_estimators": [300, 500, 700, 900],
    "learning_rate": [0.03, 0.05, 0.08],
    "max_depth": [4, 6, 8],
    "min_child_weight": [1, 3, 5],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "reg_alpha": [0.0, 0.1],
    "reg_lambda": [1.0, 5.0],
}


# ======================================================================
# CONTROLLED CONFIGURATION GENERATION
# ======================================================================

print("\nGenerating controlled parameter configurations...")

all_combinations = list(
    product(
        parameter_grid["n_estimators"],
        parameter_grid["learning_rate"],
        parameter_grid["max_depth"],
        parameter_grid["min_child_weight"],
        parameter_grid["subsample"],
        parameter_grid["colsample_bytree"],
        parameter_grid["reg_alpha"],
        parameter_grid["reg_lambda"],
    )
)

rng = np.random.default_rng(RANDOM_STATE)

# Always include the current Step 12 configuration.
baseline_config = (
    500,
    0.05,
    6,
    3,
    0.8,
    0.8,
    0.0,
    1.0,
)

remaining = [
    combo
    for combo in all_combinations
    if combo != baseline_config
]

rng.shuffle(remaining)

selected_combinations = [
    baseline_config
] + remaining[: MAX_CONFIGURATIONS - 1]


def combination_to_dict(combo):
    return {
        "n_estimators": int(combo[0]),
        "learning_rate": float(combo[1]),
        "max_depth": int(combo[2]),
        "min_child_weight": int(combo[3]),
        "subsample": float(combo[4]),
        "colsample_bytree": float(combo[5]),
        "reg_alpha": float(combo[6]),
        "reg_lambda": float(combo[7]),
    }


parameter_configs = [
    combination_to_dict(combo)
    for combo in selected_combinations
]

print(f"Candidate configurations : {len(parameter_configs)}")
print(f"Tuning horizons          : {TUNING_HORIZONS}")


# ======================================================================
# BASELINE XGBOOST CONFIGURATION
# ======================================================================

print("\n" + "=" * 70)
print("REFERENCE XGBOOST CONFIGURATION")
print("=" * 70)

print(json.dumps(baseline_config and combination_to_dict(baseline_config), indent=2))


# ======================================================================
# TUNING
# ======================================================================

print("\n" + "=" * 70)
print("STARTING HYPERPARAMETER TUNING")
print("=" * 70)

tuning_results = []

overall_tuning_start = time.time()

for config_index, params in enumerate(parameter_configs, start=1):

    print("\n" + "-" * 70)
    print(
        f"CONFIGURATION {config_index}/{len(parameter_configs)}"
    )
    print("-" * 70)

    print(json.dumps(params, indent=2))

    config_start = time.time()

    horizon_results = []

    for horizon in TUNING_HORIZONS:

        target_column = f"{TARGET_COLUMN}_t_plus_{horizon}"

        y_train = train_df[target_column].astype(np.float32)
        y_validation = validation_df[target_column].astype(np.float32)

        model = XGBRegressor(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            min_child_weight=params["min_child_weight"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            reg_alpha=params["reg_alpha"],
            reg_lambda=params["reg_lambda"],
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
        )

        model.fit(
            X_train,
            y_train,
            verbose=False,
        )

        validation_pred = model.predict(X_validation)

        validation_mae = mean_absolute_error(
            y_validation,
            validation_pred,
        )

        validation_rmse = np.sqrt(
            mean_squared_error(
                y_validation,
                validation_pred,
            )
        )

        horizon_results.append(
            {
                "horizon": horizon,
                "validation_mae": float(validation_mae),
                "validation_rmse": float(validation_rmse),
            }
        )

        print(
            f"{horizon:02d}h | "
            f"Validation MAE = {validation_mae:8.3f} | "
            f"Validation RMSE = {validation_rmse:8.3f}"
        )

    mean_mae = float(
        np.mean(
            [
                result["validation_mae"]
                for result in horizon_results
            ]
        )
    )

    mean_rmse = float(
        np.mean(
            [
                result["validation_rmse"]
                for result in horizon_results
            ]
        )
    )

    elapsed = time.time() - config_start

    result = {
        "configuration_id": config_index,
        "parameters": params,
        "mean_validation_mae": mean_mae,
        "mean_validation_rmse": mean_rmse,
        "horizon_results": horizon_results,
        "training_time_seconds": round(elapsed, 2),
    }

    tuning_results.append(result)

    print(
        f"\nConfiguration mean validation MAE : {mean_mae:.3f}"
    )

    print(
        f"Configuration mean validation RMSE: {mean_rmse:.3f}"
    )

    print(
        f"Configuration time                 : {elapsed:.1f}s"
    )


# ======================================================================
# SELECT BEST CONFIGURATION
# ======================================================================

best_result = min(
    tuning_results,
    key=lambda x: x["mean_validation_mae"],
)

best_params = best_result["parameters"]

total_tuning_time = time.time() - overall_tuning_start


# ======================================================================
# TUNING SUMMARY
# ======================================================================

print("\n" + "=" * 70)
print("HYPERPARAMETER TUNING SUMMARY")
print("=" * 70)

ranking = sorted(
    tuning_results,
    key=lambda x: x["mean_validation_mae"],
)

print(
    f"{'Rank':<6}"
    f"{'Config':<8}"
    f"{'MAE':>12}"
    f"{'RMSE':>12}"
)

print("-" * 42)

for rank, result in enumerate(ranking, start=1):

    print(
        f"{rank:<6}"
        f"{result['configuration_id']:<8}"
        f"{result['mean_validation_mae']:>12.3f}"
        f"{result['mean_validation_rmse']:>12.3f}"
    )


# ======================================================================
# BEST CONFIGURATION
# ======================================================================

print("\n" + "=" * 70)
print("BEST XGBOOST CONFIGURATION")
print("=" * 70)

print(json.dumps(best_params, indent=2))

print(
    f"\nBest tuning validation MAE : "
    f"{best_result['mean_validation_mae']:.3f}"
)

print(
    f"Best tuning validation RMSE: "
    f"{best_result['mean_validation_rmse']:.3f}"
)

print(
    f"Total tuning time          : "
    f"{total_tuning_time:.1f}s"
)


# ======================================================================
# COMPARE WITH STEP 12 CONFIGURATION
# ======================================================================

reference_result = next(
    result
    for result in tuning_results
    if result["parameters"] == combination_to_dict(baseline_config)
)

reference_mae = reference_result["mean_validation_mae"]

improvement = reference_mae - best_result["mean_validation_mae"]

improvement_percent = (
    improvement / reference_mae * 100
    if reference_mae != 0
    else 0
)

print("\n" + "=" * 70)
print("COMPARISON WITH STEP 12")
print("=" * 70)

print(
    f"Step 12 XGBoost MAE : {reference_mae:.3f}"
)

print(
    f"Tuned XGBoost MAE   : "
    f"{best_result['mean_validation_mae']:.3f}"
)

print(
    f"MAE improvement     : {improvement:.3f}"
)

print(
    f"Improvement (%)     : {improvement_percent:.2f}%"
)

if improvement > 0:
    print("Tuned XGBoost: IMPROVED")
elif improvement < 0:
    print("Tuned XGBoost: WORSE THAN REFERENCE")
else:
    print("Tuned XGBoost: NO CHANGE")


# ======================================================================
# TRAIN FINAL 72 MODELS
# ======================================================================

print("\n" + "=" * 70)
print("TRAINING FINAL TUNED XGBOOST MODELS")
print("=" * 70)

os.makedirs(MODEL_DIR, exist_ok=True)

final_results = []

final_training_start = time.time()

for horizon in range(1, FORECAST_HORIZON + 1):

    target_column = f"{TARGET_COLUMN}_t_plus_{horizon}"

    y_train = train_df[target_column].astype(np.float32)
    y_validation = validation_df[target_column].astype(np.float32)
    y_test = test_df[target_column].astype(np.float32)

    model_start = time.time()

    model = XGBRegressor(
        n_estimators=best_params["n_estimators"],
        learning_rate=best_params["learning_rate"],
        max_depth=best_params["max_depth"],
        min_child_weight=best_params["min_child_weight"],
        subsample=best_params["subsample"],
        colsample_bytree=best_params["colsample_bytree"],
        reg_alpha=best_params["reg_alpha"],
        reg_lambda=best_params["reg_lambda"],
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
    )

    model.fit(
        X_train,
        y_train,
        verbose=False,
    )

    validation_pred = model.predict(X_validation)
    test_pred = model.predict(X_test)

    validation_mae = mean_absolute_error(
        y_validation,
        validation_pred,
    )

    validation_rmse = np.sqrt(
        mean_squared_error(
            y_validation,
            validation_pred,
        )
    )

    test_mae = mean_absolute_error(
        y_test,
        test_pred,
    )

    test_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            test_pred,
        )
    )

    model_path = os.path.join(
        MODEL_DIR,
        f"xgboost_tuned_{horizon:02d}h.json",
    )

    model.save_model(model_path)

    elapsed = time.time() - model_start

    final_results.append(
        {
            "horizon": horizon,
            "validation_mae": float(validation_mae),
            "validation_rmse": float(validation_rmse),
            "test_mae": float(test_mae),
            "test_rmse": float(test_rmse),
            "training_time_seconds": round(elapsed, 2),
            "model_path": model_path,
        }
    )

    print(
        f"{horizon:02d}h | "
        f"Validation MAE = {validation_mae:8.3f} | "
        f"Validation RMSE = {validation_rmse:8.3f} | "
        f"Test MAE = {test_mae:8.3f} | "
        f"Time = {elapsed:6.1f}s"
    )


final_training_time = time.time() - final_training_start


# ======================================================================
# FINAL HORIZON SUMMARY
# ======================================================================

print("\n" + "=" * 70)
print("TUNED XGBOOST SUMMARY BY HORIZON")
print("=" * 70)

selected_horizons = [1, 6, 12, 24, 48, 72]

print(
    f"{'Model':<30}"
    f"{'1h':>10}"
    f"{'6h':>10}"
    f"{'12h':>10}"
    f"{'24h':>10}"
    f"{'48h':>10}"
    f"{'72h':>10}"
)

print("-" * 90)

validation_by_horizon = {
    result["horizon"]: result["validation_mae"]
    for result in final_results
}

test_by_horizon = {
    result["horizon"]: result["test_mae"]
    for result in final_results
}

print(
    f"{'Validation MAE':<30}"
    + "".join(
        f"{validation_by_horizon[h]:>10.3f}"
        for h in selected_horizons
    )
)

print(
    f"{'Test MAE':<30}"
    + "".join(
        f"{test_by_horizon[h]:>10.3f}"
        for h in selected_horizons
    )
)


# ======================================================================
# OVERALL METRICS
# ======================================================================

mean_validation_mae = float(
    np.mean(
        [
            result["validation_mae"]
            for result in final_results
        ]
    )
)

mean_validation_rmse = float(
    np.mean(
        [
            result["validation_rmse"]
            for result in final_results
        ]
    )
)

mean_test_mae = float(
    np.mean(
        [
            result["test_mae"]
            for result in final_results
        ]
    )
)

mean_test_rmse = float(
    np.mean(
        [
            result["test_rmse"]
            for result in final_results
        ]
    )
)

best_validation_horizon = min(
    final_results,
    key=lambda x: x["validation_mae"],
)

best_test_horizon = min(
    final_results,
    key=lambda x: x["test_mae"],
)


# ======================================================================
# BASELINE REFERENCE
# ======================================================================

# Step 10 seasonal persistence validation mean MAE.
SEASONAL_BASELINE_VALIDATION_MAE = 20.231

baseline_comparison = (
    mean_validation_mae < SEASONAL_BASELINE_VALIDATION_MAE
)


# ======================================================================
# FINAL SUMMARY
# ======================================================================

print("\n" + "=" * 70)
print("TUNED XGBOOST FINAL SUMMARY")
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
    f"{best_validation_horizon['horizon']}h"
)

print(
    f"Best validation MAE     : "
    f"{best_validation_horizon['validation_mae']:.3f}"
)

print()

print(
    f"Best test horizon       : "
    f"{best_test_horizon['horizon']}h"
)

print(
    f"Best test MAE           : "
    f"{best_test_horizon['test_mae']:.3f}"
)

print()

print(
    f"Seasonal baseline "
    f"validation MAE         : "
    f"{SEASONAL_BASELINE_VALIDATION_MAE:.3f}"
)

if baseline_comparison:
    print(
        "Tuned XGBoost baseline comparison: BETTER"
    )
else:
    print(
        "Tuned XGBoost baseline comparison: NOT BETTER"
    )


# ======================================================================
# SAVE REPORT
# ======================================================================

print("\n" + "=" * 70)
print("SAVING TUNING REPORT")
print("=" * 70)

os.makedirs(
    os.path.dirname(REPORT_FILE),
    exist_ok=True,
)

report = {
    "step": 13,
    "description": "XGBoost hyperparameter tuning",
    "target": TARGET_COLUMN,
    "forecast_horizon": FORECAST_HORIZON,
    "features_used": len(feature_columns),
    "feature_columns": feature_columns,
    "tuning_horizons": TUNING_HORIZONS,
    "random_state": RANDOM_STATE,
    "candidate_configurations": len(parameter_configs),
    "reference_configuration": combination_to_dict(
        baseline_config
    ),
    "reference_validation_mae": reference_mae,
    "best_parameters": best_params,
    "best_tuning_validation_mae": best_result[
        "mean_validation_mae"
    ],
    "best_tuning_validation_rmse": best_result[
        "mean_validation_rmse"
    ],
    "tuning_improvement_mae": improvement,
    "tuning_improvement_percent": improvement_percent,
    "tuning_results": tuning_results,
    "final_results": final_results,
    "final_summary": {
        "mean_validation_mae": mean_validation_mae,
        "mean_validation_rmse": mean_validation_rmse,
        "mean_test_mae": mean_test_mae,
        "mean_test_rmse": mean_test_rmse,
        "best_validation_horizon": best_validation_horizon[
            "horizon"
        ],
        "best_validation_mae": best_validation_horizon[
            "validation_mae"
        ],
        "best_test_horizon": best_test_horizon[
            "horizon"
        ],
        "best_test_mae": best_test_horizon[
            "test_mae"
        ],
        "seasonal_baseline_validation_mae":
            SEASONAL_BASELINE_VALIDATION_MAE,
        "beats_seasonal_baseline":
            baseline_comparison,
    },
    "timing": {
        "tuning_seconds": round(total_tuning_time, 2),
        "final_training_seconds": round(
            final_training_time,
            2,
        ),
        "total_seconds": round(
            total_tuning_time + final_training_time,
            2,
        ),
    },
    "model_directory": MODEL_DIR,
}


with open(
    REPORT_FILE,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        report,
        f,
        indent=2,
    )


# ======================================================================
# COMPLETE
# ======================================================================

print(f"Report saved to:")
print(REPORT_FILE)

print("\n" + "=" * 70)
print("STEP 13 COMPLETE")
print("=" * 70)

print(
    f"Tuning configurations : {len(parameter_configs)}"
)

print(
    f"Tuning horizons       : {len(TUNING_HORIZONS)}"
)

print(
    f"Final models trained   : {FORECAST_HORIZON}"
)

print(
    f"Features used          : {len(feature_columns)}"
)

print(
    f"Best validation MAE    : "
    f"{mean_validation_mae:.3f}"
)

print(
    f"Mean test MAE          : "
    f"{mean_test_mae:.3f}"
)

print(
    f"Final model directory:")
print(MODEL_DIR)

print(
    f"\nTuning report:")
print(REPORT_FILE)

print("\nTest set was NOT used for hyperparameter selection.")