"""
PEARLS AQI PREDICTOR
STEP 13 — XGBOOST HYPERPARAMETER TUNING

Purpose
-------
Tune XGBoost hyperparameters using the chronological validation set.

Methodology
-----------
- Train data is used for fitting.
- Validation data is used for hyperparameter selection.
- Test data is NEVER used for hyperparameter selection.
- The exact 101 ML features used by Step 12 are reused.
- Forecast horizons remain independent: t+1 ... t+72.
- MAE is the primary tuning metric.
- A representative set of horizons is used during tuning.
- After selecting the best configuration, all 72 horizon models
  are trained and evaluated.

Outputs
-------
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

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

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

# Step 12 report.
STEP12_REPORT_FILE = os.path.join(
    BASE_DIR,
    "reports",
    "xgboost_results.json",
)

# Step 10 baseline report.
BASELINE_REPORT_FILE = os.path.join(
    BASE_DIR,
    "reports",
    "baseline_results.json",
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
TUNING_HORIZONS = [1, 6, 12, 24, 48, 72]

# Number of candidate configurations.
MAX_CONFIGURATIONS = 16

EXPECTED_FEATURE_COUNT = 101


# ======================================================================
# HEADER
# ======================================================================

print("=" * 70)
print("PEARLS AQI PREDICTOR")
print("STEP 13 — XGBOOST HYPERPARAMETER TUNING")
print("=" * 70)

print(f"Base directory      : {BASE_DIR}")
print(f"Forecast horizon    : {FORECAST_HORIZON} hours")
print(f"Target              : {TARGET_COLUMN}")
print(f"Tuning horizons     : {TUNING_HORIZONS}")
print(f"Max configurations  : {MAX_CONFIGURATIONS}")


# ======================================================================
# INPUT FILE CHECK
# ======================================================================

print("\n" + "=" * 70)
print("INPUT FILE CHECK")
print("=" * 70)

required_files = [
    ("train.csv", TRAIN_FILE),
    ("validation.csv", VALIDATION_FILE),
    ("test.csv", TEST_FILE),
]

for name, path in required_files:

    if os.path.exists(path):
        print(f"{name:<18}: FOUND")
    else:
        print(f"{name:<18}: NOT FOUND")
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )


# Step 12 report is strongly recommended because it contains the
# authoritative feature protocol.
if os.path.exists(STEP12_REPORT_FILE):
    print(
        f"{'Step 12 report':<18}: FOUND"
    )
else:
    print(
        f"{'Step 12 report':<18}: NOT FOUND"
    )
    raise FileNotFoundError(
        "reports/xgboost_results.json is required so Step 13 "
        "can exactly reproduce the Step 12 101-feature protocol."
    )


# ======================================================================
# LOAD DATA
# ======================================================================

print("\n" + "=" * 70)
print("LOADING DATA")
print("=" * 70)

train_df = pd.read_csv(TRAIN_FILE)
validation_df = pd.read_csv(VALIDATION_FILE)
test_df = pd.read_csv(TEST_FILE)

print(f"Train rows          : {len(train_df):,}")
print(f"Validation rows     : {len(validation_df):,}")
print(f"Test rows           : {len(test_df):,}")


# ======================================================================
# TIMESTAMP PROCESSING
# ======================================================================

print("\n" + "=" * 70)
print("TIMESTAMP PROCESSING")
print("=" * 70)

for df in [
    train_df,
    validation_df,
    test_df,
]:

    if "time" not in df.columns:
        raise ValueError(
            "Required 'time' column is missing."
        )

    df["time"] = pd.to_datetime(
        df["time"],
        errors="raise",
    )

for name, df in [
    ("Train", train_df),
    ("Validation", validation_df),
    ("Test", test_df),
]:

    print(
        f"{name:<12}: "
        f"{df['time'].min()} -> {df['time'].max()}"
    )


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
        raise ValueError(
            f"{name}: duplicate timestamps detected."
        )

    if not df["time"].is_monotonic_increasing:
        raise ValueError(
            f"{name}: timestamps are not chronological."
        )

    if df.isna().any().any():
        missing_columns = (
            df.columns[
                df.isna().any()
            ].tolist()
        )

        raise ValueError(
            f"{name}: missing values detected in "
            f"{missing_columns[:10]}"
        )

    print(
        f"{name:<12}: PASS "
        f"(rows={len(df):,}, columns={len(df.columns)})"
    )


# ======================================================================
# FUTURE TARGET VALIDATION
# ======================================================================

print("\n" + "=" * 70)
print("FUTURE TARGET VALIDATION")
print("=" * 70)

future_target_columns = [
    f"{TARGET_COLUMN}_t_plus_{h}"
    for h in range(
        1,
        FORECAST_HORIZON + 1,
    )
]

for name, df in [
    ("Train", train_df),
    ("Validation", validation_df),
    ("Test", test_df),
]:

    missing = [
        col
        for col in future_target_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{name}: missing future target columns: "
            f"{missing[:10]}"
        )

    print(
        f"{name:<12}: "
        f"{len(future_target_columns)} future targets FOUND"
    )


# ======================================================================
# LOAD STEP 12 FEATURE PROTOCOL
# ======================================================================

print("\n" + "=" * 70)
print("LOADING STEP 12 FEATURE PROTOCOL")
print("=" * 70)

with open(
    STEP12_REPORT_FILE,
    "r",
    encoding="utf-8",
) as f:

    step12_report = json.load(f)


step12_features = step12_report.get(
    "features"
)

if not step12_features:
    raise ValueError(
        "Step 12 report does not contain the "
        "'features' list."
    )

print(
    f"Step 12 feature count : "
    f"{len(step12_features)}"
)

# Confirm Step 12 itself used the expected count.
if len(step12_features) != EXPECTED_FEATURE_COUNT:

    raise ValueError(
        "\nStep 12 feature protocol mismatch.\n"
        f"Expected: {EXPECTED_FEATURE_COUNT}\n"
        f"Found   : {len(step12_features)}\n"
        "Check reports/xgboost_results.json."
    )


# ======================================================================
# FEATURE PREPARATION
# ======================================================================

print("\n" + "=" * 70)
print("FEATURE PREPARATION")
print("=" * 70)

# IMPORTANT:
#
# We DO NOT reconstruct the feature list by excluding columns.
#
# Step 12's saved feature list is authoritative.
#
# This prevents accidental loss of one feature when the dataset
# contains target columns, metadata columns, or other generated
# columns.

feature_columns = list(step12_features)

print(
    f"Features required by Step 12 : "
    f"{len(feature_columns)}"
)

print(
    f"Expected feature count        : "
    f"{EXPECTED_FEATURE_COUNT}"
)


# ======================================================================
# FEATURE CONSISTENCY CHECK
# ======================================================================

print("\n" + "=" * 70)
print("FEATURE CONSISTENCY CHECK")
print("=" * 70)

for name, df in [
    ("Train", train_df),
    ("Validation", validation_df),
    ("Test", test_df),
]:

    missing_features = [
        col
        for col in feature_columns
        if col not in df.columns
    ]

    if missing_features:

        raise ValueError(
            f"\n{name}: missing Step 12 features.\n"
            f"Missing count: {len(missing_features)}\n"
            f"Missing features: {missing_features}"
        )

    print(
        f"{name:<12}: "
        f"{len(feature_columns)} features FOUND"
    )


# ======================================================================
# FEATURE MATRIX CREATION
# ======================================================================

print("\n" + "=" * 70)
print("CREATING FEATURE MATRICES")
print("=" * 70)

X_train = train_df[
    feature_columns
].astype(
    np.float32
)

X_validation = validation_df[
    feature_columns
].astype(
    np.float32
)

X_test = test_df[
    feature_columns
].astype(
    np.float32
)

print(
    f"X_train      : {X_train.shape}"
)

print(
    f"X_validation : {X_validation.shape}"
)

print(
    f"X_test       : {X_test.shape}"
)


# Final feature-count assertion.
if X_train.shape[1] != EXPECTED_FEATURE_COUNT:

    raise ValueError(
        "\nFeature-count mismatch.\n"
        f"Expected: {EXPECTED_FEATURE_COUNT}\n"
        f"Found   : {X_train.shape[1]}\n\n"
        "Step 13 must use the same 101-feature "
        "protocol as Step 12."
    )

if X_validation.shape[1] != EXPECTED_FEATURE_COUNT:

    raise ValueError(
        "Validation feature count does not equal 101."
    )

if X_test.shape[1] != EXPECTED_FEATURE_COUNT:

    raise ValueError(
        "Test feature count does not equal 101."
    )

print(
    "\nFeature-count verification: PASS"
)
print(
    "Exact Step 12 101-feature protocol: PASS"
)


# ======================================================================
# NUMERIC VALIDATION
# ======================================================================

print("\n" + "=" * 70)
print("FEATURE DATA VALIDATION")
print("=" * 70)

for name, X in [
    ("X_train", X_train),
    ("X_validation", X_validation),
    ("X_test", X_test),
]:

    if X.isna().any().any():

        raise ValueError(
            f"{name}: missing values detected."
        )

    if not np.isfinite(
        X.to_numpy()
    ).all():

        raise ValueError(
            f"{name}: infinite values detected."
        )

    print(
        f"{name:<15}: PASS"
    )


# ======================================================================
# HYPERPARAMETER SEARCH SPACE
# ======================================================================

print("\n" + "=" * 70)
print("HYPERPARAMETER SEARCH SPACE")
print("=" * 70)

parameter_grid = {
    "n_estimators": [
        300,
        500,
        700,
        900,
    ],

    "learning_rate": [
        0.03,
        0.05,
        0.08,
    ],

    "max_depth": [
        4,
        6,
        8,
    ],

    "min_child_weight": [
        1,
        3,
        5,
    ],

    "subsample": [
        0.8,
        1.0,
    ],

    "colsample_bytree": [
        0.8,
        1.0,
    ],

    "reg_alpha": [
        0.0,
        0.1,
    ],

    "reg_lambda": [
        1.0,
        5.0,
    ],
}

for parameter, values in parameter_grid.items():

    print(
        f"{parameter:<20}: {values}"
    )


# ======================================================================
# STEP 12 REFERENCE CONFIGURATION
# ======================================================================

print("\n" + "=" * 70)
print("STEP 12 REFERENCE CONFIGURATION")
print("=" * 70)

# Use the parameters recorded by Step 12 where available.
step12_parameters = step12_report.get(
    "xgboost_parameters"
)

if not step12_parameters:

    raise ValueError(
        "Step 12 report does not contain "
        "'xgboost_parameters'."
    )


baseline_config = {
    "n_estimators": int(
        step12_parameters["n_estimators"]
    ),

    "learning_rate": float(
        step12_parameters["learning_rate"]
    ),

    "max_depth": int(
        step12_parameters["max_depth"]
    ),

    "min_child_weight": int(
        step12_parameters["min_child_weight"]
    ),

    "subsample": float(
        step12_parameters["subsample"]
    ),

    "colsample_bytree": float(
        step12_parameters["colsample_bytree"]
    ),

    "reg_alpha": float(
        step12_parameters["reg_alpha"]
    ),

    "reg_lambda": float(
        step12_parameters["reg_lambda"]
    ),
}

print(
    json.dumps(
        baseline_config,
        indent=2,
    )
)


# ======================================================================
# CONTROLLED CONFIGURATION GENERATION
# ======================================================================

print("\n" + "=" * 70)
print("GENERATING CONTROLLED CONFIGURATIONS")
print("=" * 70)

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

print(
    f"Total possible combinations : "
    f"{len(all_combinations):,}"
)

rng = np.random.default_rng(
    RANDOM_STATE
)


def combination_to_dict(combo):

    return {
        "n_estimators": int(
            combo[0]
        ),

        "learning_rate": float(
            combo[1]
        ),

        "max_depth": int(
            combo[2]
        ),

        "min_child_weight": int(
            combo[3]
        ),

        "subsample": float(
            combo[4]
        ),

        "colsample_bytree": float(
            combo[5]
        ),

        "reg_alpha": float(
            combo[6]
        ),

        "reg_lambda": float(
            combo[7]
        ),
    }


baseline_tuple = (
    baseline_config["n_estimators"],
    baseline_config["learning_rate"],
    baseline_config["max_depth"],
    baseline_config["min_child_weight"],
    baseline_config["subsample"],
    baseline_config["colsample_bytree"],
    baseline_config["reg_alpha"],
    baseline_config["reg_lambda"],
)


# Remove reference configuration from random candidates.
remaining = [
    combo
    for combo in all_combinations
    if combo != baseline_tuple
]

rng.shuffle(
    remaining
)


selected_combinations = [
    baseline_tuple
] + remaining[
    : MAX_CONFIGURATIONS - 1
]


parameter_configs = [
    combination_to_dict(combo)
    for combo in selected_combinations
]


print(
    f"Candidate configurations : "
    f"{len(parameter_configs)}"
)

print(
    f"Tuning horizons          : "
    f"{TUNING_HORIZONS}"
)


# ======================================================================
# PRINT CONFIGURATIONS
# ======================================================================

print("\nSelected configurations:")

for index, params in enumerate(
    parameter_configs,
    start=1,
):

    label = (
        "REFERENCE"
        if index == 1
        else "CANDIDATE"
    )

    print(
        f"\nConfiguration {index:02d} "
        f"({label})"
    )

    print(
        json.dumps(
            params,
            indent=2,
        )
    )


# ======================================================================
# HELPER FUNCTION
# ======================================================================

def create_xgboost_model(params):
    """
    Create an XGBoost model using the selected parameters.
    """

    return XGBRegressor(
        n_estimators=params[
            "n_estimators"
        ],

        learning_rate=params[
            "learning_rate"
        ],

        max_depth=params[
            "max_depth"
        ],

        min_child_weight=params[
            "min_child_weight"
        ],

        subsample=params[
            "subsample"
        ],

        colsample_bytree=params[
            "colsample_bytree"
        ],

        reg_alpha=params[
            "reg_alpha"
        ],

        reg_lambda=params[
            "reg_lambda"
        ],

        objective="reg:squarederror",

        random_state=RANDOM_STATE,

        n_jobs=-1,

        tree_method="hist",
    )


# ======================================================================
# HYPERPARAMETER TUNING
# ======================================================================

print("\n" + "=" * 70)
print("STARTING HYPERPARAMETER TUNING")
print("=" * 70)

tuning_results = []

overall_tuning_start = time.time()


for config_index, params in enumerate(
    parameter_configs,
    start=1,
):

    print("\n" + "-" * 70)

    print(
        f"CONFIGURATION "
        f"{config_index}/{len(parameter_configs)}"
    )

    print("-" * 70)

    print(
        json.dumps(
            params,
            indent=2,
        )
    )

    config_start = time.time()

    horizon_results = []

    for horizon in TUNING_HORIZONS:

        target_column = (
            f"{TARGET_COLUMN}_t_plus_{horizon}"
        )

        y_train = train_df[
            target_column
        ].astype(
            np.float32
        )

        y_validation = validation_df[
            target_column
        ].astype(
            np.float32
        )

        model = create_xgboost_model(
            params
        )

        model.fit(
            X_train,
            y_train,
            verbose=False,
        )

        validation_pred = model.predict(
            X_validation
        )

        validation_mae = (
            mean_absolute_error(
                y_validation,
                validation_pred,
            )
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

                "validation_mae": float(
                    validation_mae
                ),

                "validation_rmse": float(
                    validation_rmse
                ),
            }
        )

        print(
            f"{horizon:02d}h | "
            f"Validation MAE = "
            f"{validation_mae:8.3f} | "
            f"Validation RMSE = "
            f"{validation_rmse:8.3f}"
        )


    mean_mae = float(
        np.mean(
            [
                result[
                    "validation_mae"
                ]
                for result in horizon_results
            ]
        )
    )

    mean_rmse = float(
        np.mean(
            [
                result[
                    "validation_rmse"
                ]
                for result in horizon_results
            ]
        )
    )

    elapsed = (
        time.time()
        - config_start
    )

    result = {
        "configuration_id": config_index,

        "configuration_type": (
            "reference"
            if config_index == 1
            else "candidate"
        ),

        "parameters": params,

        "mean_validation_mae": mean_mae,

        "mean_validation_rmse": mean_rmse,

        "horizon_results": horizon_results,

        "training_time_seconds": round(
            elapsed,
            2,
        ),
    }

    tuning_results.append(
        result
    )

    print(
        f"\nConfiguration mean "
        f"validation MAE : "
        f"{mean_mae:.3f}"
    )

    print(
        f"Configuration mean "
        f"validation RMSE: "
        f"{mean_rmse:.3f}"
    )

    print(
        f"Configuration time: "
        f"{elapsed:.1f}s"
    )


# ======================================================================
# SELECT BEST CONFIGURATION
# ======================================================================

best_result = min(
    tuning_results,
    key=lambda x: x[
        "mean_validation_mae"
    ],
)

best_params = best_result[
    "parameters"
]

reference_result = next(
    result
    for result in tuning_results
    if result[
        "configuration_type"
    ] == "reference"
)

reference_mae = (
    reference_result[
        "mean_validation_mae"
    ]
)

reference_rmse = (
    reference_result[
        "mean_validation_rmse"
    ]
)

best_tuning_mae = (
    best_result[
        "mean_validation_mae"
    ]
)

best_tuning_rmse = (
    best_result[
        "mean_validation_rmse"
    ]
)

improvement = (
    reference_mae
    - best_tuning_mae
)

improvement_percent = (
    improvement
    / reference_mae
    * 100
    if reference_mae != 0
    else 0
)

total_tuning_time = (
    time.time()
    - overall_tuning_start
)


# ======================================================================
# TUNING SUMMARY
# ======================================================================

print("\n" + "=" * 70)
print("HYPERPARAMETER TUNING SUMMARY")
print("=" * 70)

ranking = sorted(
    tuning_results,
    key=lambda x: x[
        "mean_validation_mae"
    ],
)

print(
    f"{'Rank':<7}"
    f"{'Config':<8}"
    f"{'Type':<12}"
    f"{'MAE':>12}"
    f"{'RMSE':>12}"
)

print("-" * 55)

for rank, result in enumerate(
    ranking,
    start=1,
):

    print(
        f"{rank:<7}"
        f"{result['configuration_id']:<8}"
        f"{result['configuration_type']:<12}"
        f"{result['mean_validation_mae']:>12.3f}"
        f"{result['mean_validation_rmse']:>12.3f}"
    )


# ======================================================================
# BEST CONFIGURATION
# ======================================================================

print("\n" + "=" * 70)
print("BEST XGBOOST CONFIGURATION")
print("=" * 70)

print(
    json.dumps(
        best_params,
        indent=2,
    )
)

print(
    f"\nBest tuning validation MAE : "
    f"{best_tuning_mae:.3f}"
)

print(
    f"Best tuning validation RMSE: "
    f"{best_tuning_rmse:.3f}"
)

print(
    f"Reference validation MAE    : "
    f"{reference_mae:.3f}"
)

print(
    f"Tuning MAE improvement      : "
    f"{improvement:.3f}"
)

print(
    f"Tuning improvement (%)      : "
    f"{improvement_percent:.2f}%"
)


if improvement > 0:

    print(
        "\nTuned configuration: "
        "IMPROVED over Step 12 reference"
    )

elif improvement < 0:

    print(
        "\nTuned configuration: "
        "WORSE than Step 12 reference"
    )

else:

    print(
        "\nTuned configuration: "
        "NO CHANGE from Step 12 reference"
    )


# ======================================================================
# TRAIN FINAL 72 MODELS
# ======================================================================

print("\n" + "=" * 70)
print("TRAINING FINAL TUNED XGBOOST MODELS")
print("=" * 70)

os.makedirs(
    MODEL_DIR,
    exist_ok=True,
)

final_results = []

final_training_start = time.time()


for horizon in range(
    1,
    FORECAST_HORIZON + 1,
):

    target_column = (
        f"{TARGET_COLUMN}_t_plus_{horizon}"
    )

    y_train = train_df[
        target_column
    ].astype(
        np.float32
    )

    y_validation = validation_df[
        target_column
    ].astype(
        np.float32
    )

    y_test = test_df[
        target_column
    ].astype(
        np.float32
    )

    model_start = time.time()

    model = create_xgboost_model(
        best_params
    )

    model.fit(
        X_train,
        y_train,
        verbose=False,
    )

    validation_pred = model.predict(
        X_validation
    )

    test_pred = model.predict(
        X_test
    )

    validation_mae = (
        mean_absolute_error(
            y_validation,
            validation_pred,
        )
    )

    validation_rmse = np.sqrt(
        mean_squared_error(
            y_validation,
            validation_pred,
        )
    )

    test_mae = (
        mean_absolute_error(
            y_test,
            test_pred,
        )
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

    model.save_model(
        model_path
    )

    elapsed = (
        time.time()
        - model_start
    )

    final_results.append(
        {
            "horizon": horizon,

            "validation_mae": float(
                validation_mae
            ),

            "validation_rmse": float(
                validation_rmse
            ),

            "test_mae": float(
                test_mae
            ),

            "test_rmse": float(
                test_rmse
            ),

            "training_time_seconds": round(
                elapsed,
                2,
            ),

            "model_path": model_path,
        }
    )

    print(
        f"{horizon:02d}h | "
        f"Validation MAE = "
        f"{validation_mae:8.3f} | "
        f"Validation RMSE = "
        f"{validation_rmse:8.3f} | "
        f"Test MAE = "
        f"{test_mae:8.3f} | "
        f"Test RMSE = "
        f"{test_rmse:8.3f} | "
        f"Time = "
        f"{elapsed:6.1f}s"
    )


final_training_time = (
    time.time()
    - final_training_start
)


# ======================================================================
# FINAL HORIZON SUMMARY
# ======================================================================

print("\n" + "=" * 70)
print("TUNED XGBOOST SUMMARY BY HORIZON")
print("=" * 70)

selected_horizons = [
    1,
    6,
    12,
    24,
    48,
    72,
]

print(
    f"{'Metric':<25}"
    f"{'1h':>10}"
    f"{'6h':>10}"
    f"{'12h':>10}"
    f"{'24h':>10}"
    f"{'48h':>10}"
    f"{'72h':>10}"
)

print("-" * 85)

validation_by_horizon = {
    result["horizon"]:
        result["validation_mae"]
    for result in final_results
}

test_by_horizon = {
    result["horizon"]:
        result["test_mae"]
    for result in final_results
}

print(
    f"{'Validation MAE':<25}"
    + "".join(
        f"{validation_by_horizon[h]:>10.3f}"
        for h in selected_horizons
    )
)

print(
    f"{'Test MAE':<25}"
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
            result[
                "validation_mae"
            ]
            for result in final_results
        ]
    )
)

mean_validation_rmse = float(
    np.mean(
        [
            result[
                "validation_rmse"
            ]
            for result in final_results
        ]
    )
)

mean_test_mae = float(
    np.mean(
        [
            result[
                "test_mae"
            ]
            for result in final_results
        ]
    )
)

mean_test_rmse = float(
    np.mean(
        [
            result[
                "test_rmse"
            ]
            for result in final_results
        ]
    )
)


best_validation_horizon = min(
    final_results,
    key=lambda x: x[
        "validation_mae"
    ],
)

best_test_horizon = min(
    final_results,
    key=lambda x: x[
        "test_mae"
    ],
)


# ======================================================================
# SEASONAL BASELINE REFERENCE
# ======================================================================

print("\n" + "=" * 70)
print("SEASONAL BASELINE REFERENCE")
print("=" * 70)

seasonal_baseline_validation_mae = None

if os.path.exists(
    BASELINE_REPORT_FILE
):

    try:

        with open(
            BASELINE_REPORT_FILE,
            "r",
            encoding="utf-8",
        ) as f:

            baseline_report = json.load(f)

        validation_results = (
            baseline_report
            .get(
                "results",
                {}
            )
            .get(
                "validation",
                []
            )
        )

        seasonal_result = next(
            (
                result
                for result
                in validation_results
                if result.get("model")
                == "Seasonal Persistence"
            ),
            None,
        )

        if seasonal_result is not None:

            overall = seasonal_result.get(
                "overall",
                {}
            )

            if "mae" in overall:

                seasonal_baseline_validation_mae = float(
                    overall["mae"]
                )

    except Exception as exc:

        print(
            "Warning: unable to read seasonal "
            f"baseline MAE: {exc}"
        )


# Fallback only if the baseline report is unavailable
# or does not contain the expected metric.
#
# The known value from the user's Step 10 output is
# approximately 20.231, but we prefer reading the actual
# report whenever possible.
if seasonal_baseline_validation_mae is None:

    seasonal_baseline_validation_mae = 20.231

    print(
        "Using fallback seasonal baseline "
        "validation MAE: 20.231"
    )

else:

    print(
        "Seasonal baseline validation MAE: "
        f"{seasonal_baseline_validation_mae:.3f}"
    )


beats_seasonal_baseline = (
    mean_validation_mae
    < seasonal_baseline_validation_mae
)

baseline_difference = (
    seasonal_baseline_validation_mae
    - mean_validation_mae
)

baseline_improvement_percent = (
    baseline_difference
    / seasonal_baseline_validation_mae
    * 100
    if seasonal_baseline_validation_mae != 0
    else 0
)


# ======================================================================
# FINAL SUMMARY
# ======================================================================

print("\n" + "=" * 70)
print("TUNED XGBOOST FINAL SUMMARY")
print("=" * 70)

print(
    f"Features used          : "
    f"{len(feature_columns)}"
)

print(
    f"Mean validation MAE    : "
    f"{mean_validation_mae:.3f}"
)

print(
    f"Mean validation RMSE   : "
    f"{mean_validation_rmse:.3f}"
)

print(
    f"Mean test MAE          : "
    f"{mean_test_mae:.3f}"
)

print(
    f"Mean test RMSE         : "
    f"{mean_test_rmse:.3f}"
)

print()

print(
    f"Best validation horizon: "
    f"{best_validation_horizon['horizon']}h"
)

print(
    f"Best validation MAE    : "
    f"{best_validation_horizon['validation_mae']:.3f}"
)

print()

print(
    f"Best test horizon      : "
    f"{best_test_horizon['horizon']}h"
)

print(
    f"Best test MAE          : "
    f"{best_test_horizon['test_mae']:.3f}"
)

print()

print(
    f"Seasonal baseline MAE  : "
    f"{seasonal_baseline_validation_mae:.3f}"
)

print(
    f"Baseline difference    : "
    f"{baseline_difference:.3f}"
)

print(
    f"Baseline improvement % : "
    f"{baseline_improvement_percent:.2f}%"
)

if beats_seasonal_baseline:

    print(
        "Tuned XGBoost baseline "
        "comparison: BETTER"
    )

else:

    print(
        "Tuned XGBoost baseline "
        "comparison: NOT BETTER"
    )


# ======================================================================
# SAVE REPORT
# ======================================================================

print("\n" + "=" * 70)
print("SAVING TUNING REPORT")
print("=" * 70)

os.makedirs(
    os.path.dirname(
        REPORT_FILE
    ),
    exist_ok=True,
)


# selected_configuration deliberately uses a simple,
# machine-readable label.
if best_result[
    "configuration_type"
] == "reference":

    selected_configuration = "reference"

else:

    selected_configuration = "tuned"


report = {

    "step": 13,

    "description":
        "XGBoost hyperparameter tuning",

    "target":
        TARGET_COLUMN,

    "forecast_horizon":
        FORECAST_HORIZON,

    "features_used":
        len(feature_columns),

    "feature_columns":
        feature_columns,

    "feature_protocol":
        "Exact Step 12 feature list",

    "tuning_horizons":
        TUNING_HORIZONS,

    "random_state":
        RANDOM_STATE,

    "candidate_configurations":
        len(parameter_configs),

    "selected_configuration":
        selected_configuration,

    "selected_configuration_id":
        best_result[
            "configuration_id"
        ],

    "reference_configuration":
        baseline_config,

    "reference_validation_mae":
        reference_mae,

    "reference_validation_rmse":
        reference_rmse,

    "best_tuning_parameters":
        best_params,

    "best_parameters":
        best_params,

    "best_tuning_validation_mae":
        best_tuning_mae,

    "best_tuning_validation_rmse":
        best_tuning_rmse,

    "tuning_improvement_mae":
        improvement,

    "tuning_improvement_percent":
        improvement_percent,

    "tuning_results":
        tuning_results,

    "final_results":
        final_results,

    "final_summary": {

        "mean_validation_mae":
            mean_validation_mae,

        "mean_validation_rmse":
            mean_validation_rmse,

        "mean_test_mae":
            mean_test_mae,

        "mean_test_rmse":
            mean_test_rmse,

        "best_validation_horizon":
            best_validation_horizon[
                "horizon"
            ],

        "best_validation_mae":
            best_validation_horizon[
                "validation_mae"
            ],

        "best_test_horizon":
            best_test_horizon[
                "horizon"
            ],

        "best_test_mae":
            best_test_horizon[
                "test_mae"
            ],

        "seasonal_baseline_validation_mae":
            seasonal_baseline_validation_mae,

        "beats_seasonal_baseline":
            beats_seasonal_baseline,

        "baseline_difference":
            baseline_difference,

        "baseline_improvement_percent":
            baseline_improvement_percent,
    },

    "timing": {

        "tuning_seconds":
            round(
                total_tuning_time,
                2,
            ),

        "final_training_seconds":
            round(
                final_training_time,
                2,
            ),

        "total_seconds":
            round(
                total_tuning_time
                + final_training_time,
                2,
            ),
    },

    "model_directory":
        MODEL_DIR,

    "test_used_for_tuning":
        False,
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

print(
    f"\nReport saved to:"
)

print(
    REPORT_FILE
)

print("\n" + "=" * 70)
print("STEP 13 COMPLETE")
print("=" * 70)

print(
    f"Tuning configurations : "
    f"{len(parameter_configs)}"
)

print(
    f"Tuning horizons       : "
    f"{len(TUNING_HORIZONS)}"
)

print(
    f"Final models trained  : "
    f"{FORECAST_HORIZON}"
)

print(
    f"Features used         : "
    f"{len(feature_columns)}"
)

print(
    f"Best tuning MAE       : "
    f"{best_tuning_mae:.3f}"
)

print(
    f"Final validation MAE  : "
    f"{mean_validation_mae:.3f}"
)

print(
    f"Mean test MAE         : "
    f"{mean_test_mae:.3f}"
)

print(
    f"Mean test RMSE        : "
    f"{mean_test_rmse:.3f}"
)

print(
    f"\nFinal model directory:"
)

print(
    MODEL_DIR
)

print(
    f"\nTuning report:"
)

print(
    REPORT_FILE
)

print(
    "\nTest set was NOT used for "
    "hyperparameter selection."
)