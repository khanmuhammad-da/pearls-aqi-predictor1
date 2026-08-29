"""
PEARLS AQI PREDICTOR
STEP 13 — XGBOOST HYPERPARAMETER TUNING

Purpose
-------
Tune XGBoost hyperparameters using the chronological validation set.

Methodology
-----------
- Train data is used for model fitting.
- Validation data is used for hyperparameter selection.
- Test data is NOT used for hyperparameter selection.
- The exact same 101 ML features from Step 12 are preserved.
- Current us_aqi is intentionally retained because Step 12 used it
  as an input feature.
- Forecast horizons remain independent: t+1 ... t+72.
- MAE is the primary tuning metric.
- A representative set of horizons is used during tuning.
- The Step 12 configuration is always included as the reference.
- A controlled random search evaluates a fixed number of candidate
  configurations.
- After selecting the best configuration, all 72 horizon models
  are trained and evaluated.

Outputs
-------
Models:
    models/artifacts/xgboost_tuned/

Report:
    reports/xgboost_tuning_results.json

Important
---------
The test set is only evaluated after hyperparameters have been selected.
It must not influence model selection.
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

STEP12_REPORT_FILE = os.path.join(
    BASE_DIR,
    "reports",
    "xgboost_results.json",
)

BASELINE_REPORT_FILE = os.path.join(
    BASE_DIR,
    "reports",
    "baseline_results.json",
)

TARGET_COLUMN = "us_aqi"

FORECAST_HORIZON = 72

RANDOM_STATE = 42

# Horizons used during hyperparameter tuning.
#
# These are representative points across the 72-hour forecast range.
TUNING_HORIZONS = [
    1,
    6,
    12,
    24,
    48,
    72,
]

# Number of candidate configurations.
#
# One configuration is always the Step 12 reference configuration.
MAX_CONFIGURATIONS = 16

# Minimum improvement required before calling tuning a meaningful
# improvement over the Step 12 reference configuration.
MIN_IMPROVEMENT_PERCENT = 0.0


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

required_files = [
    ("train.csv", TRAIN_FILE),
    ("validation.csv", VALIDATION_FILE),
    ("test.csv", TEST_FILE),
    ("Step 12 report", STEP12_REPORT_FILE),
    ("baseline report", BASELINE_REPORT_FILE),
]

for name, path in required_files:

    if os.path.exists(path):
        print(f"{name:<20}: FOUND")
    else:
        print(f"{name:<20}: NOT FOUND")
        raise FileNotFoundError(
            f"Required file not found: {path}"
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

print(f"Train      : {len(train_df):,} rows")
print(f"Validation : {len(validation_df):,} rows")
print(f"Test       : {len(test_df):,} rows")


# ======================================================================
# LOAD STEP 12 REPORT
# ======================================================================

print("\n" + "=" * 70)
print("LOADING STEP 12 REFERENCE")
print("=" * 70)

with open(
    STEP12_REPORT_FILE,
    "r",
    encoding="utf-8",
) as f:

    step12_report = json.load(f)

step12_features = step12_report.get(
    "features",
    []
)

if not step12_features:

    raise ValueError(
        "Step 12 report does not contain a valid "
        "'features' list."
    )

print(
    f"Step 12 reported features : "
    f"{len(step12_features)}"
)

print(
    f"Step 12 reported model     : "
    f"{step12_report.get('model', 'UNKNOWN')}"
)


# ======================================================================
# TIMESTAMP CONVERSION
# ======================================================================

print("\n" + "=" * 70)
print("TIMESTAMP CONVERSION")
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
        df["time"]
    )

print("Timestamp conversion: PASS")


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
            f"{missing_columns}"
        )

    print(
        f"{name:<12}: PASS"
    )


# ======================================================================
# FORECAST TARGET VALIDATION
# ======================================================================

print("\n" + "=" * 70)
print("FORECAST TARGET VALIDATION")
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

    missing_targets = [
        col
        for col in future_target_columns
        if col not in df.columns
    ]

    if missing_targets:

        raise ValueError(
            f"{name}: missing future target columns. "
            f"Examples: {missing_targets[:5]}"
        )

    print(
        f"{name:<12}: "
        f"{len(future_target_columns)} target columns found"
    )


# ======================================================================
# FEATURE PREPARATION
# ======================================================================

print("\n" + "=" * 70)
print("FEATURE PREPARATION")
print("=" * 70)

"""
IMPORTANT

Step 12 used 101 features, including current us_aqi.

Therefore we do NOT exclude TARGET_COLUMN here.

We only exclude:
    - time
    - future target columns

This preserves the exact Step 12 feature protocol.
"""

excluded_columns = (
    ["time"]
    + future_target_columns
)

feature_columns = [
    col
    for col in train_df.columns
    if col not in excluded_columns
]

print(
    f"Total train columns : "
    f"{len(train_df.columns)}"
)

print(
    f"Future targets      : "
    f"{len(future_target_columns)}"
)

print(
    f"Usable ML features  : "
    f"{len(feature_columns)}"
)

print(
    f"Expected Step 12    : "
    f"{len(step12_features)}"
)


# ======================================================================
# FEATURE COUNT CHECK
# ======================================================================

print("\n" + "=" * 70)
print("FEATURE CONSISTENCY CHECK")
print("=" * 70)

if len(feature_columns) != len(step12_features):

    raise ValueError(
        "\nFeature count mismatch.\n"
        f"Step 12: {len(step12_features)} features\n"
        f"Step 13: {len(feature_columns)} features\n\n"
        "Step 13 must use the exact same feature set "
        "as Step 12."
    )

print(
    "Feature count: PASS"
)


# ======================================================================
# FEATURE NAME CHECK
# ======================================================================

step12_feature_set = set(
    step12_features
)

current_feature_set = set(
    feature_columns
)

missing_from_step13 = sorted(
    step12_feature_set
    - current_feature_set
)

extra_in_step13 = sorted(
    current_feature_set
    - step12_feature_set
)

if missing_from_step13:

    raise ValueError(
        "Step 13 is missing features used by Step 12:\n"
        + "\n".join(
            missing_from_step13
        )
    )

if extra_in_step13:

    raise ValueError(
        "Step 13 contains features not used by Step 12:\n"
        + "\n".join(
            extra_in_step13
        )
    )

print(
    "Feature names: PASS"
)

print(
    "Exact Step 12 feature set preserved."
)


# ======================================================================
# PRESERVE STEP 12 FEATURE ORDER
# ======================================================================

"""
Even if the feature sets match, explicitly use the Step 12 order.

This protects against column-order changes in the CSV.
"""

feature_columns = [
    col
    for col in step12_features
]


# ======================================================================
# VERIFY FEATURES EXIST IN ALL DATASETS
# ======================================================================

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
            f"{name}: missing features:\n"
            + "\n".join(
                missing_features
            )
        )

print(
    "All 101 features exist in train, validation, and test."
)


# ======================================================================
# CREATE FEATURE MATRICES
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

if X_train.isna().any().any():

    raise ValueError(
        "Missing values found in X_train."
    )

if X_validation.isna().any().any():

    raise ValueError(
        "Missing values found in X_validation."
    )

if X_test.isna().any().any():

    raise ValueError(
        "Missing values found in X_test."
    )

print(
    "Feature missing-value check: PASS"
)


# ======================================================================
# STEP 12 REFERENCE CONFIGURATION
# ======================================================================

print("\n" + "=" * 70)
print("STEP 12 REFERENCE CONFIGURATION")
print("=" * 70)

step12_parameters = step12_report.get(
    "xgboost_parameters"
)

if not step12_parameters:

    raise ValueError(
        "Step 12 report does not contain "
        "'xgboost_parameters'."
    )

print(
    json.dumps(
        step12_parameters,
        indent=2
    )
)


# ======================================================================
# EXTRACT ONLY TUNABLE PARAMETERS
# ======================================================================

baseline_config = (
    int(
        step12_parameters[
            "n_estimators"
        ]
    ),
    float(
        step12_parameters[
            "learning_rate"
        ]
    ),
    int(
        step12_parameters[
            "max_depth"
        ]
    ),
    int(
        step12_parameters[
            "min_child_weight"
        ]
    ),
    float(
        step12_parameters[
            "subsample"
        ]
    ),
    float(
        step12_parameters[
            "colsample_bytree"
        ]
    ),
    float(
        step12_parameters[
            "reg_alpha"
        ]
    ),
    float(
        step12_parameters[
            "reg_lambda"
        ]
    ),
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


baseline_config_dict = (
    combination_to_dict(
        baseline_config
    )
)


# ======================================================================
# HYPERPARAMETER SEARCH SPACE
# ======================================================================

print("\n" + "=" * 70)
print("CONTROLLED RANDOM SEARCH SPACE")
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

print(
    json.dumps(
        parameter_grid,
        indent=2
    )
)


# ======================================================================
# CONTROLLED RANDOM CONFIGURATION GENERATION
# ======================================================================

print("\n" + "=" * 70)
print("GENERATING CONFIGURATIONS")
print("=" * 70)

all_combinations = list(
    product(
        parameter_grid[
            "n_estimators"
        ],
        parameter_grid[
            "learning_rate"
        ],
        parameter_grid[
            "max_depth"
        ],
        parameter_grid[
            "min_child_weight"
        ],
        parameter_grid[
            "subsample"
        ],
        parameter_grid[
            "colsample_bytree"
        ],
        parameter_grid[
            "reg_alpha"
        ],
        parameter_grid[
            "reg_lambda"
        ],
    )
)

print(
    f"Full search space : "
    f"{len(all_combinations):,} configurations"
)

if MAX_CONFIGURATIONS > len(
    all_combinations
):

    raise ValueError(
        "MAX_CONFIGURATIONS exceeds "
        "the available search space."
    )

rng = np.random.default_rng(
    RANDOM_STATE
)

remaining = [
    combo
    for combo in all_combinations
    if combo != baseline_config
]

rng.shuffle(
    remaining
)

selected_combinations = [
    baseline_config
] + remaining[
    : MAX_CONFIGURATIONS - 1
]

parameter_configs = [
    combination_to_dict(
        combo
    )
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

print(
    "\nConfiguration 1 is guaranteed "
    "to be the Step 12 reference."
)


# ======================================================================
# TUNING
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
        f"{config_index}/"
        f"{len(parameter_configs)}"
    )

    print("-" * 70)

    print(
        json.dumps(
            params,
            indent=2
        )
    )

    config_start = time.time()

    horizon_results = []

    for horizon in TUNING_HORIZONS:

        target_column = (
            f"{TARGET_COLUMN}"
            f"_t_plus_{horizon}"
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

        model = XGBRegressor(

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

        "configuration_id":
            config_index,

        "is_step12_reference":
            params == baseline_config_dict,

        "parameters":
            params,

        "mean_validation_mae":
            mean_mae,

        "mean_validation_rmse":
            mean_rmse,

        "horizon_results":
            horizon_results,

        "training_time_seconds":
            round(
                elapsed,
                2
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
# SELECT BEST TUNING CONFIGURATION
# ======================================================================

best_result = min(
    tuning_results,
    key=lambda x:
        x["mean_validation_mae"],
)

best_tuning_params = (
    best_result["parameters"]
)

total_tuning_time = (
    time.time()
    - overall_tuning_start
)


# ======================================================================
# REFERENCE RESULT
# ======================================================================

reference_result = next(
    result
    for result in tuning_results
    if result[
        "is_step12_reference"
    ]
)

reference_mae = float(
    reference_result[
        "mean_validation_mae"
    ]
)

reference_rmse = float(
    reference_result[
        "mean_validation_rmse"
    ]
)


# ======================================================================
# TUNING IMPROVEMENT
# ======================================================================

raw_improvement = (
    reference_mae
    - best_result[
        "mean_validation_mae"
    ]
)

if reference_mae != 0:

    improvement_percent = (
        raw_improvement
        / reference_mae
        * 100
    )

else:

    improvement_percent = 0.0


# ======================================================================
# SELECT FINAL PARAMETERS
# ======================================================================

"""
The best candidate is normally used.

MIN_IMPROVEMENT_PERCENT can be raised if you want a stricter
deployment rule.

At 0.0%, any measurable improvement is accepted.
"""

if (
    improvement_percent
    >= MIN_IMPROVEMENT_PERCENT
):

    final_params = (
        best_tuning_params
    )

    selected_configuration = (
        "tuned"
    )

else:

    final_params = (
        baseline_config_dict
    )

    selected_configuration = (
        "step12_reference"
    )


# ======================================================================
# TUNING SUMMARY
# ======================================================================

print("\n" + "=" * 70)
print("HYPERPARAMETER TUNING SUMMARY")
print("=" * 70)

ranking = sorted(
    tuning_results,
    key=lambda x:
        x["mean_validation_mae"],
)

print(
    f"{'Rank':<6}"
    f"{'Config':<8}"
    f"{'MAE':>12}"
    f"{'RMSE':>12}"
    f"{'Step12':>10}"
)

print("-" * 52)

for rank, result in enumerate(
    ranking,
    start=1,
):

    reference_marker = (
        "YES"
        if result[
            "is_step12_reference"
        ]
        else ""
    )

    print(
        f"{rank:<6}"
        f"{result['configuration_id']:<8}"
        f"{result['mean_validation_mae']:>12.3f}"
        f"{result['mean_validation_rmse']:>12.3f}"
        f"{reference_marker:>10}"
    )


# ======================================================================
# BEST CONFIGURATION
# ======================================================================

print("\n" + "=" * 70)
print("BEST TUNING CONFIGURATION")
print("=" * 70)

print(
    json.dumps(
        best_tuning_params,
        indent=2
    )
)

print(
    f"\nBest tuning MAE : "
    f"{best_result['mean_validation_mae']:.3f}"
)

print(
    f"Step 12 MAE     : "
    f"{reference_mae:.3f}"
)

print(
    f"Improvement      : "
    f"{raw_improvement:.3f}"
)

print(
    f"Improvement (%)  : "
    f"{improvement_percent:.2f}%"
)

print(
    f"Selected model   : "
    f"{selected_configuration}"
)


# ======================================================================
# LOAD SEASONAL BASELINE
# ======================================================================

print("\n" + "=" * 70)
print("LOADING SEASONAL BASELINE")
print("=" * 70)

with open(
    BASELINE_REPORT_FILE,
    "r",
    encoding="utf-8",
) as f:

    baseline_report = json.load(f)

seasonal_baseline_result = next(
    result
    for result
    in baseline_report[
        "results"
    ][
        "validation"
    ]
    if result[
        "model"
    ] == "Seasonal Persistence"
)

SEASONAL_BASELINE_VALIDATION_MAE = (
    float(
        seasonal_baseline_result[
            "overall"
        ][
            "mae"
        ]
    )
)

SEASONAL_BASELINE_VALIDATION_RMSE = (
    float(
        seasonal_baseline_result[
            "overall"
        ][
            "rmse"
        ]
    )
)

print(
    f"Seasonal Persistence "
    f"validation MAE : "
    f"{SEASONAL_BASELINE_VALIDATION_MAE:.3f}"
)

print(
    f"Seasonal Persistence "
    f"validation RMSE: "
    f"{SEASONAL_BASELINE_VALIDATION_RMSE:.3f}"
)


# ======================================================================
# TRAIN FINAL 72 MODELS
# ======================================================================

print("\n" + "=" * 70)
print("TRAINING FINAL XGBOOST MODELS")
print("=" * 70)

print(
    f"Using configuration: "
    f"{selected_configuration}"
)

print(
    json.dumps(
        final_params,
        indent=2
    )
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

final_results = []

final_training_start = (
    time.time()
)

for horizon in range(
    1,
    FORECAST_HORIZON + 1,
):

    target_column = (
        f"{TARGET_COLUMN}"
        f"_t_plus_{horizon}"
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

    model = XGBRegressor(

        n_estimators=final_params[
            "n_estimators"
        ],

        learning_rate=final_params[
            "learning_rate"
        ],

        max_depth=final_params[
            "max_depth"
        ],

        min_child_weight=final_params[
            "min_child_weight"
        ],

        subsample=final_params[
            "subsample"
        ],

        colsample_bytree=final_params[
            "colsample_bytree"
        ],

        reg_alpha=final_params[
            "reg_alpha"
        ],

        reg_lambda=final_params[
            "reg_lambda"
        ],

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
            "horizon":
                horizon,

            "validation_mae":
                float(
                    validation_mae
                ),

            "validation_rmse":
                float(
                    validation_rmse
                ),

            "test_mae":
                float(
                    test_mae
                ),

            "test_rmse":
                float(
                    test_rmse
                ),

            "training_time_seconds":
                round(
                    elapsed,
                    2
                ),

            "model_path":
                model_path,
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
# FINAL HORIZON DICTIONARIES
# ======================================================================

validation_by_horizon = {
    result[
        "horizon"
    ]: result[
        "validation_mae"
    ]
    for result
    in final_results
}

test_by_horizon = {
    result[
        "horizon"
    ]: result[
        "test_mae"
    ]
    for result
    in final_results
}


# ======================================================================
# OVERALL METRICS
# ======================================================================

mean_validation_mae = float(
    np.mean(
        [
            result[
                "validation_mae"
            ]
            for result
            in final_results
        ]
    )
)

mean_validation_rmse = float(
    np.mean(
        [
            result[
                "validation_rmse"
            ]
            for result
            in final_results
        ]
    )
)

mean_test_mae = float(
    np.mean(
        [
            result[
                "test_mae"
            ]
            for result
            in final_results
        ]
    )
)

mean_test_rmse = float(
    np.mean(
        [
            result[
                "test_rmse"
            ]
            for result
            in final_results
        ]
    )
)


# ======================================================================
# BEST HORIZONS
# ======================================================================

best_validation_horizon = min(
    final_results,
    key=lambda x:
        x["validation_mae"],
)

best_test_horizon = min(
    final_results,
    key=lambda x:
        x["test_mae"],
)


# ======================================================================
# STEP 12 COMPARISON BY TUNING HORIZON
# ======================================================================

print("\n" + "=" * 70)
print("STEP 12 VS SELECTED CONFIGURATION")
print("=" * 70)

step12_validation_by_horizon = {
    result[
        "horizon"
    ]: result[
        "mae"
    ]
    for result
    in step12_report[
        "validation"
    ][
        "by_horizon"
    ]
}

print(
    f"{'Horizon':<10}"
    f"{'Step12 MAE':>14}"
    f"{'Selected MAE':>16}"
    f"{'Improvement':>16}"
    f"{'Improvement %':>16}"
)

print("-" * 74)

horizon_comparisons = []

for horizon in TUNING_HORIZONS:

    step12_mae = (
        step12_validation_by_horizon[
            horizon
        ]
    )

    selected_mae = (
        validation_by_horizon[
            horizon
        ]
    )

    improvement = (
        step12_mae
        - selected_mae
    )

    if step12_mae != 0:

        improvement_pct = (
            improvement
            / step12_mae
            * 100
        )

    else:

        improvement_pct = 0.0

    horizon_comparisons.append(
        {
            "horizon":
                horizon,

            "step12_mae":
                float(
                    step12_mae
                ),

            "selected_mae":
                float(
                    selected_mae
                ),

            "improvement":
                float(
                    improvement
                ),

            "improvement_percent":
                float(
                    improvement_pct
                ),
        }
    )

    print(
        f"{horizon:<10}"
        f"{step12_mae:>14.3f}"
        f"{selected_mae:>16.3f}"
        f"{improvement:>16.3f}"
        f"{improvement_pct:>15.2f}%"
    )


# ======================================================================
# FINAL SELECTED HORIZON SUMMARY
# ======================================================================

print("\n" + "=" * 70)
print("SELECTED XGBOOST SUMMARY BY HORIZON")
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
    f"{'Metric':<30}"
    f"{'1h':>10}"
    f"{'6h':>10}"
    f"{'12h':>10}"
    f"{'24h':>10}"
    f"{'48h':>10}"
    f"{'72h':>10}"
)

print("-" * 90)

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
# BASELINE COMPARISON
# ======================================================================

beats_seasonal_baseline = (
    mean_validation_mae
    < SEASONAL_BASELINE_VALIDATION_MAE
)

baseline_difference = (
    SEASONAL_BASELINE_VALIDATION_MAE
    - mean_validation_mae
)

if (
    SEASONAL_BASELINE_VALIDATION_MAE
    != 0
):

    baseline_improvement_percent = (
        baseline_difference
        / SEASONAL_BASELINE_VALIDATION_MAE
        * 100
    )

else:

    baseline_improvement_percent = 0.0


# ======================================================================
# FINAL SUMMARY
# ======================================================================

print("\n" + "=" * 70)
print("STEP 13 FINAL SUMMARY")
print("=" * 70)

print(
    f"Features used          : "
    f"{len(feature_columns)}"
)

print(
    f"Tuning configurations  : "
    f"{len(parameter_configs)}"
)

print(
    f"Tuning horizons        : "
    f"{len(TUNING_HORIZONS)}"
)

print(
    f"Final models trained   : "
    f"{FORECAST_HORIZON}"
)

print()

print(
    "Selected configuration : "
    f"{selected_configuration}"
)

print()

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
    f"Step 12 reference MAE  : "
    f"{reference_mae:.3f}"
)

print(
    f"Tuning improvement     : "
    f"{raw_improvement:.3f}"
)

print(
    f"Tuning improvement (%) : "
    f"{improvement_percent:.2f}%"
)

print()

print(
    f"Seasonal baseline MAE  : "
    f"{SEASONAL_BASELINE_VALIDATION_MAE:.3f}"
)

print(
    f"Difference vs seasonal : "
    f"{baseline_difference:.3f}"
)

print(
    f"Improvement vs seasonal: "
    f"{baseline_improvement_percent:.2f}%"
)

if beats_seasonal_baseline:

    print(
        "Tuned/selected XGBoost vs "
        "Seasonal Persistence: BETTER"
    )

else:

    print(
        "Tuned/selected XGBoost vs "
        "Seasonal Persistence: NOT BETTER"
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

report = {

    "step":
        13,

    "description":
        "XGBoost controlled random "
        "hyperparameter tuning",

    "target":
        TARGET_COLUMN,

    "forecast_horizon":
        FORECAST_HORIZON,

    "features_used":
        len(feature_columns),

    "feature_columns":
        feature_columns,

    "feature_consistency":
        {
            "step12_feature_count":
                len(step12_features),

            "step13_feature_count":
                len(feature_columns),

            "exact_match":
                feature_columns
                == step12_features,
        },

    "tuning_horizons":
        TUNING_HORIZONS,

    "random_state":
        RANDOM_STATE,

    "candidate_configurations":
        len(parameter_configs),

    "search_space_size":
        len(all_combinations),

    "search_method":
        "controlled_random_search",

    "reference_configuration":
        baseline_config_dict,

    "reference_validation_mae":
        reference_mae,

    "reference_validation_rmse":
        reference_rmse,

    "best_tuning_parameters":
        best_tuning_params,

    "best_tuning_validation_mae":
        best_result[
            "mean_validation_mae"
        ],

    "best_tuning_validation_rmse":
        best_result[
            "mean_validation_rmse"
        ],

    "tuning_improvement_mae":
        raw_improvement,

    "tuning_improvement_percent":
        improvement_percent,

    "minimum_improvement_percent":
        MIN_IMPROVEMENT_PERCENT,

    "selected_configuration":
        selected_configuration,

    "selected_parameters":
        final_params,

    "tuning_results":
        tuning_results,

    "horizon_comparisons":
        horizon_comparisons,

    "final_results":
        final_results,

    "final_summary":
        {
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

            "step12_reference_validation_mae":
                reference_mae,

            "tuning_improvement_mae":
                raw_improvement,

            "tuning_improvement_percent":
                improvement_percent,

            "seasonal_baseline_validation_mae":
                SEASONAL_BASELINE_VALIDATION_MAE,

            "seasonal_baseline_validation_rmse":
                SEASONAL_BASELINE_VALIDATION_RMSE,

            "difference_vs_seasonal_baseline":
                baseline_difference,

            "improvement_vs_seasonal_baseline_percent":
                baseline_improvement_percent,

            "beats_seasonal_baseline":
                beats_seasonal_baseline,
        },

    "timing":
        {
            "tuning_seconds":
                round(
                    total_tuning_time,
                    2
                ),

            "final_training_seconds":
                round(
                    final_training_time,
                    2
                ),

            "total_seconds":
                round(
                    total_tuning_time
                    + final_training_time,
                    2
                ),
        },

    "model_directory":
        MODEL_DIR,

    "methodology":
        {
            "train_used_for_fitting":
                True,

            "validation_used_for_selection":
                True,

            "test_used_for_selection":
                False,

            "independent_horizon_models":
                True,

            "mae_primary_metric":
                True,

            "exact_step12_features":
                True,
        },
}


with open(
    REPORT_FILE,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        report,
        f,
        indent=2
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
    f"Features used          : "
    f"{len(feature_columns)}"
)

print(
    f"Tuning configurations  : "
    f"{len(parameter_configs)}"
)

print(
    f"Tuning horizons        : "
    f"{len(TUNING_HORIZONS)}"
)

print(
    f"Final models trained   : "
    f"{FORECAST_HORIZON}"
)

print(
    f"Selected configuration : "
    f"{selected_configuration}"
)

print(
    f"Mean validation MAE    : "
    f"{mean_validation_mae:.3f}"
)

print(
    f"Mean test MAE          : "
    f"{mean_test_mae:.3f}"
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
    "\nTest set was NOT used "
    "for hyperparameter selection."
)

print(
    "Step 12 feature set was "
    "verified and preserved exactly."
)