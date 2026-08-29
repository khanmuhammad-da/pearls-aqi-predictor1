"""
PEARLS AQI PREDICTOR
STEP 14 — XGBOOST PRODUCTION INFERENCE

Purpose
-------
Run production inference using the locked XGBoost deployment candidate.

Deployment candidate
--------------------
- XGBoost
- 72 independent horizon models
- Horizons t+1 ... t+72
- Exact 101-feature protocol from Step 13
- No hyperparameter selection is performed here
- No validation/test data are used here
- Production inference uses the latest available feature row

Authoritative feature protocol
--------------------------------
The exact feature list is read from:

    reports/xgboost_tuning_results.json

This is the Step 13 report generated after the final tuned
XGBoost models were trained.

Model artifacts
---------------
    models/artifacts/xgboost_tuned/
        xgboost_tuned_01h.json
        ...
        xgboost_tuned_72h.json

Outputs
-------
    reports/predictions/production_72h_predictions.csv
    reports/production_prediction_results.json
"""

import json
import os
import time
import warnings

import numpy as np
import pandas as pd
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

# ----------------------------------------------------------------------
# Step 13 tuning report
# ----------------------------------------------------------------------

STEP13_REPORT_FILE = os.path.join(
    BASE_DIR,
    "reports",
    "xgboost_tuning_results.json",
)

# ----------------------------------------------------------------------
# Tuned model artifacts
# ----------------------------------------------------------------------

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "artifacts",
    "xgboost_tuned",
)

# ----------------------------------------------------------------------
# Production input
#
# We use the processed test file as the current production inference
# feature source unless a dedicated production feature file exists.
#
# IMPORTANT:
# test.csv is used ONLY as a feature container here.
# Future target columns are NEVER passed to the model.
# ----------------------------------------------------------------------

TEST_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "splits",
    "test.csv",
)

# Optional dedicated production input.
PRODUCTION_INPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "production_features.csv",
)

# ----------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------

PREDICTION_DIR = os.path.join(
    BASE_DIR,
    "reports",
    "predictions",
)

PREDICTION_FILE = os.path.join(
    PREDICTION_DIR,
    "production_72h_predictions.csv",
)

REPORT_FILE = os.path.join(
    BASE_DIR,
    "reports",
    "production_prediction_results.json",
)

# ----------------------------------------------------------------------
# Project constants
# ----------------------------------------------------------------------

TARGET_COLUMN = "us_aqi"

FORECAST_HORIZON = 72

EXPECTED_FEATURE_COUNT = 101

RANDOM_STATE = 42


# ======================================================================
# HEADER
# ======================================================================

print("=" * 70)
print("PEARLS AQI PREDICTOR")
print("STEP 14 — XGBOOST PRODUCTION INFERENCE")
print("=" * 70)

print(
    f"Base directory      : {BASE_DIR}"
)

print(
    f"Forecast horizon    : "
    f"{FORECAST_HORIZON} hours"
)

print(
    f"Target              : "
    f"{TARGET_COLUMN}"
)

print(
    f"Expected features   : "
    f"{EXPECTED_FEATURE_COUNT}"
)

print(
    f"Model directory     : "
    f"{MODEL_DIR}"
)


# ======================================================================
# MODEL ARTIFACT CHECK
# ======================================================================

print("\n" + "=" * 70)
print("XGBOOST MODEL ARTIFACT CHECK")
print("=" * 70)

if not os.path.isdir(MODEL_DIR):

    raise FileNotFoundError(
        "\nXGBoost tuned model directory does not exist:\n"
        f"{MODEL_DIR}\n\n"
        "Run Step 13 first."
    )

model_files = []

for horizon in range(
    1,
    FORECAST_HORIZON + 1,
):

    filename = (
        f"xgboost_tuned_{horizon:02d}h.json"
    )

    path = os.path.join(
        MODEL_DIR,
        filename,
    )

    if not os.path.exists(path):

        raise FileNotFoundError(
            "\nMissing XGBoost production artifact:\n"
            f"{path}\n\n"
            "Step 14 requires all 72 horizon models."
        )

    model_files.append(path)


print(
    "01h model       : FOUND"
)

print(
    "72h model       : FOUND"
)

print(
    f"Total models    : "
    f"{len(model_files)}"
)

if len(model_files) != FORECAST_HORIZON:

    raise RuntimeError(
        "Expected exactly 72 XGBoost horizon models."
    )

print(
    "Artifact check  : PASS"
)


# ======================================================================
# LOAD STEP 13 FEATURE PROTOCOL
# ======================================================================

print("\n" + "=" * 70)
print("LOADING STEP 13 FEATURE PROTOCOL")
print("=" * 70)

if not os.path.exists(
    STEP13_REPORT_FILE
):

    raise FileNotFoundError(
        "\nCould not locate the Step 13 tuning report:\n"
        f"{STEP13_REPORT_FILE}\n\n"
        "Step 14 uses the feature_columns stored by Step 13.\n"
        "Do not manually reconstruct the 101 features."
    )


with open(
    STEP13_REPORT_FILE,
    "r",
    encoding="utf-8",
) as f:

    step13_report = json.load(f)


feature_columns = step13_report.get(
    "feature_columns"
)

if not feature_columns:

    raise ValueError(
        "\nStep 13 report does not contain "
        "'feature_columns'.\n\n"
        "The production inference stage cannot safely "
        "guess the feature protocol."
    )


feature_columns = list(
    feature_columns
)


print(
    f"Step 13 feature count : "
    f"{len(feature_columns)}"
)

print(
    f"Expected feature count: "
    f"{EXPECTED_FEATURE_COUNT}"
)

if len(feature_columns) != EXPECTED_FEATURE_COUNT:

    raise ValueError(
        "\nProduction feature protocol mismatch.\n"
        f"Expected: {EXPECTED_FEATURE_COUNT}\n"
        f"Found   : {len(feature_columns)}\n\n"
        "Step 14 will not guess or silently modify features."
    )


# Check duplicate feature names.

if len(feature_columns) != len(
    set(feature_columns)
):

    raise ValueError(
        "Step 13 feature protocol contains duplicate "
        "feature names."
    )


print(
    "Feature protocol     : PASS"
)

print(
    "Exact 101-feature set: LOCKED"
)


# ======================================================================
# VERIFY DEPLOYMENT CONFIGURATION
# ======================================================================

print("\n" + "=" * 70)
print("VERIFYING DEPLOYMENT CONFIGURATION")
print("=" * 70)

selected_configuration = (
    step13_report.get(
        "selected_configuration"
    )
)

selected_parameters = (
    step13_report.get(
        "best_parameters"
    )
)

if selected_configuration is None:

    print(
        "Selected configuration : "
        "NOT EXPLICITLY STORED"
    )

else:

    print(
        "Selected configuration : "
        f"{selected_configuration}"
    )


if selected_parameters:

    print(
        "Best parameters        :"
    )

    print(
        json.dumps(
            selected_parameters,
            indent=2,
        )
    )

else:

    print(
        "Best parameters        : "
        "NOT STORED"
    )


print(
    "\nDeployment candidate   : XGBoost tuned"
)

print(
    "Model selection here   : NOT PERFORMED"
)

print(
    "Validation used here   : NO"
)

print(
    "Test used for selection: NO"
)


# ======================================================================
# LOAD PRODUCTION INPUT
# ======================================================================

print("\n" + "=" * 70)
print("LOADING PRODUCTION INPUT")
print("=" * 70)

# Prefer a dedicated production feature file if it exists.
#
# Otherwise use test.csv as the available latest feature container.
# Future target columns are explicitly excluded later.

if os.path.exists(
    PRODUCTION_INPUT_FILE
):

    INPUT_FILE = PRODUCTION_INPUT_FILE

    input_source = (
        "Dedicated production feature file"
    )

else:

    INPUT_FILE = TEST_FILE

    input_source = (
        "Latest available processed test feature container"
    )


if not os.path.exists(
    INPUT_FILE
):

    raise FileNotFoundError(
        "\nProduction input file not found:\n"
        f"{INPUT_FILE}"
    )


print(
    f"Input source : {input_source}"
)

print(
    f"Input file   : {INPUT_FILE}"
)


input_df = pd.read_csv(
    INPUT_FILE
)

print(
    f"Input rows   : "
    f"{len(input_df):,}"
)

print(
    f"Input columns: "
    f"{len(input_df.columns)}"
)


# ======================================================================
# INPUT TIMESTAMP VALIDATION
# ======================================================================

print("\n" + "=" * 70)
print("INPUT TIMESTAMP VALIDATION")
print("=" * 70)

if "time" not in input_df.columns:

    raise ValueError(
        "Production input must contain a 'time' column."
    )


input_df["time"] = pd.to_datetime(
    input_df["time"],
    errors="raise",
)


if input_df["time"].isna().any():

    raise ValueError(
        "Production input contains invalid timestamps."
    )


if input_df["time"].duplicated().any():

    raise ValueError(
        "Production input contains duplicate timestamps."
    )


if not input_df[
    "time"
].is_monotonic_increasing:

    raise ValueError(
        "Production input timestamps are not chronological."
    )


print(
    f"Time range   : "
    f"{input_df['time'].min()} -> "
    f"{input_df['time'].max()}"
)

print(
    "Timestamp validation: PASS"
)


# ======================================================================
# SELECT LATEST PRODUCTION ROW
# ======================================================================

print("\n" + "=" * 70)
print("SELECTING LATEST PRODUCTION FEATURE ROW")
print("=" * 70)

latest_row = input_df.iloc[
    -1
]

latest_timestamp = latest_row[
    "time"
]

print(
    f"Latest timestamp: "
    f"{latest_timestamp}"
)


# ======================================================================
# FEATURE CONSISTENCY CHECK
# ======================================================================

print("\n" + "=" * 70)
print("FEATURE CONSISTENCY CHECK")
print("=" * 70)

missing_features = [
    feature
    for feature in feature_columns
    if feature not in input_df.columns
]

if missing_features:

    print(
        f"Missing feature count: "
        f"{len(missing_features)}"
    )

    print(
        "Missing features:"
    )

    for feature in missing_features:

        print(
            f"  - {feature}"
        )

    raise ValueError(
        "\nProduction input does not contain the "
        "complete Step 13 feature protocol."
    )


print(
    f"Required features : "
    f"{len(feature_columns)}"
)

print(
    "All required features FOUND"
)

print(
    "Feature consistency: PASS"
)


# ======================================================================
# CREATE PRODUCTION FEATURE MATRIX
# ======================================================================

print("\n" + "=" * 70)
print("CREATING PRODUCTION FEATURE MATRIX")
print("=" * 70)

X_production = (
    latest_row[
        feature_columns
    ]
    .to_frame()
    .T
)


X_production = X_production.astype(
    np.float32
)


print(
    f"X_production shape: "
    f"{X_production.shape}"
)


if X_production.shape[1] != (
    EXPECTED_FEATURE_COUNT
):

    raise ValueError(
        "\nProduction feature count mismatch.\n"
        f"Expected: {EXPECTED_FEATURE_COUNT}\n"
        f"Found   : {X_production.shape[1]}"
    )


# ======================================================================
# NUMERIC FEATURE VALIDATION
# ======================================================================

print("\n" + "=" * 70)
print("PRODUCTION FEATURE VALIDATION")
print("=" * 70)

if X_production.isna().any().any():

    missing = X_production.columns[
        X_production.isna().any()
    ].tolist()

    raise ValueError(
        "\nProduction feature row contains missing values.\n"
        f"Features: {missing}"
    )


X_array = X_production.to_numpy(
    dtype=np.float32
)


if not np.isfinite(
    X_array
).all():

    raise ValueError(
        "Production feature row contains "
        "non-finite values."
    )


print(
    "Missing values : NONE"
)

print(
    "Infinite values: NONE"
)

print(
    "Numeric validation: PASS"
)


# ======================================================================
# LOAD AND RUN 72 MODELS
# ======================================================================

print("\n" + "=" * 70)
print("RUNNING XGBOOST PRODUCTION INFERENCE")
print("=" * 70)

print(
    f"Forecast origin : "
    f"{latest_timestamp}"
)

print(
    f"Horizons        : "
    f"1 -> {FORECAST_HORIZON} hours"
)


predictions = []

inference_start = time.time()


for horizon, model_path in enumerate(
    model_files,
    start=1,
):

    model_start = time.time()

    model = XGBRegressor()

    model.load_model(
        model_path
    )

    prediction = model.predict(
        X_production
    )

    prediction_value = float(
        prediction[0]
    )

    if not np.isfinite(
        prediction_value
    ):

        raise ValueError(
            f"Model {horizon:02d}h produced "
            "a non-finite prediction."
        )


    if prediction_value < 0:

        raise ValueError(
            f"Model {horizon:02d}h produced "
            f"an invalid negative AQI prediction: "
            f"{prediction_value}"
        )


    forecast_timestamp = (
        latest_timestamp
        + pd.Timedelta(
            hours=horizon
        )
    )


    elapsed = (
        time.time()
        - model_start
    )


    predictions.append(
        {
            "forecast_origin": (
                latest_timestamp
            ),

            "forecast_timestamp": (
                forecast_timestamp
            ),

            "horizon_hours": (
                horizon
            ),

            "predicted_us_aqi": (
                prediction_value
            ),

            "model": (
                f"xgboost_tuned_{horizon:02d}h"
            ),

            "model_artifact": (
                model_path
            ),
        }
    )


    print(
        f"{horizon:02d}h | "
        f"Predicted AQI = "
        f"{prediction_value:8.3f} | "
        f"Time = "
        f"{elapsed:6.2f}s"
    )


total_inference_time = (
    time.time()
    - inference_start
)


# ======================================================================
# CREATE FORECAST DATAFRAME
# ======================================================================

forecast_df = pd.DataFrame(
    predictions
)


# ======================================================================
# FORECAST VALIDATION
# ======================================================================

print("\n" + "=" * 70)
print("VALIDATING PRODUCTION FORECAST")
print("=" * 70)

if len(forecast_df) != (
    FORECAST_HORIZON
):

    raise ValueError(
        "Production forecast does not contain "
        "exactly 72 rows."
    )


expected_horizons = list(
    range(
        1,
        FORECAST_HORIZON + 1,
    )
)


actual_horizons = forecast_df[
    "horizon_hours"
].tolist()


if actual_horizons != (
    expected_horizons
):

    raise ValueError(
        "Forecast horizon sequence is invalid."
    )


if forecast_df[
    "forecast_timestamp"
].duplicated().any():

    raise ValueError(
        "Duplicate forecast timestamps detected."
    )


if not forecast_df[
    "predicted_us_aqi"
].apply(
    np.isfinite
).all():

    raise ValueError(
        "Non-finite production predictions detected."
    )


negative_count = int(
    (
        forecast_df[
            "predicted_us_aqi"
        ]
        < 0
    ).sum()
)


if negative_count != 0:

    raise ValueError(
        f"Invalid negative AQI predictions: "
        f"{negative_count}"
    )


timestamp_diffs = (
    forecast_df[
        "forecast_timestamp"
    ]
    .diff()
    .dropna()
)


if not (
    timestamp_diffs
    == pd.Timedelta(hours=1)
).all():

    raise ValueError(
        "Forecast timestamps are not "
        "hourly continuous."
    )


print(
    f"Forecast rows          : "
    f"{len(forecast_df)}"
)

print(
    f"Expected rows          : "
    f"{FORECAST_HORIZON}"
)

print(
    f"Horizon range          : "
    f"{forecast_df['horizon_hours'].min()} "
    f"-> "
    f"{forecast_df['horizon_hours'].max()}"
)

print(
    "Horizon sequence       : PASS"
)

print(
    f"Duplicate timestamps   : "
    f"{forecast_df['forecast_timestamp'].duplicated().sum()}"
)

print(
    "Timestamp continuity   : PASS"
)

print(
    "Prediction values finite: "
    f"{forecast_df['predicted_us_aqi'].apply(np.isfinite).all()}"
)

print(
    f"Invalid negative AQI   : "
    f"{negative_count}"
)

print(
    "Production forecast validation: PASS"
)


# ======================================================================
# SAVE PRODUCTION FORECAST
# ======================================================================

print("\n" + "=" * 70)
print("SAVING PRODUCTION FORECAST")
print("=" * 70)

os.makedirs(
    PREDICTION_DIR,
    exist_ok=True,
)


forecast_df.to_csv(
    PREDICTION_FILE,
    index=False,
)


print(
    "Prediction file saved:"
)

print(
    PREDICTION_FILE
)


# ======================================================================
# PRODUCTION SUMMARY
# ======================================================================

prediction_values = (
    forecast_df[
        "predicted_us_aqi"
    ]
    .to_numpy(
        dtype=float
    )
)


summary = {

    "minimum_prediction": float(
        np.min(
            prediction_values
        )
    ),

    "maximum_prediction": float(
        np.max(
            prediction_values
        )
    ),

    "mean_prediction": float(
        np.mean(
            prediction_values
        )
    ),

    "median_prediction": float(
        np.median(
            prediction_values
        )
    ),

    "first_hour_prediction": float(
        prediction_values[0]
    ),

    "twenty_four_hour_prediction": float(
        prediction_values[23]
    ),

    "forty_eight_hour_prediction": float(
        prediction_values[47]
    ),

    "seventy_two_hour_prediction": float(
        prediction_values[71]
    ),
}


# ======================================================================
# SAVE PRODUCTION REPORT
# ======================================================================

print("\n" + "=" * 70)
print("SAVING PRODUCTION INFERENCE REPORT")
print("=" * 70)

report = {

    "step": 14,

    "project":
        "PEARLS AQI PREDICTOR",

    "model":
        "XGBoost tuned",

    "deployment_candidate":
        True,

    "target":
        TARGET_COLUMN,

    "forecast_horizon":
        FORECAST_HORIZON,

    "forecast_origin":
        str(
            latest_timestamp
        ),

    "features_used":
        len(feature_columns),

    "feature_columns":
        feature_columns,

    "feature_protocol_source":
        "reports/xgboost_tuning_results.json",

    "model_directory":
        MODEL_DIR,

    "model_count":
        len(model_files),

    "model_artifacts":
        model_files,

    "selected_configuration":
        selected_configuration,

    "selected_parameters":
        selected_parameters,

    "model_selection_performed":
        False,

    "validation_used":
        False,

    "test_used_for_selection":
        False,

    "future_target_columns_used":
        False,

    "input_source":
        input_source,

    "input_file":
        INPUT_FILE,

    "inference_rows":
        1,

    "inference_time_seconds":
        round(
            total_inference_time,
            3,
        ),

    "forecast_summary":
        summary,

    "prediction_file":
        PREDICTION_FILE,

    "production_ready":
        True,
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


print(
    "Inference report saved:"
)

print(
    REPORT_FILE
)


# ======================================================================
# FINAL OUTPUT
# ======================================================================

print("\n" + "=" * 70)
print("STEP 14 — XGBOOST PRODUCTION INFERENCE COMPLETE")
print("=" * 70)

print(
    f"Deployment candidate : XGBoost tuned"
)

print(
    f"Models loaded        : "
    f"{len(model_files)}"
)

print(
    f"Features used        : "
    f"{len(feature_columns)}"
)

print(
    f"Forecast horizon     : "
    f"{FORECAST_HORIZON} hours"
)

print(
    f"Forecast origin      : "
    f"{latest_timestamp}"
)

print(
    f"Minimum predicted AQI: "
    f"{summary['minimum_prediction']:.3f}"
)

print(
    f"Maximum predicted AQI: "
    f"{summary['maximum_prediction']:.3f}"
)

print(
    f"Mean predicted AQI   : "
    f"{summary['mean_prediction']:.3f}"
)

print(
    f"1h predicted AQI     : "
    f"{summary['first_hour_prediction']:.3f}"
)

print(
    f"24h predicted AQI    : "
    f"{summary['twenty_four_hour_prediction']:.3f}"
)

print(
    f"48h predicted AQI    : "
    f"{summary['forty_eight_hour_prediction']:.3f}"
)

print(
    f"72h predicted AQI    : "
    f"{summary['seventy_two_hour_prediction']:.3f}"
)

print()
print(
    "Production forecast : READY"
)

print()
print(
    "Prediction file:"
)

print(
    PREDICTION_FILE
)

print()
print(
    "Inference report:"
)

print(
    REPORT_FILE
)

print("=" * 70)