"""
PEARLS AQI PREDICTOR
STEP 17 — XGBOOST PRODUCTION API TEST

Purpose
-------
Validate the Step 16 production API as an external client.

This script:
    - Tests API connectivity.
    - Tests /health.
    - Tests /metadata.
    - Loads the latest production feature row.
    - Uses the exact locked Step 13 feature protocol.
    - Sends one prediction request to /predict.
    - Explicitly sends forecast_origin using the Step 16 API contract.
    - Validates the returned 72-hour forecast.
    - Validates horizon ordering.
    - Validates timestamp continuity.
    - Validates prediction values.
    - Compares API output with Step 15 when raw predictions are available.
    - Compares API output with Step 14 when raw predictions are available.
    - Saves an API test report.

This script NEVER:
    - trains a model
    - tunes hyperparameters
    - retrains models
    - performs model selection
    - uses the test set for model selection
    - modifies model artifacts
"""

import json
import math
import os
import sys
import time
from datetime import timedelta

import numpy as np
import pandas as pd
import requests


# ======================================================================
# CONFIGURATION
# ======================================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

API_BASE_URL = "http://127.0.0.1:8000"

HEALTH_ENDPOINT = (
    f"{API_BASE_URL}/health"
)

METADATA_ENDPOINT = (
    f"{API_BASE_URL}/metadata"
)

PREDICT_ENDPOINT = (
    f"{API_BASE_URL}/predict"
)

TEST_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "splits",
    "test.csv",
)

STEP13_REPORT_FILE = os.path.join(
    BASE_DIR,
    "reports",
    "xgboost_tuning_results.json",
)

STEP15_REPORT_FILE = os.path.join(
    BASE_DIR,
    "reports",
    "production_service_results.json",
)

STEP14_REPORT_FILE = os.path.join(
    BASE_DIR,
    "reports",
    "production_prediction_results.json",
)

REPORT_FILE = os.path.join(
    BASE_DIR,
    "reports",
    "xgboost_api_test_results.json",
)

EXPECTED_FEATURE_COUNT = 101

FORECAST_HORIZON = 72

TARGET_COLUMN = "us_aqi"

REQUEST_TIMEOUT_SECONDS = 30

RANDOM_STATE = 42


# ======================================================================
# HEADER
# ======================================================================

print("=" * 70)
print("PEARLS AQI PREDICTOR")
print("STEP 17 — XGBOOST PRODUCTION API TEST")
print("=" * 70)

print(
    f"Base directory      : {BASE_DIR}"
)

print(
    f"API base URL        : {API_BASE_URL}"
)

print(
    f"Forecast horizon    : {FORECAST_HORIZON}"
)

print(
    f"Target              : {TARGET_COLUMN}"
)

print(
    f"Expected features   : "
    f"{EXPECTED_FEATURE_COUNT}"
)


# ======================================================================
# UTILITY FUNCTIONS
# ======================================================================

def fail(message):
    """
    Print a standardized failure message and terminate.
    """

    print("\n" + "=" * 70)
    print("STEP 17 FAILED")
    print("=" * 70)

    print(message)

    sys.exit(1)


def print_json(data):
    """
    Pretty-print JSON safely.
    """

    print(
        json.dumps(
            data,
            indent=2,
            default=str,
        )
    )


def is_finite_number(value):
    """
    Return True when value can be interpreted as a finite float.
    """

    try:

        numeric_value = float(
            value
        )

        return math.isfinite(
            numeric_value
        )

    except (
        TypeError,
        ValueError,
    ):

        return False


def normalize_bool(value):
    """
    Normalize API/report boolean-like values.
    """

    if isinstance(
        value,
        bool,
    ):

        return value

    if isinstance(
        value,
        str,
    ):

        return value.strip().lower() in {
            "true",
            "yes",
            "1",
            "pass",
            "passed",
        }

    return bool(value)


def extract_prediction_list(response_json):
    """
    Extract the prediction list from common Step 16 response shapes.
    """

    if isinstance(
        response_json,
        list,
    ):

        return response_json

    if not isinstance(
        response_json,
        dict,
    ):

        return None

    possible_keys = [
        "predictions",
        "forecast",
        "results",
        "forecast_rows",
        "data",
    ]

    for key in possible_keys:

        value = response_json.get(
            key
        )

        if isinstance(
            value,
            list,
        ):

            return value

    return None


def extract_prediction_value(row):
    """
    Extract AQI prediction from a forecast row.
    """

    if isinstance(
        row,
        (int, float),
    ):

        return float(row)

    if isinstance(
        row,
        dict,
    ):

        possible_keys = [
            "predicted_aqi",
            "prediction",
            "predicted_us_aqi",
            "aqi_prediction",
            "predicted_value",
            "value",
            "us_aqi",
        ]

        for key in possible_keys:

            if key in row:

                value = row[key]

                if is_finite_number(
                    value
                ):

                    return float(
                        value
                    )

    return None


def extract_horizon(
    row,
    default_horizon,
):
    """
    Extract forecast horizon.
    """

    if isinstance(
        row,
        dict,
    ):

        possible_keys = [
            "horizon",
            "forecast_horizon",
            "horizon_hours",
            "hour",
            "step",
        ]

        for key in possible_keys:

            if key in row:

                try:

                    return int(
                        row[key]
                    )

                except (
                    TypeError,
                    ValueError,
                ):

                    pass

    return default_horizon


def extract_timestamp(row):
    """
    Extract forecast timestamp.
    """

    if not isinstance(
        row,
        dict,
    ):

        return None

    possible_keys = [
        "timestamp",
        "forecast_timestamp",
        "forecast_time",
        "time",
        "datetime",
    ]

    for key in possible_keys:

        if key in row:

            value = row[key]

            if value is None:

                continue

            try:

                parsed = pd.to_datetime(
                    value
                )

                return parsed

            except Exception:

                continue

    return None


def extract_api_forecast_origin(
    response_json
):
    """
    Extract the top-level forecast_origin returned by Step 16.
    """

    if not isinstance(
        response_json,
        dict,
    ):

        return None

    value = response_json.get(
        "forecast_origin"
    )

    if value is None:

        return None

    try:

        return pd.to_datetime(
            value
        )

    except Exception:

        return None


def make_json_safe(value):
    """
    Recursively convert NumPy/Pandas values into JSON-safe values.
    """

    if isinstance(
        value,
        dict,
    ):

        return {
            str(key):
                make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(
        value,
        list,
    ):

        return [
            make_json_safe(item)
            for item in value
        ]

    if isinstance(
        value,
        tuple,
    ):

        return [
            make_json_safe(item)
            for item in value
        ]

    if isinstance(
        value,
        np.integer,
    ):

        return int(value)

    if isinstance(
        value,
        np.floating,
    ):

        return float(value)

    if isinstance(
        value,
        np.bool_,
    ):

        return bool(value)

    if isinstance(
        value,
        pd.Timestamp,
    ):

        return value.isoformat()

    return value


# ======================================================================
# INPUT FILE CHECK
# ======================================================================

print("\n" + "=" * 70)
print("INPUT FILE CHECK")
print("=" * 70)

if not os.path.exists(
    TEST_FILE
):

    fail(
        "Production feature container not found:\n"
        f"{TEST_FILE}"
    )

print(
    "test.csv            : FOUND"
)


if not os.path.exists(
    STEP13_REPORT_FILE
):

    fail(
        "Step 13 tuning report not found:\n"
        f"{STEP13_REPORT_FILE}"
    )

print(
    "Step 13 report      : FOUND"
)


if os.path.exists(
    STEP15_REPORT_FILE
):

    print(
        "Step 15 report      : FOUND"
    )

else:

    print(
        "Step 15 report      : NOT FOUND "
        "(comparison will be skipped)"
    )


if os.path.exists(
    STEP14_REPORT_FILE
):

    print(
        "Step 14 report      : FOUND"
    )

else:

    print(
        "Step 14 report      : NOT FOUND "
        "(comparison will be skipped)"
    )


# ======================================================================
# LOAD STEP 13 FEATURE PROTOCOL
# ======================================================================

print("\n" + "=" * 70)
print("LOADING STEP 13 FEATURE PROTOCOL")
print("=" * 70)

try:

    with open(
        STEP13_REPORT_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        step13_report = json.load(
            f
        )

except Exception as exc:

    fail(
        "Unable to load Step 13 report:\n"
        f"{exc}"
    )


feature_columns = (
    step13_report.get(
        "feature_columns"
    )
)

if not feature_columns:

    feature_columns = (
        step13_report.get(
            "features"
        )
    )


if not feature_columns:

    fail(
        "Step 13 report does not contain "
        "'feature_columns' or 'features'."
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


if len(
    feature_columns
) != EXPECTED_FEATURE_COUNT:

    fail(
        "Step 13 feature-count mismatch.\n"
        f"Expected: {EXPECTED_FEATURE_COUNT}\n"
        f"Found   : {len(feature_columns)}"
    )


print(
    "Feature protocol     : PASS"
)

print(
    "Exact 101-feature set: LOCKED"
)


# ======================================================================
# LOAD PRODUCTION FEATURE CONTAINER
# ======================================================================

print("\n" + "=" * 70)
print("LOADING PRODUCTION FEATURE CONTAINER")
print("=" * 70)

try:

    test_df = pd.read_csv(
        TEST_FILE
    )

except Exception as exc:

    fail(
        "Unable to load test.csv:\n"
        f"{exc}"
    )


print(
    f"Input rows           : "
    f"{len(test_df):,}"
)

print(
    f"Input columns        : "
    f"{len(test_df.columns)}"
)


# ======================================================================
# TIMESTAMP VALIDATION
# ======================================================================

print("\n" + "=" * 70)
print("VALIDATING INPUT TIMESTAMP")
print("=" * 70)

if "time" not in test_df.columns:

    fail(
        "Required 'time' column is missing."
    )


try:

    test_df["time"] = pd.to_datetime(
        test_df["time"],
        errors="raise",
    )

except Exception as exc:

    fail(
        "Unable to parse timestamps:\n"
        f"{exc}"
    )


if not test_df[
    "time"
].is_unique:

    fail(
        "Duplicate timestamps detected."
    )


if not test_df[
    "time"
].is_monotonic_increasing:

    fail(
        "Input timestamps are not chronological."
    )


print(
    f"Time range           : "
    f"{test_df['time'].min()} -> "
    f"{test_df['time'].max()}"
)

print(
    "Timestamp validation  : PASS"
)


# ======================================================================
# FEATURE CONSISTENCY
# ======================================================================

print("\n" + "=" * 70)
print("VALIDATING LOCKED FEATURES")
print("=" * 70)

missing_features = [
    feature
    for feature in feature_columns
    if feature not in test_df.columns
]


if missing_features:

    fail(
        "Missing required features:\n"
        + "\n".join(
            missing_features
        )
    )


print(
    f"Required features    : "
    f"{len(feature_columns)}"
)

print(
    "All required features FOUND"
)


# ======================================================================
# LATEST PRODUCTION ROW
# ======================================================================

print("\n" + "=" * 70)
print("SELECTING LATEST PRODUCTION ROW")
print("=" * 70)

latest_row = (
    test_df
    .sort_values(
        "time"
    )
    .iloc[-1]
)


forecast_origin = pd.Timestamp(
    latest_row["time"]
)


# Normalize the origin to UTC.

if forecast_origin.tzinfo is None:

    forecast_origin = (
        forecast_origin.tz_localize(
            "UTC"
        )
    )

else:

    forecast_origin = (
        forecast_origin.tz_convert(
            "UTC"
        )
    )


print(
    f"Latest timestamp     : "
    f"{forecast_origin}"
)


# ======================================================================
# PRODUCTION FEATURE MATRIX
# ======================================================================

print("\n" + "=" * 70)
print("CREATING API REQUEST FEATURE MATRIX")
print("=" * 70)

X_production = (
    latest_row[
        feature_columns
    ]
    .to_numpy(
        dtype=np.float32
    )
    .reshape(
        1,
        -1,
    )
)


print(
    f"X_production shape   : "
    f"{X_production.shape}"
)


if X_production.shape != (
    1,
    EXPECTED_FEATURE_COUNT,
):

    fail(
        "Production feature matrix shape mismatch.\n"
        f"Expected: (1, {EXPECTED_FEATURE_COUNT})\n"
        f"Found   : {X_production.shape}"
    )


if not np.isfinite(
    X_production
).all():

    fail(
        "Production feature matrix contains "
        "NaN or infinite values."
    )


print(
    "Numeric validation   : PASS"
)

print(
    "Missing values       : NONE"
)

print(
    "Infinite values      : NONE"
)


# ======================================================================
# BUILD API PAYLOAD
# ======================================================================

print("\n" + "=" * 70)
print("BUILDING PRODUCTION API REQUEST")
print("=" * 70)

feature_payload = {}

for index, feature in enumerate(
    feature_columns
):

    value = X_production[
        0,
        index
    ]

    feature_payload[
        feature
    ] = float(value)


# IMPORTANT:
#
# Step 16 expects the optional field:
#
#     forecast_origin
#
# NOT:
#
#     timestamp
#
# If forecast_origin is omitted, Step 16 deliberately uses the
# current UTC timestamp. That caused the original Step 17 failure.
#
# Therefore the primary and fallback payloads BOTH preserve
# forecast_origin.

payload = {
    "forecast_origin":
        forecast_origin.isoformat(),

    "features":
        feature_payload,
}


print(
    f"Request features    : "
    f"{len(feature_payload)}"
)

print(
    f"Forecast origin      : "
    f"{forecast_origin}"
)

print(
    "Request payload      : READY"
)

print(
    "API field            : forecast_origin"
)


# ======================================================================
# TEST API HEALTH
# ======================================================================

print("\n" + "=" * 70)
print("TESTING /HEALTH")
print("=" * 70)

health_start = time.time()

try:

    health_response = requests.get(
        HEALTH_ENDPOINT,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

except requests.RequestException as exc:

    fail(
        "Unable to connect to Step 16 API.\n"
        f"Endpoint: {HEALTH_ENDPOINT}\n"
        f"Error: {exc}\n\n"
        "Make sure 16_xgboost_production_api.py "
        "is running."
    )


health_time = (
    time.time()
    - health_start
)


print(
    f"HTTP status         : "
    f"{health_response.status_code}"
)


if health_response.status_code != 200:

    fail(
        "/health did not return HTTP 200.\n"
        f"Response: {health_response.text}"
    )


try:

    health_json = (
        health_response.json()
    )

except Exception as exc:

    fail(
        "/health did not return valid JSON.\n"
        f"{exc}"
    )


print(
    "JSON response       : PASS"
)

print(
    f"Response time       : "
    f"{health_time:.3f}s"
)

print(
    "Health response:"
)

print_json(
    health_json
)


health_status = health_json.get(
    "status"
)

if str(
    health_status
).lower() != "ok":

    fail(
        f"API health status is not 'ok': "
        f"{health_status}"
    )


models_loaded = health_json.get(
    "models_loaded",
    0,
)

if int(
    models_loaded
) != FORECAST_HORIZON:

    fail(
        "Incorrect number of loaded models.\n"
        f"Expected: {FORECAST_HORIZON}\n"
        f"Found   : {models_loaded}"
    )


features_locked = health_json.get(
    "features_locked",
    0,
)

if int(
    features_locked
) != EXPECTED_FEATURE_COUNT:

    fail(
        "Incorrect number of locked features.\n"
        f"Expected: {EXPECTED_FEATURE_COUNT}\n"
        f"Found   : {features_locked}"
    )


if health_json.get(
    "artifact_status"
) != "READY":

    fail(
        "API artifact status is not READY."
    )


print(
    "Health validation    : PASS"
)


# ======================================================================
# TEST API METADATA
# ======================================================================

print("\n" + "=" * 70)
print("TESTING /METADATA")
print("=" * 70)

metadata_start = time.time()

try:

    metadata_response = requests.get(
        METADATA_ENDPOINT,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

except requests.RequestException as exc:

    fail(
        "Unable to request /metadata.\n"
        f"Error: {exc}"
    )


metadata_time = (
    time.time()
    - metadata_start
)


print(
    f"HTTP status         : "
    f"{metadata_response.status_code}"
)


if metadata_response.status_code != 200:

    fail(
        "/metadata did not return HTTP 200.\n"
        f"Response: {metadata_response.text}"
    )


try:

    metadata_json = (
        metadata_response.json()
    )

except Exception as exc:

    fail(
        "/metadata did not return valid JSON.\n"
        f"{exc}"
    )


print(
    "JSON response       : PASS"
)

print(
    f"Response time       : "
    f"{metadata_time:.3f}s"
)

print(
    "Metadata response:"
)

print_json(
    metadata_json
)


metadata_checks = {
    "service":
        "Pearls AQI Predictor",

    "step":
        16,

    "service_type":
        "production_api",

    "status":
        "READY",

    "target":
        TARGET_COLUMN,

    "forecast_horizon":
        FORECAST_HORIZON,

    "models_loaded":
        FORECAST_HORIZON,

    "features_used":
        EXPECTED_FEATURE_COUNT,

    "feature_protocol":
        "Exact Step 13 feature list",
}


for key, expected in (
    metadata_checks.items()
):

    actual = metadata_json.get(
        key
    )

    if actual != expected:

        if str(actual) != str(
            expected
        ):

            fail(
                "Metadata contract mismatch.\n"
                f"Field: {key}\n"
                f"Expected: {expected}\n"
                f"Found: {actual}"
            )


if not normalize_bool(
    metadata_json.get(
        "feature_order_locked",
        False,
    )
):

    fail(
        "Metadata reports that feature order "
        "is not locked."
    )


for forbidden_flag in [
    "model_selection",
    "hyperparameter_tuning",
    "model_retraining",
    "validation",
    "test_set_usage",
]:

    if normalize_bool(
        metadata_json.get(
            forbidden_flag,
            False,
        )
    ):

        fail(
            "Production API reports forbidden "
            f"operation '{forbidden_flag}' "
            "was performed."
        )


print(
    "Metadata validation  : PASS"
)


# ======================================================================
# TEST /PREDICT
# ======================================================================

print("\n" + "=" * 70)
print("TESTING /PREDICT")
print("=" * 70)

print(
    f"Endpoint             : "
    f"{PREDICT_ENDPOINT}"
)

print(
    f"Features sent        : "
    f"{len(feature_payload)}"
)

print(
    f"Forecast origin      : "
    f"{forecast_origin}"
)

print(
    "Request contract     : "
    "forecast_origin + features"
)


prediction_start = time.time()

prediction_response = None

prediction_json = None

primary_request_error = None


# ======================================================================
# PRIMARY REQUEST
# ======================================================================

try:

    prediction_response = requests.post(
        PREDICT_ENDPOINT,
        json=payload,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

except requests.RequestException as exc:

    primary_request_error = str(
        exc
    )


# ======================================================================
# FALLBACK REQUEST
# ======================================================================

# The fallback MUST also preserve forecast_origin.
#
# It is retained only for compatibility with an API implementation
# that rejects the first payload shape. It does NOT remove
# forecast_origin.

if (
    prediction_response is None
    or prediction_response.status_code
    in {
        400,
        404,
        422,
    }
):

    fallback_payload = {
        "forecast_origin":
            forecast_origin.isoformat(),

        "features":
            feature_payload,
    }

    try:

        fallback_response = requests.post(
            PREDICT_ENDPOINT,
            json=fallback_payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

        if (
            prediction_response is None
            or fallback_response.status_code
            == 200
        ):

            prediction_response = (
                fallback_response
            )

            payload = (
                fallback_payload
            )

    except requests.RequestException as exc:

        if prediction_response is None:

            primary_request_error = str(
                exc
            )


prediction_time = (
    time.time()
    - prediction_start
)


if prediction_response is None:

    fail(
        "Prediction request failed.\n"
        f"Error: {primary_request_error}"
    )


print(
    f"HTTP status         : "
    f"{prediction_response.status_code}"
)

print(
    f"Response time       : "
    f"{prediction_time:.3f}s"
)


if prediction_response.status_code != 200:

    print(
        "\nAPI response body:"
    )

    print(
        prediction_response.text
    )

    fail(
        "Prediction endpoint did not return "
        "HTTP 200."
    )


try:

    prediction_json = (
        prediction_response.json()
    )

except Exception as exc:

    fail(
        "Prediction endpoint did not return "
        "valid JSON.\n"
        f"{exc}"
    )


print(
    "JSON response       : PASS"
)

print(
    "Prediction response:"
)

print_json(
    prediction_json
)


# ======================================================================
# API FORECAST ORIGIN VALIDATION
# ======================================================================

print("\n" + "=" * 70)
print("VALIDATING API FORECAST ORIGIN")
print("=" * 70)

api_forecast_origin = (
    extract_api_forecast_origin(
        prediction_json
    )
)


if api_forecast_origin is None:

    fail(
        "API response does not contain a valid "
        "'forecast_origin' field."
    )


if api_forecast_origin.tzinfo is None:

    api_forecast_origin = (
        api_forecast_origin.tz_localize(
            "UTC"
        )
    )

else:

    api_forecast_origin = (
        api_forecast_origin.tz_convert(
            "UTC"
        )
    )


if api_forecast_origin != forecast_origin:

    fail(
        "API forecast origin mismatch.\n"
        f"Expected: {forecast_origin}\n"
        f"Found   : {api_forecast_origin}\n\n"
        "This indicates that Step 16 did not use "
        "the forecast_origin supplied by Step 17."
    )


print(
    f"Requested origin     : "
    f"{forecast_origin}"
)

print(
    f"API returned origin  : "
    f"{api_forecast_origin}"
)

print(
    "Forecast origin      : PASS"
)


# ======================================================================
# EXTRACT API FORECAST
# ======================================================================

print("\n" + "=" * 70)
print("EXTRACTING API FORECAST")
print("=" * 70)

prediction_rows = (
    extract_prediction_list(
        prediction_json
    )
)


if prediction_rows is None:

    fail(
        "Unable to locate forecast rows in API response.\n"
        "Expected one of:\n"
        "  predictions\n"
        "  forecast\n"
        "  results\n"
        "  forecast_rows\n"
        "  data"
    )


print(
    f"Forecast rows       : "
    f"{len(prediction_rows)}"
)

print(
    f"Expected rows       : "
    f"{FORECAST_HORIZON}"
)


if len(
    prediction_rows
) != FORECAST_HORIZON:

    fail(
        "API returned an incorrect number "
        "of forecast rows."
    )


# ======================================================================
# FORECAST STRUCTURE VALIDATION
# ======================================================================

print("\n" + "=" * 70)
print("VALIDATING API FORECAST STRUCTURE")
print("=" * 70)

horizons = []

predictions = []

timestamps = []


for index, row in enumerate(
    prediction_rows,
    start=1,
):

    horizon = extract_horizon(
        row,
        index,
    )

    prediction = extract_prediction_value(
        row
    )

    timestamp = extract_timestamp(
        row
    )

    horizons.append(
        horizon
    )

    predictions.append(
        prediction
    )

    timestamps.append(
        timestamp
    )


expected_horizons = list(
    range(
        1,
        FORECAST_HORIZON + 1,
    )
)


if horizons != expected_horizons:

    fail(
        "Horizon sequence mismatch.\n"
        f"Expected: {expected_horizons}\n"
        f"Found: {horizons}"
    )


print(
    "Horizon range       : "
    f"{min(horizons)} -> {max(horizons)}"
)

print(
    "Horizon sequence    : PASS"
)


if len(
    set(horizons)
) != FORECAST_HORIZON:

    fail(
        "Duplicate forecast horizons detected."
    )


print(
    "Duplicate horizons  : 0"
)


# ======================================================================
# PREDICTION VALUE VALIDATION
# ======================================================================

print("\n" + "=" * 70)
print("VALIDATING PREDICTION VALUES")
print("=" * 70)

invalid_prediction_indices = []


for index, prediction in enumerate(
    predictions,
    start=1,
):

    if prediction is None:

        invalid_prediction_indices.append(
            index
        )

        continue

    if not is_finite_number(
        prediction
    ):

        invalid_prediction_indices.append(
            index
        )


if invalid_prediction_indices:

    fail(
        "Invalid prediction values detected "
        f"at rows: {invalid_prediction_indices}"
    )


negative_predictions = [
    index
    for index, prediction
    in enumerate(
        predictions,
        start=1,
    )
    if float(prediction) < 0
]


if negative_predictions:

    fail(
        "Negative AQI predictions detected "
        f"at rows: {negative_predictions}"
    )


predictions = [
    float(prediction)
    for prediction in predictions
]


print(
    "Prediction values finite: True"
)

print(
    "Invalid negative AQI : 0"
)

print(
    "Prediction validation : PASS"
)


# ======================================================================
# TIMESTAMP VALIDATION
# ======================================================================

print("\n" + "=" * 70)
print("VALIDATING FORECAST TIMESTAMPS")
print("=" * 70)

timestamps_available = all(
    timestamp is not None
    for timestamp in timestamps
)


timestamp_validation = {
    "timestamps_present":
        timestamps_available,

    "duplicate_timestamps":
        None,

    "continuity":
        None,

    "timestamp_match_origin":
        None,
}


if timestamps_available:

    normalized_timestamps = []

    for timestamp in timestamps:

        if timestamp.tzinfo is None:

            timestamp = (
                timestamp.tz_localize(
                    "UTC"
                )
            )

        else:

            timestamp = (
                timestamp.tz_convert(
                    "UTC"
                )
            )

        normalized_timestamps.append(
            timestamp
        )

    timestamps = (
        normalized_timestamps
    )

    timestamp_duplicates = (
        len(
            set(
                timestamps
            )
        )
        != FORECAST_HORIZON
    )

    timestamp_validation[
        "duplicate_timestamps"
    ] = timestamp_duplicates


    if timestamp_duplicates:

        fail(
            "Duplicate forecast timestamps detected."
        )


    expected_timestamp_sequence = [
        forecast_origin
        + timedelta(
            hours=h
        )
        for h in expected_horizons
    ]


    timestamp_continuity = all(
        actual == expected
        for actual, expected in zip(
            timestamps,
            expected_timestamp_sequence,
        )
    )


    timestamp_validation[
        "continuity"
    ] = timestamp_continuity


    timestamp_validation[
        "timestamp_match_origin"
    ] = bool(
        timestamps[0]
        == (
            forecast_origin
            + timedelta(
                hours=1
            )
        )
    )


    if not timestamp_continuity:

        print(
            "Expected timestamp sequence:"
        )

        for expected in (
            expected_timestamp_sequence
        ):

            print(
                expected
            )

        print(
            "\nActual timestamp sequence:"
        )

        for actual in timestamps:

            print(
                actual
            )

        fail(
            "Forecast timestamp continuity failed."
        )


    print(
        "Timestamp fields     : FOUND"
    )

    print(
        "Duplicate timestamps : 0"
    )

    print(
        "Timestamp continuity : PASS"
    )

else:

    print(
        "Timestamp fields     : "
        "NOT RETURNED BY API"
    )

    print(
        "Timestamp continuity : "
        "SKIPPED"
    )


# ======================================================================
# FORECAST SUMMARY
# ======================================================================

print("\n" + "=" * 70)
print("API FORECAST SUMMARY")
print("=" * 70)

minimum_prediction = float(
    np.min(
        predictions
    )
)

maximum_prediction = float(
    np.max(
        predictions
    )
)

mean_prediction = float(
    np.mean(
        predictions
    )
)

median_prediction = float(
    np.median(
        predictions
    )
)

std_prediction = float(
    np.std(
        predictions
    )
)


print(
    f"Minimum predicted AQI: "
    f"{minimum_prediction:.3f}"
)

print(
    f"Maximum predicted AQI: "
    f"{maximum_prediction:.3f}"
)

print(
    f"Mean predicted AQI   : "
    f"{mean_prediction:.3f}"
)

print(
    f"Median predicted AQI : "
    f"{median_prediction:.3f}"
)

print(
    f"Std predicted AQI    : "
    f"{std_prediction:.3f}"
)


print()


for horizon in [
    1,
    6,
    12,
    24,
    48,
    72,
]:

    index = horizon - 1

    print(
        f"{horizon:02d}h | "
        f"Predicted AQI = "
        f"{predictions[index]:8.3f}"
    )


# ======================================================================
# STEP 15 COMPARISON
# ======================================================================

print("\n" + "=" * 70)
print("COMPARING WITH STEP 15 PRODUCTION SERVICE")
print("=" * 70)

step15_comparison = {
    "available":
        False,

    "status":
        "NOT_AVAILABLE",

    "max_absolute_difference":
        None,

    "mean_absolute_difference":
        None,

    "all_predictions_match":
        None,
}


step15_predictions = None


if os.path.exists(
    STEP15_REPORT_FILE
):

    try:

        with open(
            STEP15_REPORT_FILE,
            "r",
            encoding="utf-8",
        ) as f:

            step15_report = json.load(
                f
            )


        final_results = (
            step15_report.get(
                "final_results"
            )
        )


        if isinstance(
            final_results,
            list,
        ):

            sorted_results = sorted(
                final_results,
                key=lambda item:
                    int(
                        item.get(
                            "horizon",
                            0,
                        )
                    ),
            )


            candidate_keys = [
                "prediction",
                "predicted_aqi",
                "predicted_value",
            ]


            extracted = []


            for result in sorted_results:

                found = None

                for key in candidate_keys:

                    if key in result:

                        found = result[key]

                        break

                extracted.append(
                    found
                )


            if (
                len(extracted)
                == FORECAST_HORIZON
                and all(
                    value is not None
                    for value in extracted
                )
            ):

                step15_predictions = [
                    float(value)
                    for value in extracted
                ]


        if step15_predictions is None:

            possible_keys = [
                "predictions",
                "forecast",
                "forecast_predictions",
            ]


            for key in possible_keys:

                candidate = (
                    step15_report.get(
                        key
                    )
                )


                if (
                    isinstance(
                        candidate,
                        list,
                    )
                    and len(candidate)
                    == FORECAST_HORIZON
                ):

                    extracted = []


                    for row in candidate:

                        value = (
                            extract_prediction_value(
                                row
                            )
                        )


                        if value is None:

                            extracted = []

                            break


                        extracted.append(
                            value
                        )


                    if len(
                        extracted
                    ) == FORECAST_HORIZON:

                        step15_predictions = [
                            float(value)
                            for value in extracted
                        ]

                        break


    except Exception as exc:

        print(
            "Warning: unable to read Step 15 "
            f"report: {exc}"
        )


if step15_predictions is not None:

    differences = np.abs(
        np.asarray(
            predictions,
            dtype=float,
        )
        -
        np.asarray(
            step15_predictions,
            dtype=float,
        )
    )


    max_difference = float(
        np.max(
            differences
        )
    )


    mean_difference = float(
        np.mean(
            differences
        )
    )


    exact_match = bool(
        np.allclose(
            predictions,
            step15_predictions,
            rtol=1e-6,
            atol=1e-5,
        )
    )


    step15_comparison = {
        "available":
            True,

        "status":
            "PASS"
            if exact_match
            else "MISMATCH",

        "max_absolute_difference":
            max_difference,

        "mean_absolute_difference":
            mean_difference,

        "all_predictions_match":
            exact_match,
    }


    print(
        f"Max absolute difference : "
        f"{max_difference:.10f}"
    )

    print(
        f"Mean absolute difference: "
        f"{mean_difference:.10f}"
    )

    print(
        f"Prediction match        : "
        f"{exact_match}"
    )


else:

    print(
        "Step 15 report does not contain "
        "raw prediction values."
    )

    print(
        "Direct numerical comparison: SKIPPED"
    )

    print(
        "This is not an API failure."
    )


# ======================================================================
# STEP 14 COMPARISON
# ======================================================================

print("\n" + "=" * 70)
print("COMPARING WITH STEP 14 PRODUCTION INFERENCE")
print("=" * 70)

step14_comparison = {
    "available":
        False,

    "status":
        "NOT_AVAILABLE",

    "max_absolute_difference":
        None,

    "mean_absolute_difference":
        None,

    "all_predictions_match":
        None,
}


step14_predictions = None


if os.path.exists(
    STEP14_REPORT_FILE
):

    try:

        with open(
            STEP14_REPORT_FILE,
            "r",
            encoding="utf-8",
        ) as f:

            step14_report = json.load(
                f
            )


        possible_keys = [
            "predictions",
            "forecast",
            "forecast_predictions",
            "final_predictions",
        ]


        for key in possible_keys:

            candidate = (
                step14_report.get(
                    key
                )
            )


            if (
                isinstance(
                    candidate,
                    list,
                )
                and len(candidate)
                == FORECAST_HORIZON
            ):

                extracted = []


                for row in candidate:

                    value = (
                        extract_prediction_value(
                            row
                        )
                    )


                    if value is None:

                        extracted = []

                        break


                    extracted.append(
                        value
                    )


                if len(
                    extracted
                ) == FORECAST_HORIZON:

                    step14_predictions = [
                        float(value)
                        for value in extracted
                    ]

                    break


    except Exception as exc:

        print(
            "Warning: unable to read Step 14 "
            f"report: {exc}"
        )


if step14_predictions is not None:

    differences = np.abs(
        np.asarray(
            predictions,
            dtype=float,
        )
        -
        np.asarray(
            step14_predictions,
            dtype=float,
        )
    )


    max_difference = float(
        np.max(
            differences
        )
    )


    mean_difference = float(
        np.mean(
            differences
        )
    )


    exact_match = bool(
        np.allclose(
            predictions,
            step14_predictions,
            rtol=1e-6,
            atol=1e-5,
        )
    )


    step14_comparison = {
        "available":
            True,

        "status":
            "PASS"
            if exact_match
            else "MISMATCH",

        "max_absolute_difference":
            max_difference,

        "mean_absolute_difference":
            mean_difference,

        "all_predictions_match":
            exact_match,
    }


    print(
        f"Max absolute difference : "
        f"{max_difference:.10f}"
    )

    print(
        f"Mean absolute difference: "
        f"{mean_difference:.10f}"
    )

    print(
        f"Prediction match        : "
        f"{exact_match}"
    )


else:

    print(
        "Step 14 report does not contain "
        "raw prediction values."
    )

    print(
        "Direct numerical comparison: SKIPPED"
    )


# ======================================================================
# FINAL API CONTRACT CHECK
# ======================================================================

print("\n" + "=" * 70)
print("FINAL API CONTRACT CHECK")
print("=" * 70)


api_contract_checks = {
    "health_endpoint":
        health_response.status_code == 200,

    "health_status_ok":
        str(
            health_json.get(
                "status"
            )
        ).lower()
        == "ok",

    "metadata_endpoint":
        metadata_response.status_code == 200,

    "metadata_ready":
        metadata_json.get(
            "status"
        )
        == "READY",

    "models_loaded":
        int(
            health_json.get(
                "models_loaded",
                0,
            )
        )
        == FORECAST_HORIZON,

    "features_locked":
        int(
            health_json.get(
                "features_locked",
                0,
            )
        )
        == EXPECTED_FEATURE_COUNT,

    "feature_order_locked":
        normalize_bool(
            metadata_json.get(
                "feature_order_locked",
                False,
            )
        ),

    "prediction_endpoint":
        prediction_response.status_code
        == 200,

    "forecast_origin":
        api_forecast_origin
        == forecast_origin,

    "forecast_row_count":
        len(prediction_rows)
        == FORECAST_HORIZON,

    "horizon_sequence":
        horizons
        == expected_horizons,

    "finite_predictions":
        all(
            is_finite_number(
                value
            )
            for value in predictions
        ),

    "non_negative_predictions":
        all(
            value >= 0
            for value in predictions
        ),
}


if timestamps_available:

    api_contract_checks[
        "timestamp_continuity"
    ] = bool(
        timestamp_validation[
            "continuity"
        ]
    )

    api_contract_checks[
        "timestamp_match_origin"
    ] = bool(
        timestamp_validation[
            "timestamp_match_origin"
        ]
    )


for name, passed in (
    api_contract_checks.items()
):

    print(
        f"{name:<30}: "
        f"{'PASS' if passed else 'FAIL'}"
    )


failed_checks = [
    name
    for name, passed
    in api_contract_checks.items()
    if not passed
]


if failed_checks:

    fail(
        "API contract validation failed:\n"
        + "\n".join(
            failed_checks
        )
    )


print(
    "\nAPI contract validation: PASS"
)


# ======================================================================
# SAVE API TEST REPORT
# ======================================================================

print("\n" + "=" * 70)
print("SAVING API TEST REPORT")
print("=" * 70)

os.makedirs(
    os.path.dirname(
        REPORT_FILE
    ),
    exist_ok=True,
)


report = {
    "step":
        17,

    "description":
        "XGBoost production API integration test",

    "service":
        "Pearls AQI Predictor",

    "service_type":
        "production_api",

    "api_base_url":
        API_BASE_URL,

    "health_endpoint":
        HEALTH_ENDPOINT,

    "metadata_endpoint":
        METADATA_ENDPOINT,

    "prediction_endpoint":
        PREDICT_ENDPOINT,

    "target":
        TARGET_COLUMN,

    "forecast_horizon":
        FORECAST_HORIZON,

    "expected_features":
        EXPECTED_FEATURE_COUNT,

    "features_used":
        len(feature_columns),

    "feature_columns":
        feature_columns,

    "feature_protocol":
        "Exact Step 13 feature list",

    "feature_order_locked":
        True,

    "forecast_origin":
        forecast_origin,

    "api_forecast_origin":
        api_forecast_origin,

    "request_contract":
        "forecast_origin + features",

    "request_payload_feature_count":
        len(feature_payload),

    "input_file":
        TEST_FILE,

    "input_rows":
        len(test_df),

    "input_columns":
        len(test_df.columns),

    "api_health":
        health_json,

    "api_metadata":
        metadata_json,

    "prediction_response":
        prediction_json,

    "forecast_validation": {
        "rows":
            len(prediction_rows),

        "expected_rows":
            FORECAST_HORIZON,

        "horizons":
            horizons,

        "horizon_sequence_pass":
            horizons
            == expected_horizons,

        "duplicate_horizons":
            len(
                set(horizons)
            )
            != FORECAST_HORIZON,

        "prediction_values_finite":
            all(
                is_finite_number(
                    value
                )
                for value in predictions
            ),

        "negative_predictions":
            len(
                negative_predictions
            ),

        "timestamps_available":
            timestamps_available,

        "timestamp_validation":
            timestamp_validation,
    },

    "forecast_summary": {
        "minimum_predicted_aqi":
            minimum_prediction,

        "maximum_predicted_aqi":
            maximum_prediction,

        "mean_predicted_aqi":
            mean_prediction,

        "median_predicted_aqi":
            median_prediction,

        "standard_deviation":
            std_prediction,

        "predictions": [
            {
                "horizon":
                    horizons[index],

                "timestamp":
                    (
                        timestamps[index]
                        if timestamps[index]
                        is not None
                        else None
                    ),

                "predicted_aqi":
                    predictions[index],
            }
            for index in range(
                len(predictions)
            )
        ],
    },

    "step15_comparison":
        step15_comparison,

    "step14_comparison":
        step14_comparison,

    "contract_checks":
        api_contract_checks,

    "failed_checks":
        failed_checks,

    "timing": {
        "health_seconds":
            round(
                health_time,
                4,
            ),

        "metadata_seconds":
            round(
                metadata_time,
                4,
            ),

        "prediction_seconds":
            round(
                prediction_time,
                4,
            ),

        "total_seconds":
            round(
                health_time
                + metadata_time
                + prediction_time,
                4,
            ),
    },

    "production_data_usage": {
        "model_selection":
            False,

        "hyperparameter_tuning":
            False,

        "model_retraining":
            False,

        "validation":
            False,

        "test_set_for_selection":
            False,

        "training":
            False,
    },

    "status":
        "READY",
}


report = make_json_safe(
    report
)


try:

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

except Exception as exc:

    fail(
        "Unable to save API test report:\n"
        f"{exc}"
    )


print(
    "API test report saved:"
)

print(
    REPORT_FILE
)


# ======================================================================
# FINAL RESULT
# ======================================================================

print("\n" + "=" * 70)
print("STEP 17 — XGBOOST PRODUCTION API TEST COMPLETE")
print("=" * 70)

print(
    "API status           : READY"
)

print(
    "Health endpoint      : PASS"
)

print(
    "Metadata endpoint    : PASS"
)

print(
    "Prediction endpoint  : PASS"
)

print(
    f"Models verified      : "
    f"{FORECAST_HORIZON}"
)

print(
    f"Features verified    : "
    f"{EXPECTED_FEATURE_COUNT}"
)

print(
    f"Forecast rows        : "
    f"{len(prediction_rows)}"
)

print(
    "Horizon sequence     : PASS"
)

print(
    "Forecast origin      : PASS"
)

print(
    "Predictions finite   : PASS"
)

print(
    f"Negative AQI values  : "
    f"{len(negative_predictions)}"
)


if timestamps_available:

    print(
        "Timestamp continuity : PASS"
    )

else:

    print(
        "Timestamp continuity : "
        "NOT RETURNED BY API"
    )


print(
    f"\nMinimum predicted AQI: "
    f"{minimum_prediction:.3f}"
)

print(
    f"Maximum predicted AQI: "
    f"{maximum_prediction:.3f}"
)

print(
    f"Mean predicted AQI   : "
    f"{mean_prediction:.3f}"
)

print(
    f"Median predicted AQI : "
    f"{median_prediction:.3f}"
)


print()

print(
    "Model selection      : NOT PERFORMED"
)

print(
    "Hyperparameter tuning: NOT PERFORMED"
)

print(
    "Model retraining     : NOT PERFORMED"
)

print(
    "Validation           : NOT PERFORMED"
)

print(
    "Test-set selection   : NOT PERFORMED"
)

print(
    "\nAPI integration test : PASS"
)

print(
    "\nReport:"
)

print(
    REPORT_FILE
)

print(
    "\n" + "=" * 70
)