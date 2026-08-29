"""
PEARLS AQI PREDICTOR
STEP 16 — XGBOOST PRODUCTION API

Purpose
-------
Expose the locked Step 15 XGBoost production service through a
stable HTTP API.

Deployment contract
-------------------
- Deployment candidate: XGBoost tuned
- 72 independent horizon models
- Exact 101-feature Step 13 protocol
- Locked feature order
- No model selection
- No hyperparameter tuning
- No model retraining
- No validation-set usage
- No test-set usage

Endpoints
---------
GET  /health
    Service health and artifact status.

GET  /metadata
    Deployment and feature contract metadata.

POST /predict
    Generate a 72-hour AQI forecast.

Request formats
---------------
Named features:

{
    "features": {
        "feature_001": 1.23,
        "feature_002": 4.56,
        ...
    }
}

Ordered feature vector:

{
    "features": [
        1.23,
        4.56,
        ...
    ]
}

Optional request field:
    "forecast_origin": "2026-08-25T23:00:00"

If forecast_origin is omitted, the API uses the current UTC
timestamp.

Run
---
python scripts\\16_xgboost_production_api.py

Default:
    http://127.0.0.1:8000

Optional environment variables:
    PAQI_API_HOST
    PAQI_API_PORT
"""

import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from xgboost import XGBRegressor


# ======================================================================
# CONFIGURATION
# ======================================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
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

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "artifacts",
    "xgboost_tuned",
)

TARGET_COLUMN = "us_aqi"

FORECAST_HORIZON = 72

EXPECTED_FEATURE_COUNT = 101

RANDOM_STATE = 42

DEFAULT_HOST = "127.0.0.1"

DEFAULT_PORT = 8000


# ======================================================================
# APPLICATION
# ======================================================================

app = FastAPI(
    title="Pearls AQI Predictor",
    description=(
        "Production API for the locked XGBoost AQI forecasting service."
    ),
    version="1.0.0",
)


# ======================================================================
# GLOBAL DEPLOYMENT STATE
# ======================================================================

FEATURE_COLUMNS = []

DEPLOYMENT_PARAMETERS = {}

MODELS = {}

DEPLOYMENT_CONFIGURATION = None

ARTIFACT_STATUS = "NOT_INITIALIZED"

INITIALIZATION_ERROR = None

SERVICE_INITIALIZED_AT = None


# ======================================================================
# REQUEST / RESPONSE MODELS
# ======================================================================

class PredictionRequest(BaseModel):
    """
    Production prediction request.

    features may be:
        1. A dictionary mapping exact feature names to values.
        2. A list containing values in the exact Step 13 feature order.
    """

    features: object = Field(
        ...,
        description=(
            "Either a dictionary of 101 named features or an ordered "
            "list of 101 feature values."
        ),
    )

    forecast_origin: str | None = Field(
        default=None,
        description=(
            "Optional ISO-8601 forecast origin timestamp."
        ),
    )


class HealthResponse(BaseModel):
    status: str
    deployment_candidate: str | None
    models_loaded: int
    expected_models: int
    features_locked: int
    expected_features: int
    artifact_status: str


# ======================================================================
# UTILITY FUNCTIONS
# ======================================================================

def fail(message):
    """
    Raise a deployment initialization error.
    """

    raise RuntimeError(message)


def load_json(path):
    """
    Load a JSON file.
    """

    if not os.path.exists(path):
        fail(
            f"Required deployment contract not found: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def validate_step13_contract(report):
    """
    Validate the Step 13 deployment contract.

    The API deliberately does not recreate or infer the feature list.
    """

    global FEATURE_COLUMNS
    global DEPLOYMENT_PARAMETERS
    global DEPLOYMENT_CONFIGURATION

    feature_columns = report.get(
        "feature_columns"
    )

    if not feature_columns:
        fail(
            "Step 13 report does not contain "
            "'feature_columns'."
        )

    if not isinstance(
        feature_columns,
        list,
    ):
        fail(
            "Step 13 'feature_columns' must be a list."
        )

    if len(feature_columns) != EXPECTED_FEATURE_COUNT:
        fail(
            "\nStep 13 feature protocol mismatch.\n"
            f"Expected: {EXPECTED_FEATURE_COUNT}\n"
            f"Found   : {len(feature_columns)}"
        )

    if len(set(feature_columns)) != len(feature_columns):
        fail(
            "Step 13 feature protocol contains duplicate "
            "feature names."
        )

    parameters = report.get(
        "best_parameters"
    )

    if not parameters:
        parameters = report.get(
            "best_tuning_parameters"
        )

    if not parameters:
        fail(
            "Step 13 report does not contain the locked "
            "XGBoost parameters."
        )

    selected_configuration = report.get(
        "selected_configuration"
    )

    if selected_configuration != "tuned":
        fail(
            "Deployment contract mismatch.\n"
            f"Expected selected_configuration='tuned'.\n"
            f"Found: {selected_configuration!r}"
        )

    required_parameters = [
        "n_estimators",
        "learning_rate",
        "max_depth",
        "min_child_weight",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
    ]

    missing_parameters = [
        name
        for name in required_parameters
        if name not in parameters
    ]

    if missing_parameters:
        fail(
            "Step 13 deployment parameters are incomplete.\n"
            f"Missing: {missing_parameters}"
        )

    FEATURE_COLUMNS = list(
        feature_columns
    )

    DEPLOYMENT_PARAMETERS = {
        "n_estimators": int(
            parameters["n_estimators"]
        ),
        "learning_rate": float(
            parameters["learning_rate"]
        ),
        "max_depth": int(
            parameters["max_depth"]
        ),
        "min_child_weight": int(
            parameters["min_child_weight"]
        ),
        "subsample": float(
            parameters["subsample"]
        ),
        "colsample_bytree": float(
            parameters["colsample_bytree"]
        ),
        "reg_alpha": float(
            parameters["reg_alpha"]
        ),
        "reg_lambda": float(
            parameters["reg_lambda"]
        ),
    }

    DEPLOYMENT_CONFIGURATION = (
        selected_configuration
    )


def validate_step15_contract():
    """
    Validate that the Step 15 production service report exists.

    Step 15 is used as an additional deployment-contract check.
    """

    report = load_json(
        STEP15_REPORT_FILE
    )

    candidate = report.get(
        "deployment_candidate"
    )

    if candidate != "XGBoost tuned":
        fail(
            "Step 15 deployment candidate mismatch.\n"
            f"Expected: 'XGBoost tuned'\n"
            f"Found   : {candidate!r}"
        )

    models_loaded = report.get(
        "models_loaded"
    )

    if models_loaded is not None:
        if int(models_loaded) != FORECAST_HORIZON:
            fail(
                "Step 15 model count mismatch.\n"
                f"Expected: {FORECAST_HORIZON}\n"
                f"Found   : {models_loaded}"
            )

    features_used = report.get(
        "features_used"
    )

    if features_used is not None:
        if int(features_used) != EXPECTED_FEATURE_COUNT:
            fail(
                "Step 15 feature count mismatch.\n"
                f"Expected: {EXPECTED_FEATURE_COUNT}\n"
                f"Found   : {features_used}"
            )

    return report


def create_model():
    """
    Create an XGBoost model container using the locked parameters.

    No fitting occurs here.
    """

    return XGBRegressor(
        n_estimators=DEPLOYMENT_PARAMETERS[
            "n_estimators"
        ],

        learning_rate=DEPLOYMENT_PARAMETERS[
            "learning_rate"
        ],

        max_depth=DEPLOYMENT_PARAMETERS[
            "max_depth"
        ],

        min_child_weight=DEPLOYMENT_PARAMETERS[
            "min_child_weight"
        ],

        subsample=DEPLOYMENT_PARAMETERS[
            "subsample"
        ],

        colsample_bytree=DEPLOYMENT_PARAMETERS[
            "colsample_bytree"
        ],

        reg_alpha=DEPLOYMENT_PARAMETERS[
            "reg_alpha"
        ],

        reg_lambda=DEPLOYMENT_PARAMETERS[
            "reg_lambda"
        ],

        objective="reg:squarederror",

        random_state=RANDOM_STATE,

        n_jobs=-1,

        tree_method="hist",
    )


def load_models():
    """
    Load all 72 locked XGBoost model artifacts.
    """

    global MODELS

    MODELS = {}

    missing_models = []

    for horizon in range(
        1,
        FORECAST_HORIZON + 1,
    ):

        model_path = os.path.join(
            MODEL_DIR,
            f"xgboost_tuned_{horizon:02d}h.json",
        )

        if not os.path.exists(model_path):
            missing_models.append(
                model_path
            )
            continue

        model = create_model()

        model.load_model(
            model_path
        )

        MODELS[horizon] = model

    if missing_models:
        fail(
            "Missing XGBoost production artifacts.\n"
            f"Missing count: {len(missing_models)}\n"
            f"First missing: {missing_models[0]}"
        )

    if len(MODELS) != FORECAST_HORIZON:
        fail(
            "Incorrect number of loaded XGBoost models.\n"
            f"Expected: {FORECAST_HORIZON}\n"
            f"Loaded  : {len(MODELS)}"
        )


def initialize_service():
    """
    Initialize the immutable production model state.
    """

    global ARTIFACT_STATUS
    global INITIALIZATION_ERROR
    global SERVICE_INITIALIZED_AT

    try:

        print("=" * 70)
        print("PEARLS AQI PREDICTOR")
        print("STEP 16 — XGBOOST PRODUCTION API")
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

        # --------------------------------------------------------------
        # STEP 13 CONTRACT
        # --------------------------------------------------------------

        print("\n" + "=" * 70)
        print("LOADING LOCKED DEPLOYMENT CONTRACT")
        print("=" * 70)

        step13_report = load_json(
            STEP13_REPORT_FILE
        )

        validate_step13_contract(
            step13_report
        )

        print(
            f"Selected configuration : "
            f"{DEPLOYMENT_CONFIGURATION}"
        )

        print(
            "Deployment parameters:"
        )

        print(
            json.dumps(
                DEPLOYMENT_PARAMETERS,
                indent=2,
            )
        )

        print(
            f"\nStep 13 feature count : "
            f"{len(FEATURE_COLUMNS)}"
        )

        print(
            f"Expected feature count: "
            f"{EXPECTED_FEATURE_COUNT}"
        )

        print(
            "Feature protocol     : PASS"
        )

        print(
            "Exact 101-feature set: LOCKED"
        )

        # --------------------------------------------------------------
        # STEP 15 CONTRACT
        # --------------------------------------------------------------

        print("\n" + "=" * 70)
        print("VERIFYING STEP 15 SERVICE CONTRACT")
        print("=" * 70)

        validate_step15_contract()

        print(
            "Step 15 service contract: PASS"
        )

        print(
            "Model selection        : NOT PERFORMED"
        )

        print(
            "Hyperparameter tuning  : NOT PERFORMED"
        )

        print(
            "Model retraining       : NOT PERFORMED"
        )

        print(
            "Validation             : NOT PERFORMED"
        )

        print(
            "Test-set usage         : NOT PERFORMED"
        )

        # --------------------------------------------------------------
        # MODEL ARTIFACTS
        # --------------------------------------------------------------

        print("\n" + "=" * 70)
        print("LOADING LOCKED XGBOOST MODELS")
        print("=" * 70)

        load_models()

        for horizon in range(
            1,
            FORECAST_HORIZON + 1,
        ):

            print(
                f"{horizon:02d}h model       : LOADED"
            )

        print(
            f"\nModels loaded        : "
            f"{len(MODELS)}"
        )

        print(
            "Model artifact check : PASS"
        )

        ARTIFACT_STATUS = "READY"

        SERVICE_INITIALIZED_AT = (
            datetime.now(
                timezone.utc
            ).isoformat()
        )

        print("\n" + "=" * 70)
        print("STEP 16 API INITIALIZATION COMPLETE")
        print("=" * 70)

        print(
            "API artifact status : READY"
        )

        print(
            f"Models loaded       : "
            f"{len(MODELS)}"
        )

        print(
            f"Features locked     : "
            f"{len(FEATURE_COLUMNS)}"
        )

        print(
            "\nNo training or model selection is "
            "performed by this API."
        )

    except Exception as exc:

        ARTIFACT_STATUS = "FAILED"

        INITIALIZATION_ERROR = str(
            exc
        )

        print("\n" + "=" * 70)
        print("STEP 16 INITIALIZATION FAILED")
        print("=" * 70)

        print(
            f"ERROR: {exc}"
        )

        raise


# ======================================================================
# FEATURE INPUT VALIDATION
# ======================================================================

def build_feature_matrix(features):
    """
    Convert an API feature payload into the exact locked
    101-feature matrix.
    """

    if isinstance(
        features,
        dict,
    ):

        missing = [
            feature
            for feature in FEATURE_COLUMNS
            if feature not in features
        ]

        if missing:
            raise HTTPException(
                status_code=422,
                detail={
                    "error":
                        "Missing required features.",
                    "missing_count":
                        len(missing),
                    "missing_features":
                        missing,
                },
            )

        extra = [
            feature
            for feature in features
            if feature not in FEATURE_COLUMNS
        ]

        if extra:
            raise HTTPException(
                status_code=422,
                detail={
                    "error":
                        "Unknown feature(s) supplied.",
                    "extra_count":
                        len(extra),
                    "extra_features":
                        extra[:25],
                },
            )

        ordered_values = [
            features[feature]
            for feature in FEATURE_COLUMNS
        ]

    elif isinstance(
        features,
        list,
    ):

        if len(features) != EXPECTED_FEATURE_COUNT:
            raise HTTPException(
                status_code=422,
                detail={
                    "error":
                        "Ordered feature vector has "
                        "incorrect length.",
                    "expected":
                        EXPECTED_FEATURE_COUNT,
                    "received":
                        len(features),
                },
            )

        ordered_values = list(
            features
        )

    else:

        raise HTTPException(
            status_code=422,
            detail={
                "error":
                    "'features' must be either "
                    "an object or an array."
            },
        )

    # --------------------------------------------------------------
    # NUMERIC CONVERSION
    # --------------------------------------------------------------

    try:

        numeric_values = [
            float(value)
            for value in ordered_values
        ]

    except Exception as exc:

        raise HTTPException(
            status_code=422,
            detail={
                "error":
                    "All feature values must be numeric.",
                "message":
                    str(exc),
            },
        )

    array = np.asarray(
        numeric_values,
        dtype=np.float32,
    )

    if array.shape != (
        EXPECTED_FEATURE_COUNT,
    ):

        raise HTTPException(
            status_code=422,
            detail={
                "error":
                    "Feature vector shape mismatch.",
                "expected":
                    EXPECTED_FEATURE_COUNT,
                "received":
                    int(array.size),
            },
        )

    if not np.isfinite(
        array
    ).all():

        raise HTTPException(
            status_code=422,
            detail={
                "error":
                    "Feature vector contains "
                    "NaN or infinite values."
            },
        )

    return array.reshape(
        1,
        EXPECTED_FEATURE_COUNT,
    )


# ======================================================================
# FORECAST ORIGIN VALIDATION
# ======================================================================

def resolve_forecast_origin(value):
    """
    Resolve and validate the forecast origin.

    The API accepts ISO-8601 timestamps.
    """

    if value is None:

        return datetime.now(
            timezone.utc
        )

    try:

        normalized = value.strip()

        if normalized.endswith(
            "Z"
        ):
            normalized = (
                normalized[:-1]
                + "+00:00"
            )

        parsed = datetime.fromisoformat(
            normalized
        )

    except Exception as exc:

        raise HTTPException(
            status_code=422,
            detail={
                "error":
                    "Invalid forecast_origin. "
                    "Use ISO-8601 format.",
                "message":
                    str(exc),
            },
        )

    if parsed.tzinfo is None:

        parsed = parsed.replace(
            tzinfo=timezone.utc
        )

    return parsed


# ======================================================================
# PRODUCTION INFERENCE
# ======================================================================

def run_inference(
    X_production,
    forecast_origin,
):
    """
    Generate the 72-hour forecast using only the locked artifacts.
    """

    forecast_rows = []

    inference_start = time.time()

    for horizon in range(
        1,
        FORECAST_HORIZON + 1,
    ):

        model = MODELS[
            horizon
        ]

        prediction = model.predict(
            X_production
        )

        value = float(
            prediction[0]
        )

        if not np.isfinite(
            value
        ):

            raise HTTPException(
                status_code=500,
                detail={
                    "error":
                        "Model produced a "
                        "non-finite prediction.",
                    "horizon":
                        horizon,
                },
            )

        if value < 0:

            raise HTTPException(
                status_code=500,
                detail={
                    "error":
                        "Model produced a "
                        "negative AQI prediction.",
                    "horizon":
                        horizon,
                    "prediction":
                        value,
                },
            )

        forecast_timestamp = (
            forecast_origin
            + pd.Timedelta(
                hours=horizon
            )
        )

        forecast_rows.append(
            {
                "horizon": horizon,

                "timestamp":
                    forecast_timestamp.isoformat(),

                "predicted_aqi":
                    value,
            }
        )

    inference_time = (
        time.time()
        - inference_start
    )

    predictions = [
        row["predicted_aqi"]
        for row in forecast_rows
    ]

    return (
        forecast_rows,
        inference_time,
        predictions,
    )


# ======================================================================
# HEALTH ENDPOINT
# ======================================================================

@app.get(
    "/health",
    response_model=HealthResponse,
)
def health():
    """
    Service health endpoint.
    """

    return HealthResponse(
        status=(
            "ok"
            if ARTIFACT_STATUS == "READY"
            else "error"
        ),

        deployment_candidate=(
            "XGBoost tuned"
            if ARTIFACT_STATUS == "READY"
            else None
        ),

        models_loaded=len(
            MODELS
        ),

        expected_models=
            FORECAST_HORIZON,

        features_locked=len(
            FEATURE_COLUMNS
        ),

        expected_features=
            EXPECTED_FEATURE_COUNT,

        artifact_status=
            ARTIFACT_STATUS,
    )


# ======================================================================
# METADATA ENDPOINT
# ======================================================================

@app.get(
    "/metadata"
)
def metadata():
    """
    Return the immutable deployment contract.
    """

    if ARTIFACT_STATUS != "READY":

        raise HTTPException(
            status_code=503,
            detail={
                "error":
                    "Production model artifacts "
                    "are not ready.",
                "initialization_error":
                    INITIALIZATION_ERROR,
            },
        )

    return {
        "service":
            "Pearls AQI Predictor",

        "step":
            16,

        "service_type":
            "production_api",

        "status":
            "READY",

        "deployment_candidate":
            "XGBoost tuned",

        "target":
            TARGET_COLUMN,

        "forecast_horizon":
            FORECAST_HORIZON,

        "models_loaded":
            len(MODELS),

        "features_used":
            len(FEATURE_COLUMNS),

        "feature_protocol":
            "Exact Step 13 feature list",

        "feature_order_locked":
            True,

        "feature_columns":
            FEATURE_COLUMNS,

        "deployment_parameters":
            DEPLOYMENT_PARAMETERS,

        "model_selection":
            False,

        "hyperparameter_tuning":
            False,

        "model_retraining":
            False,

        "validation":
            False,

        "test_set_usage":
            False,

        "model_directory":
            MODEL_DIR,

        "initialized_at":
            SERVICE_INITIALIZED_AT,
    }


# ======================================================================
# PREDICTION ENDPOINT
# ======================================================================

@app.post(
    "/predict"
)
def predict(
    request: PredictionRequest,
):
    """
    Generate a 72-hour AQI forecast.
    """

    if ARTIFACT_STATUS != "READY":

        raise HTTPException(
            status_code=503,
            detail={
                "error":
                    "Production service is not ready.",
                "artifact_status":
                    ARTIFACT_STATUS,
                "initialization_error":
                    INITIALIZATION_ERROR,
            },
        )

    request_start = time.time()

    # --------------------------------------------------------------
    # FEATURE MATRIX
    # --------------------------------------------------------------

    X_production = (
        build_feature_matrix(
            request.features
        )
    )

    # --------------------------------------------------------------
    # FORECAST ORIGIN
    # --------------------------------------------------------------

    forecast_origin = (
        resolve_forecast_origin(
            request.forecast_origin
        )
    )

    # --------------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------------

    try:

        (
            forecast_rows,
            inference_time,
            predictions,
        ) = run_inference(
            X_production,
            forecast_origin,
        )

    except HTTPException:
        raise

    except Exception as exc:

        print(
            traceback.format_exc()
        )

        raise HTTPException(
            status_code=500,
            detail={
                "error":
                    "Production inference failed.",
                "message":
                    str(exc),
            },
        )

    # --------------------------------------------------------------
    # FINAL OUTPUT VALIDATION
    # --------------------------------------------------------------

    if len(
        forecast_rows
    ) != FORECAST_HORIZON:

        raise HTTPException(
            status_code=500,
            detail={
                "error":
                    "Incorrect forecast length.",
                "expected":
                    FORECAST_HORIZON,
                "received":
                    len(forecast_rows),
            },
        )

    horizons = [
        row["horizon"]
        for row in forecast_rows
    ]

    if horizons != list(
        range(
            1,
            FORECAST_HORIZON + 1,
        )
    ):

        raise HTTPException(
            status_code=500,
            detail={
                "error":
                    "Forecast horizon sequence "
                    "validation failed."
            },
        )

    if not np.isfinite(
        np.asarray(
            predictions,
            dtype=np.float64,
        )
    ).all():

        raise HTTPException(
            status_code=500,
            detail={
                "error":
                    "Forecast contains "
                    "non-finite values."
            },
        )

    if any(
        value < 0
        for value in predictions
    ):

        raise HTTPException(
            status_code=500,
            detail={
                "error":
                    "Forecast contains "
                    "negative AQI values."
            },
        )

    total_time = (
        time.time()
        - request_start
    )

    return {
        "status":
            "success",

        "service":
            "Pearls AQI Predictor",

        "deployment_candidate":
            "XGBoost tuned",

        "target":
            TARGET_COLUMN,

        "forecast_origin":
            forecast_origin.isoformat(),

        "forecast_horizon_hours":
            FORECAST_HORIZON,

        "forecast_start":
            forecast_rows[0][
                "timestamp"
            ],

        "forecast_end":
            forecast_rows[-1][
                "timestamp"
            ],

        "features_used":
            EXPECTED_FEATURE_COUNT,

        "model_count":
            FORECAST_HORIZON,

        "predictions":
            forecast_rows,

        "summary": {
            "minimum_predicted_aqi":
                float(
                    np.min(
                        predictions
                    )
                ),

            "maximum_predicted_aqi":
                float(
                    np.max(
                        predictions
                    )
                ),

            "mean_predicted_aqi":
                float(
                    np.mean(
                        predictions
                    )
                ),

            "one_hour_aqi":
                predictions[0],

            "twenty_four_hour_aqi":
                predictions[23],

            "forty_eight_hour_aqi":
                predictions[47],

            "seventy_two_hour_aqi":
                predictions[71],
        },

        "timing": {
            "inference_seconds":
                round(
                    inference_time,
                    4,
                ),

            "total_request_seconds":
                round(
                    total_time,
                    4,
                ),
        },

        "model_selection":
            False,

        "hyperparameter_tuning":
            False,

        "model_retraining":
            False,

        "validation":
            False,

        "test_set_usage":
            False,
    }


# ======================================================================
# STARTUP
# ======================================================================

initialize_service()


# ======================================================================
# MAIN
# ======================================================================

if __name__ == "__main__":

    host = os.environ.get(
        "PAQI_API_HOST",
        DEFAULT_HOST,
    )

    try:

        port = int(
            os.environ.get(
                "PAQI_API_PORT",
                DEFAULT_PORT,
            )
        )

    except ValueError:

        raise ValueError(
            "PAQI_API_PORT must be an integer."
        )

    print("\n" + "=" * 70)
    print("STARTING PEARLS AQI PRODUCTION API")
    print("=" * 70)

    print(
        f"Host                : {host}"
    )

    print(
        f"Port                : {port}"
    )

    print(
        f"Health endpoint     : "
        f"http://{host}:{port}/health"
    )

    print(
        f"Metadata endpoint   : "
        f"http://{host}:{port}/metadata"
    )

    print(
        f"Prediction endpoint : "
        f"http://{host}:{port}/predict"
    )

    print(
        "\nPress CTRL+C to stop the API."
    )

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
    )