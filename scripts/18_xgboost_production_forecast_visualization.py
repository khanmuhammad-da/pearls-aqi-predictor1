"""
PEARLS AQI PREDICTOR
STEP 18 — XGBOOST PRODUCTION FORECAST VISUALIZATION

Purpose
-------
Consume the locked XGBoost production API forecast and create:

1. 72-hour forecast visualization
2. AQI category visualization
3. AQI risk/threshold visualization
4. 24-hour block visualization
5. Dashboard-ready CSV
6. Dashboard-ready JSON
7. Step 18 visualization report

Deployment contract
-------------------
- Production model: XGBoost tuned
- Forecast horizon: 72
- Target: us_aqi
- API: Step 16 production API
- No model selection
- No hyperparameter tuning
- No model retraining
- No validation usage
- No test-set usage
- No future target leakage

This step does NOT train a model.
It only consumes the already-validated production API forecast.
"""

import json
import os
import sys
import time
from datetime import datetime, timezone

import matplotlib.pyplot as plt
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

API_BASE_URL = os.environ.get(
    "PAQI_API_URL",
    "http://127.0.0.1:8000",
)

API_PREDICT_URL = (
    API_BASE_URL.rstrip("/")
    + "/predict"
)

FORECAST_HORIZON = 72

TARGET_COLUMN = "us_aqi"

EXPECTED_FEATURE_COUNT = 101

# Step 17 verified latest production row.
DEFAULT_FORECAST_ORIGIN = (
    "2026-08-25T23:00:00+00:00"
)

# Existing production feature container used by Step 17.
FEATURE_CONTAINER_CANDIDATES = [
    os.path.join(
        BASE_DIR,
        "data",
        "processed",
        "production_features.csv",
    ),
    os.path.join(
        BASE_DIR,
        "data",
        "processed",
        "production_feature_container.csv",
    ),
    os.path.join(
        BASE_DIR,
        "data",
        "production_features.csv",
    ),
    os.path.join(
        BASE_DIR,
        "data",
        "features",
        "production_features.csv",
    ),
]

# Outputs
REPORTS_DIR = os.path.join(
    BASE_DIR,
    "reports",
)

VISUALIZATION_DIR = os.path.join(
    REPORTS_DIR,
    "visualizations",
)

PREDICTION_DIR = os.path.join(
    REPORTS_DIR,
    "predictions",
)

DASHBOARD_CSV = os.path.join(
    REPORTS_DIR,
    "dashboard_ready_forecast.csv",
)

DASHBOARD_JSON = os.path.join(
    VISUALIZATION_DIR,
    "dashboard_data.json",
)

VISUALIZATION_REPORT = os.path.join(
    REPORTS_DIR,
    "production_visualization_results.json",
)

FORECAST_PNG = os.path.join(
    VISUALIZATION_DIR,
    "production_72h_forecast.png",
)

CATEGORY_PNG = os.path.join(
    VISUALIZATION_DIR,
    "production_72h_categories.png",
)

RISK_PNG = os.path.join(
    VISUALIZATION_DIR,
    "production_72h_risk.png",
)

BLOCK_PNG = os.path.join(
    VISUALIZATION_DIR,
    "production_24h_blocks.png",
)


# ======================================================================
# UTILITY FUNCTIONS
# ======================================================================

def banner(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def fail(message):
    print()
    print("=" * 70)
    print("STEP 18 FAILED")
    print("=" * 70)
    print(message)
    sys.exit(1)


def json_safe(value):
    """
    Convert NumPy/Pandas values into JSON-safe Python values.
    """

    if isinstance(value, dict):
        return {
            str(k): json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, list):
        return [
            json_safe(v)
            for v in value
        ]

    if isinstance(
        value,
        (
            np.integer,
            np.int64,
            np.int32,
        ),
    ):
        return int(value)

    if isinstance(
        value,
        (
            np.floating,
            np.float64,
            np.float32,
        ),
    ):
        return float(value)

    if isinstance(
        value,
        (
            np.bool_,
        ),
    ):
        return bool(value)

    if isinstance(
        value,
        (
            pd.Timestamp,
        ),
    ):
        return value.isoformat()

    return value


def classify_aqi(aqi):
    """
    US AQI category classification.
    """

    if aqi <= 50:
        return "Good"

    if aqi <= 100:
        return "Moderate"

    if aqi <= 150:
        return "Unhealthy for Sensitive Groups"

    if aqi <= 200:
        return "Unhealthy"

    if aqi <= 300:
        return "Very Unhealthy"

    return "Hazardous"


def category_order():
    return [
        "Good",
        "Moderate",
        "Unhealthy for Sensitive Groups",
        "Unhealthy",
        "Very Unhealthy",
        "Hazardous",
    ]


def find_feature_container():
    """
    Locate the production feature container.

    If the exact Step 17 file has a different location,
    the script can also use PAQI_FEATURE_FILE.
    """

    override = os.environ.get(
        "PAQI_FEATURE_FILE"
    )

    if override:
        path = os.path.abspath(override)

        if os.path.isfile(path):
            return path

        raise FileNotFoundError(
            "PAQI_FEATURE_FILE was supplied but does not exist: "
            + path
        )

    for path in FEATURE_CONTAINER_CANDIDATES:

        if os.path.isfile(path):
            return path

    raise FileNotFoundError(
        "Could not locate the production feature container. "
        "Set PAQI_FEATURE_FILE to the exact CSV used by Step 17."
    )


def load_latest_features():
    """
    Load the latest production row and extract the exact
    101 feature columns returned by the API metadata.
    """

    banner(
        "LOADING XGBOOST PRODUCTION FEATURE CONTAINER"
    )

    feature_file = find_feature_container()

    print(
        "Feature container   :",
        feature_file,
    )

    df = pd.read_csv(
        feature_file
    )

    if df.empty:
        raise ValueError(
            "Production feature container is empty."
        )

    print(
        "Input rows          :",
        len(df),
    )

    print(
        "Input columns       :",
        len(df.columns),
    )

    # ------------------------------------------------------------------
    # Timestamp column
    # ------------------------------------------------------------------

    timestamp_candidates = [
        "timestamp",
        "datetime",
        "date",
        "time",
    ]

    timestamp_column = None

    for column in timestamp_candidates:

        if column in df.columns:
            timestamp_column = column
            break

    if timestamp_column is None:
        raise ValueError(
            "No timestamp column found in production feature container."
        )

    df[timestamp_column] = pd.to_datetime(
        df[timestamp_column],
        utc=True,
        errors="coerce",
    )

    if df[timestamp_column].isna().any():
        raise ValueError(
            "Invalid timestamps found in production feature container."
        )

    df = df.sort_values(
        timestamp_column
    ).reset_index(
        drop=True
    )

    latest_row = df.iloc[-1]

    forecast_origin = latest_row[
        timestamp_column
    ]

    return (
        df,
        latest_row,
        timestamp_column,
        forecast_origin,
    )


def get_api_metadata():
    banner(
        "LOADING XGBOOST API METADATA"
    )

    response = requests.get(
        API_BASE_URL.rstrip("/")
        + "/metadata",
        timeout=15,
    )

    print(
        "HTTP status         :",
        response.status_code,
    )

    response.raise_for_status()

    metadata = response.json()

    if metadata.get(
        "status"
    ) != "READY":
        raise RuntimeError(
            "Production API metadata is not READY."
        )

    if metadata.get(
        "deployment_candidate"
    ) != "XGBoost tuned":
        raise RuntimeError(
            "Unexpected deployment candidate: "
            + str(
                metadata.get(
                    "deployment_candidate"
                )
            )
        )

    if int(
        metadata.get(
            "forecast_horizon",
            0,
        )
    ) != FORECAST_HORIZON:
        raise RuntimeError(
            "API forecast horizon is not 72."
        )

    if int(
        metadata.get(
            "features_used",
            0,
        )
    ) != EXPECTED_FEATURE_COUNT:
        raise RuntimeError(
            "API feature count is not 101."
        )

    if not metadata.get(
        "feature_order_locked",
        False,
    ):
        raise RuntimeError(
            "API feature order is not locked."
        )

    print(
        "Deployment candidate:",
        metadata.get(
            "deployment_candidate"
        ),
    )

    print(
        "Models loaded       :",
        metadata.get(
            "models_loaded"
        ),
    )

    print(
        "Features used       :",
        metadata.get(
            "features_used"
        ),
    )

    print(
        "Feature order locked:",
        metadata.get(
            "feature_order_locked"
        ),
    )

    feature_columns = metadata.get(
        "feature_columns"
    )

    if not isinstance(
        feature_columns,
        list,
    ):
        raise RuntimeError(
            "API metadata did not return feature_columns."
        )

    if len(feature_columns) != EXPECTED_FEATURE_COUNT:
        raise RuntimeError(
            "API feature_columns does not contain exactly 101 features."
        )

    return (
        metadata,
        feature_columns,
    )


def build_api_payload(
    latest_row,
    feature_columns,
    forecast_origin,
):
    banner(
        "BUILDING XGBOOST PRODUCTION API REQUEST"
    )

    missing = [
        feature
        for feature in feature_columns
        if feature not in latest_row.index
    ]

    if missing:
        raise ValueError(
            "Missing required API features: "
            + ", ".join(missing[:20])
        )

    features = {}

    for feature in feature_columns:

        value = latest_row[
            feature
        ]

        if pd.isna(value):
            raise ValueError(
                "Missing value in required feature: "
                + feature
            )

        value = float(value)

        if not np.isfinite(value):
            raise ValueError(
                "Non-finite value in required feature: "
                + feature
            )

        features[
            feature
        ] = value

    origin = pd.Timestamp(
        forecast_origin
    )

    if origin.tzinfo is None:
        origin = origin.tz_localize(
            "UTC"
        )

    origin = origin.tz_convert(
        "UTC"
    )

    payload = {
        "features": features,
        "forecast_origin": origin.isoformat(),
    }

    print(
        "Request features    :",
        len(features),
    )

    print(
        "Forecast origin     :",
        origin.isoformat(),
    )

    return payload


def call_prediction_api(payload):
    banner(
        "REQUESTING XGBOOST 72-HOUR PRODUCTION FORECAST"
    )

    start = time.time()

    response = requests.post(
        API_PREDICT_URL,
        json=payload,
        timeout=30,
    )

    elapsed = time.time() - start

    print(
        "HTTP status         :",
        response.status_code,
    )

    print(
        "Response time       :",
        f"{elapsed:.4f}s",
    )

    response.raise_for_status()

    data = response.json()

    if data.get(
        "status"
    ) != "success":
        raise RuntimeError(
            "Prediction API did not return success."
        )

    return (
        data,
        elapsed,
    )


def validate_forecast(
    prediction_response,
    requested_origin,
):
    banner(
        "VALIDATING XGBOOST FORECAST"
    )

    predictions = prediction_response.get(
        "predictions"
    )

    if not isinstance(
        predictions,
        list,
    ):
        raise ValueError(
            "API predictions field is not a list."
        )

    if len(predictions) != FORECAST_HORIZON:
        raise ValueError(
            "Expected 72 predictions but received "
            + str(len(predictions))
        )

    horizons = [
        int(row["horizon"])
        for row in predictions
    ]

    expected_horizons = list(
        range(
            1,
            FORECAST_HORIZON + 1,
        )
    )

    if horizons != expected_horizons:
        raise ValueError(
            "Horizon sequence is not exactly 1..72."
        )

    timestamps = pd.to_datetime(
        [
            row["timestamp"]
            for row in predictions
        ],
        utc=True,
        errors="coerce",
    )

    if timestamps.isna().any():
        raise ValueError(
            "Invalid forecast timestamp found."
        )

    if timestamps.duplicated().any():
        raise ValueError(
            "Duplicate forecast timestamps found."
        )

    timestamp_diffs = timestamps.to_series().diff().dropna()

    if not (
        timestamp_diffs
        == pd.Timedelta(hours=1)
    ).all():
        raise ValueError(
            "Forecast timestamps are not continuous hourly timestamps."
        )

    values = np.asarray(
        [
            float(row["predicted_aqi"])
            for row in predictions
        ],
        dtype=float,
    )

    if not np.isfinite(values).all():
        raise ValueError(
            "Forecast contains non-finite AQI values."
        )

    if (values < 0).any():
        raise ValueError(
            "Forecast contains negative AQI values."
        )

    origin_returned = pd.Timestamp(
        prediction_response[
            "forecast_origin"
        ]
    )

    origin_requested = pd.Timestamp(
        requested_origin
    )

    if origin_requested.tzinfo is None:
        origin_requested = (
            origin_requested.tz_localize(
                "UTC"
            )
        )

    origin_requested = (
        origin_requested.tz_convert(
            "UTC"
        )
    )

    if origin_returned != origin_requested:
        raise ValueError(
            "API returned a different forecast origin."
        )

    forecast_df = pd.DataFrame(
        {
            "horizon": horizons,
            "timestamp": timestamps,
            "predicted_aqi": values,
        }
    )

    forecast_df[
        "aqi_category"
    ] = forecast_df[
        "predicted_aqi"
    ].apply(
        classify_aqi
    )

    print(
        "Forecast rows       :",
        len(forecast_df),
    )

    print(
        "Horizon sequence    : PASS"
    )

    print(
        "Timestamp continuity: PASS"
    )

    print(
        "Prediction values   : PASS"
    )

    print(
        "Negative AQI values :",
        int((values < 0).sum()),
    )

    return forecast_df


# ======================================================================
# VISUALIZATIONS
# ======================================================================

def create_forecast_plot(df):
    banner(
        "CREATING 72-HOUR AQI FORECAST VISUALIZATION"
    )

    plt.figure(
        figsize=(14, 6)
    )

    plt.plot(
        df["timestamp"],
        df["predicted_aqi"],
        marker="o",
        markersize=3,
        linewidth=1.5,
    )

    plt.axhline(
        50,
        linestyle="--",
        linewidth=1,
        label="AQI 50",
    )

    plt.axhline(
        100,
        linestyle="--",
        linewidth=1,
        label="AQI 100",
    )

    plt.axhline(
        150,
        linestyle="--",
        linewidth=1,
        label="AQI 150",
    )

    plt.axhline(
        200,
        linestyle="--",
        linewidth=1,
        label="AQI 200",
    )

    plt.axhline(
        300,
        linestyle="--",
        linewidth=1,
        label="AQI 300",
    )

    plt.title(
        "Pearls AQI Predictor — XGBoost 72-Hour Production Forecast"
    )

    plt.xlabel(
        "Forecast Timestamp"
    )

    plt.ylabel(
        "Predicted US AQI"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.xticks(
        rotation=30,
        ha="right",
    )

    plt.tight_layout()

    plt.savefig(
        FORECAST_PNG,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "Forecast plot saved:",
        FORECAST_PNG,
    )


def create_category_plot(df):
    banner(
        "CREATING AQI CATEGORY VISUALIZATION"
    )

    counts = (
        df["aqi_category"]
        .value_counts()
        .reindex(
            category_order(),
            fill_value=0,
        )
    )

    plt.figure(
        figsize=(12, 6)
    )

    plt.bar(
        counts.index,
        counts.values,
    )

    plt.title(
        "XGBoost 72-Hour AQI Category Distribution"
    )

    plt.xlabel(
        "AQI Category"
    )

    plt.ylabel(
        "Forecast Hours"
    )

    plt.xticks(
        rotation=25,
        ha="right",
    )

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        CATEGORY_PNG,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "Category plot saved:",
        CATEGORY_PNG,
    )


def create_risk_plot(df):
    banner(
        "CREATING AQI RISK VISUALIZATION"
    )

    thresholds = [
        50,
        100,
        150,
        200,
        300,
    ]

    counts = {
        f"AQI >= {threshold}": int(
            (
                df["predicted_aqi"]
                >= threshold
            ).sum()
        )
        for threshold in thresholds
    }

    labels = list(
        counts.keys()
    )

    values = list(
        counts.values()
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        labels,
        values,
    )

    plt.title(
        "XGBoost 72-Hour AQI Risk Thresholds"
    )

    plt.xlabel(
        "AQI Threshold"
    )

    plt.ylabel(
        "Forecast Hours"
    )

    plt.xticks(
        rotation=20,
        ha="right",
    )

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        RISK_PNG,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "Risk plot saved:",
        RISK_PNG,
    )

    return counts


def create_24h_block_plot(df):
    banner(
        "CREATING 24-HOUR BLOCK VISUALIZATION"
    )

    working = df.copy()

    working[
        "block"
    ] = (
        (working["horizon"] - 1)
        // 24
    ) + 1

    block_summary = (
        working
        .groupby("block")[
            "predicted_aqi"
        ]
        .agg(
            [
                "mean",
                "min",
                "max",
            ]
        )
        .reset_index()
    )

    labels = [
        f"Hours {(block - 1) * 24 + 1}-"
        f"{min(block * 24, 72)}"
        for block in block_summary[
            "block"
        ]
    ]

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        labels,
        block_summary["mean"],
    )

    plt.title(
        "XGBoost 24-Hour AQI Block Means"
    )

    plt.xlabel(
        "Forecast Block"
    )

    plt.ylabel(
        "Mean Predicted AQI"
    )

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        BLOCK_PNG,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "24-hour block plot saved:",
        BLOCK_PNG,
    )

    return block_summary


# ======================================================================
# DASHBOARD OUTPUTS
# ======================================================================

def save_dashboard_outputs(
    df,
    risk_counts,
    block_summary,
    api_response,
    metadata,
):
    banner(
        "SAVING DASHBOARD-READY FORECAST"
    )

    dashboard_df = df.copy()

    dashboard_df[
        "timestamp"
    ] = dashboard_df[
        "timestamp"
    ].dt.strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    dashboard_df.to_csv(
        DASHBOARD_CSV,
        index=False,
    )

    print(
        "Dashboard CSV saved:",
        DASHBOARD_CSV,
    )

    values = df[
        "predicted_aqi"
    ].to_numpy(
        dtype=float
    )

    category_counts = (
        df["aqi_category"]
        .value_counts()
        .reindex(
            category_order(),
            fill_value=0,
        )
    )

    dominant_category = (
        category_counts.idxmax()
    )

    summary = {
        "minimum_predicted_aqi":
            float(values.min()),

        "maximum_predicted_aqi":
            float(values.max()),

        "mean_predicted_aqi":
            float(values.mean()),

        "median_predicted_aqi":
            float(np.median(values)),

        "std_predicted_aqi":
            float(values.std()),

        "one_hour_aqi":
            float(
                df.iloc[0][
                    "predicted_aqi"
                ]
            ),

        "twenty_four_hour_aqi":
            float(
                df.iloc[23][
                    "predicted_aqi"
                ]
            ),

        "forty_eight_hour_aqi":
            float(
                df.iloc[47][
                    "predicted_aqi"
                ]
            ),

        "seventy_two_hour_aqi":
            float(
                df.iloc[71][
                    "predicted_aqi"
                ]
            ),

        "dominant_aqi_category":
            dominant_category,
    }

    category_distribution = {
        category: int(
            category_counts[
                category
            ]
        )
        for category in category_order()
    }

    blocks = []

    for _, row in block_summary.iterrows():

        block_number = int(
            row["block"]
        )

        blocks.append(
            {
                "block":
                    block_number,

                "start_horizon":
                    (block_number - 1)
                    * 24
                    + 1,

                "end_horizon":
                    min(
                        block_number * 24,
                        72,
                    ),

                "mean_predicted_aqi":
                    float(
                        row["mean"]
                    ),

                "minimum_predicted_aqi":
                    float(
                        row["min"]
                    ),

                "maximum_predicted_aqi":
                    float(
                        row["max"]
                    ),
            }
        )

    dashboard_data = {
        "service":
            "Pearls AQI Predictor",

        "step":
            18,

        "production_model":
            "XGBoost tuned",

        "target":
            TARGET_COLUMN,

        "forecast_horizon":
            FORECAST_HORIZON,

        "forecast_origin":
            api_response[
                "forecast_origin"
            ],

        "forecast_start":
            api_response[
                "forecast_start"
            ],

        "forecast_end":
            api_response[
                "forecast_end"
            ],

        "features_used":
            EXPECTED_FEATURE_COUNT,

        "model_count":
            72,

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

        "future_target_leakage":
            "NONE",

        "summary":
            summary,

        "category_distribution":
            category_distribution,

        "risk_thresholds":
            risk_counts,

        "blocks_24h":
            blocks,

        "forecast":
            json_safe(
                dashboard_df.to_dict(
                    orient="records"
                )
            ),

        "source_api":
            API_PREDICT_URL,

        "api_metadata":
            {
                "deployment_candidate":
                    metadata.get(
                        "deployment_candidate"
                    ),

                "models_loaded":
                    metadata.get(
                        "models_loaded"
                    ),

                "features_used":
                    metadata.get(
                        "features_used"
                    ),

                "feature_order_locked":
                    metadata.get(
                        "feature_order_locked"
                    ),
            },

        "visualizations": {
            "forecast":
                os.path.basename(
                    FORECAST_PNG
                ),

            "categories":
                os.path.basename(
                    CATEGORY_PNG
                ),

            "risk":
                os.path.basename(
                    RISK_PNG
                ),

            "blocks":
                os.path.basename(
                    BLOCK_PNG
                ),
        },
    }

    with open(
        DASHBOARD_JSON,
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            json_safe(
                dashboard_data
            ),
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "Dashboard JSON saved:",
        DASHBOARD_JSON,
    )

    return (
        summary,
        category_distribution,
    )


def save_visualization_report(
    df,
    summary,
    category_distribution,
    risk_counts,
    metadata,
    api_response,
    elapsed,
):
    banner(
        "SAVING STEP 18 VISUALIZATION REPORT"
    )

    report = {
        "project":
            "PEARLS AQI PREDICTOR",

        "step":
            18,

        "step_name":
            "XGBoost Production Forecast Visualization",

        "status":
            "SUCCESS",

        "production_model":
            "XGBoost tuned",

        "target":
            TARGET_COLUMN,

        "forecast_horizon":
            FORECAST_HORIZON,

        "model_count":
            72,

        "features_used":
            EXPECTED_FEATURE_COUNT,

        "forecast_origin":
            api_response[
                "forecast_origin"
            ],

        "forecast_start":
            api_response[
                "forecast_start"
            ],

        "forecast_end":
            api_response[
                "forecast_end"
            ],

        "forecast_rows":
            int(len(df)),

        "summary":
            summary,

        "category_distribution":
            category_distribution,

        "risk_thresholds":
            risk_counts,

        "model_selection_performed":
            False,

        "hyperparameter_tuning_performed":
            False,

        "model_retraining_performed":
            False,

        "validation_used":
            False,

        "test_used":
            False,

        "future_target_leakage":
            "NONE",

        "api_contract": {
            "deployment_candidate":
                metadata.get(
                    "deployment_candidate"
                ),

            "models_loaded":
                metadata.get(
                    "models_loaded"
                ),

            "expected_models":
                metadata.get(
                    "expected_models",
                    72,
                ),

            "features_locked":
                metadata.get(
                    "features_used"
                ),

            "expected_features":
                EXPECTED_FEATURE_COUNT,

            "feature_order_locked":
                metadata.get(
                    "feature_order_locked"
                ),
        },

        "output_files": {
            "dashboard_csv":
                DASHBOARD_CSV,

            "dashboard_json":
                DASHBOARD_JSON,

            "forecast_png":
                FORECAST_PNG,

            "category_png":
                CATEGORY_PNG,

            "risk_png":
                RISK_PNG,

            "block_png":
                BLOCK_PNG,

            "visualization_report":
                VISUALIZATION_REPORT,
        },

        "execution_seconds":
            float(elapsed),
    }

    with open(
        VISUALIZATION_REPORT,
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            json_safe(report),
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "Visualization report saved:",
        VISUALIZATION_REPORT,
    )


# ======================================================================
# MAIN
# ======================================================================

def main():

    start_time = time.time()

    banner(
        "PEARLS AQI PREDICTOR"
    )

    banner(
        "STEP 18 — XGBOOST PRODUCTION FORECAST VISUALIZATION"
    )

    print(
        "Target                  :",
        TARGET_COLUMN,
    )

    print(
        "Forecast horizon        :",
        FORECAST_HORIZON,
    )

    print(
        "Production model        :",
        "XGBoost tuned",
    )

    print(
        "API                     :",
        API_BASE_URL,
    )

    print(
        "Model selection         :",
        "NOT performed",
    )

    print(
        "Validation/test usage   :",
        "NOT used",
    )

    # ------------------------------------------------------------------
    # Create output directories
    # ------------------------------------------------------------------

    os.makedirs(
        REPORTS_DIR,
        exist_ok=True,
    )

    os.makedirs(
        VISUALIZATION_DIR,
        exist_ok=True,
    )

    os.makedirs(
        PREDICTION_DIR,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # Check API health
    # ------------------------------------------------------------------

    banner(
        "CHECKING PRODUCTION API HEALTH"
    )

    health_response = requests.get(
        API_BASE_URL.rstrip("/")
        + "/health",
        timeout=15,
    )

    print(
        "HTTP status         :",
        health_response.status_code,
    )

    health_response.raise_for_status()

    health = health_response.json()

    if health.get(
        "status"
    ) != "ok":
        fail(
            "Production API health status is not OK."
        )

    if health.get(
        "deployment_candidate"
    ) != "XGBoost tuned":
        fail(
            "Production API is not running the expected XGBoost deployment."
        )

    if int(
        health.get(
            "models_loaded",
            0,
        )
    ) != 72:
        fail(
            "Production API does not have exactly 72 models loaded."
        )

    if int(
        health.get(
            "features_locked",
            0,
        )
    ) != 101:
        fail(
            "Production API does not have exactly 101 locked features."
        )

    print(
        "Health status        : PASS"
    )

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    metadata, feature_columns = (
        get_api_metadata()
    )

    # ------------------------------------------------------------------
    # Load latest production row
    # ------------------------------------------------------------------

    (
        production_df,
        latest_row,
        timestamp_column,
        forecast_origin,
    ) = load_latest_features()

    print(
        "Latest timestamp     :",
        forecast_origin,
    )

    # ------------------------------------------------------------------
    # Validate 101-feature protocol
    # ------------------------------------------------------------------

    banner(
        "VALIDATING LOCKED 101-FEATURE PROTOCOL"
    )

    if len(feature_columns) != 101:
        fail(
            "API did not return exactly 101 features."
        )

    missing = [
        feature
        for feature in feature_columns
        if feature not in production_df.columns
    ]

    if missing:
        fail(
            "Production feature container is missing required features: "
            + ", ".join(missing[:20])
        )

    print(
        "Required features    :",
        101,
    )

    print(
        "All required features: FOUND"
    )

    # ------------------------------------------------------------------
    # Build request
    # ------------------------------------------------------------------

    payload = build_api_payload(
        latest_row,
        feature_columns,
        forecast_origin,
    )

    # ------------------------------------------------------------------
    # API forecast
    # ------------------------------------------------------------------

    api_response, api_elapsed = (
        call_prediction_api(
            payload
        )
    )

    # ------------------------------------------------------------------
    # Validate forecast
    # ------------------------------------------------------------------

    forecast_df = validate_forecast(
        api_response,
        forecast_origin,
    )

    # ------------------------------------------------------------------
    # Print forecast summary
    # ------------------------------------------------------------------

    banner(
        "XGBOOST PRODUCTION FORECAST SUMMARY"
    )

    values = forecast_df[
        "predicted_aqi"
    ].to_numpy(
        dtype=float
    )

    print(
        "Minimum predicted AQI :",
        f"{values.min():.3f}",
    )

    print(
        "Maximum predicted AQI :",
        f"{values.max():.3f}",
    )

    print(
        "Mean predicted AQI    :",
        f"{values.mean():.3f}",
    )

    print(
        "Median predicted AQI  :",
        f"{np.median(values):.3f}",
    )

    print(
        "Std predicted AQI     :",
        f"{values.std():.3f}",
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

        row = forecast_df.iloc[
            horizon - 1
        ]

        print(
            f"{horizon:02d}h | "
            f"Predicted AQI = "
            f"{row['predicted_aqi']:8.3f}"
        )

    # ------------------------------------------------------------------
    # Categories
    # ------------------------------------------------------------------

    banner(
        "AQI CATEGORY DISTRIBUTION"
    )

    category_counts = (
        forecast_df[
            "aqi_category"
        ]
        .value_counts()
        .reindex(
            category_order(),
            fill_value=0,
        )
    )

    for category in category_order():

        count = int(
            category_counts[
                category
            ]
        )

        percentage = (
            count
            / FORECAST_HORIZON
            * 100
        )

        print(
            f"{category:<40}"
            f"{count:3d} hours "
            f"({percentage:6.2f}%)"
        )

    # ------------------------------------------------------------------
    # Risk
    # ------------------------------------------------------------------

    risk_counts = create_risk_plot(
        forecast_df
    )

    # ------------------------------------------------------------------
    # Other visualizations
    # ------------------------------------------------------------------

    create_forecast_plot(
        forecast_df
    )

    create_category_plot(
        forecast_df
    )

    block_summary = (
        create_24h_block_plot(
            forecast_df
        )
    )

    # ------------------------------------------------------------------
    # Dashboard outputs
    # ------------------------------------------------------------------

    (
        summary,
        category_distribution,
    ) = save_dashboard_outputs(
        forecast_df,
        risk_counts,
        block_summary,
        api_response,
        metadata,
    )

    # ------------------------------------------------------------------
    # Visualization report
    # ------------------------------------------------------------------

    elapsed = (
        time.time()
        - start_time
    )

    save_visualization_report(
        forecast_df,
        summary,
        category_distribution,
        risk_counts,
        metadata,
        api_response,
        elapsed,
    )

    # ------------------------------------------------------------------
    # Final output
    # ------------------------------------------------------------------

    banner(
        "STEP 18 COMPLETE"
    )

    print(
        "Production model                : XGBoost tuned"
    )

    print(
        "Forecast horizons               : 72"
    )

    print(
        "Forecast start                  :",
        api_response[
            "forecast_start"
        ],
    )

    print(
        "Forecast end                    :",
        api_response[
            "forecast_end"
        ],
    )

    print(
        "Minimum predicted AQI           :",
        f"{values.min():.3f}",
    )

    print(
        "Maximum predicted AQI           :",
        f"{values.max():.3f}",
    )

    print(
        "Mean predicted AQI              :",
        f"{values.mean():.3f}",
    )

    print(
        "Dominant AQI category           :",
        summary[
            "dominant_aqi_category"
        ],
    )

    print(
        "Model selection performed       : FALSE"
    )

    print(
        "Hyperparameter tuning performed : FALSE"
    )

    print(
        "Model retraining performed      : FALSE"
    )

    print(
        "Validation used                 : FALSE"
    )

    print(
        "Test used                       : FALSE"
    )

    print(
        "Future target leakage           : NONE"
    )

    print(
        "Visualization status            : SUCCESS"
    )

    print()

    print(
        "Dashboard-ready CSV:"
    )

    print(
        DASHBOARD_CSV
    )

    print()

    print(
        "Dashboard-ready JSON:"
    )

    print(
        DASHBOARD_JSON
    )

    print()

    print(
        "Forecast visualization:"
    )

    print(
        FORECAST_PNG
    )

    print()

    print(
        "Category visualization:"
    )

    print(
        CATEGORY_PNG
    )

    print()

    print(
        "Risk visualization:"
    )

    print(
        RISK_PNG
    )

    print()

    print(
        "24-hour block visualization:"
    )

    print(
        BLOCK_PNG
    )

    print()

    print(
        "Production visualization report:"
    )

    print(
        VISUALIZATION_REPORT
    )

    print()

    print(
        "Execution time                  :",
        f"{elapsed:.4f}s",
    )


if __name__ == "__main__":

    try:
        main()

    except KeyboardInterrupt:

        print()
        print(
            "STEP 18 interrupted by user."
        )
        sys.exit(130)

    except Exception as exc:

        print()
        print(
            "=" * 70
        )
        print(
            "STEP 18 FAILED"
        )
        print(
            "=" * 70
        )
        print(
            type(exc).__name__
            + ": "
            + str(exc)
        )
        sys.exit(1)