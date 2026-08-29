"""
PEARLS AQI PREDICTOR
STEP 15 — XGBOOST PRODUCTION SERVICE

Purpose
-------
Serve the locked XGBoost deployment candidate operationally.

Architecture
------------
Step 12:
    Defines the authoritative 101-feature protocol.

Step 13:
    Selects and trains the deployment candidate.

Step 14:
    Validates production inference and generates a 72-hour forecast.

Step 15:
    Loads the locked artifacts and provides a reusable production
    prediction service.

Important
---------
- NO hyperparameter tuning.
- NO model selection.
- NO model retraining.
- NO test-set evaluation.
- NO feature reconstruction.
- The exact Step 13 101-feature protocol is authoritative.
- The exact 72 trained XGBoost horizon models are loaded.
- One production feature row produces 72 hourly predictions.

Inputs
------
Default production input:
    data/processed/splits/test.csv

Deployment artifacts:
    models/artifacts/xgboost_tuned/

Step 13 report:
    reports/xgboost_tuning_results.json

Step 14 report:
    reports/production_prediction_results.json

Outputs
-------
Production forecast:
    reports/predictions/production_72h_service_predictions.csv

Service report:
    reports/production_service_results.json

Usage
-----
    python scripts/15_xgboost_production_service.py

The service can also be imported and used programmatically:

    from scripts.15_xgboost_production_service import (
        XGBoostProductionService
    )

    service = XGBoostProductionService()
    forecast = service.predict_from_dataframe(df)
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

TARGET_COLUMN = "us_aqi"

FORECAST_HORIZON = 72

EXPECTED_FEATURE_COUNT = 101

RANDOM_STATE = 42


# ----------------------------------------------------------------------
# Step 13 deployment report
# ----------------------------------------------------------------------

STEP13_REPORT_FILE = os.path.join(
    BASE_DIR,
    "reports",
    "xgboost_tuning_results.json",
)


# ----------------------------------------------------------------------
# Step 14 production inference report
# ----------------------------------------------------------------------

STEP14_REPORT_FILE = os.path.join(
    BASE_DIR,
    "reports",
    "production_prediction_results.json",
)


# ----------------------------------------------------------------------
# Production input
# ----------------------------------------------------------------------

PRODUCTION_INPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "splits",
    "test.csv",
)


# ----------------------------------------------------------------------
# Model artifacts
# ----------------------------------------------------------------------

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "artifacts",
    "xgboost_tuned",
)


# ----------------------------------------------------------------------
# Production output
# ----------------------------------------------------------------------

PREDICTION_DIR = os.path.join(
    BASE_DIR,
    "reports",
    "predictions",
)

PREDICTION_FILE = os.path.join(
    PREDICTION_DIR,
    "production_72h_service_predictions.csv",
)


SERVICE_REPORT_FILE = os.path.join(
    BASE_DIR,
    "reports",
    "production_service_results.json",
)


# ======================================================================
# HEADER
# ======================================================================

print("=" * 70)
print("PEARLS AQI PREDICTOR")
print("STEP 15 — XGBOOST PRODUCTION SERVICE")
print("=" * 70)

print(
    f"Base directory      : {BASE_DIR}"
)

print(
    f"Forecast horizon    : {FORECAST_HORIZON} hours"
)

print(
    f"Target              : {TARGET_COLUMN}"
)

print(
    f"Expected features   : {EXPECTED_FEATURE_COUNT}"
)

print(
    f"Model directory     : {MODEL_DIR}"
)


# ======================================================================
# UTILITY FUNCTIONS
# ======================================================================

def fail(message):
    """
    Raise a production-service error with a clear message.
    """

    raise RuntimeError(
        "\n" + "=" * 70
        + "\nPRODUCTION SERVICE ERROR\n"
        + "=" * 70
        + "\n"
        + str(message)
    )


def load_json(path):
    """
    Load a JSON file.
    """

    if not os.path.exists(path):

        fail(
            f"Required JSON file does not exist:\n{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


def model_path_for_horizon(horizon):
    """
    Return the exact Step 13 model artifact path.
    """

    return os.path.join(
        MODEL_DIR,
        f"xgboost_tuned_{horizon:02d}h.json",
    )


# ======================================================================
# PRODUCTION SERVICE
# ======================================================================

class XGBoostProductionService:
    """
    Production inference service for the locked XGBoost deployment.

    The service:

        1. Loads the Step 13 deployment report.
        2. Verifies the selected configuration.
        3. Loads the exact 101-feature protocol.
        4. Loads all 72 XGBoost model artifacts.
        5. Validates production input.
        6. Generates a 72-hour forecast.
        7. Validates the forecast.
        8. Returns a machine-readable DataFrame.

    No training occurs anywhere in this class.
    """

    def __init__(self):

        self.feature_columns = []

        self.models = {}

        self.parameters = None

        self.selected_configuration = None

        self.loaded = False

        self._load_deployment_contract()

        self._load_models()

        self.loaded = True


    # ==================================================================
    # DEPLOYMENT CONTRACT
    # ==================================================================

    def _load_deployment_contract(self):

        print("\n" + "=" * 70)
        print("LOADING DEPLOYMENT CONTRACT")
        print("=" * 70)

        step13_report = load_json(
            STEP13_REPORT_FILE
        )

        # --------------------------------------------------------------
        # Verify Step 13
        # --------------------------------------------------------------

        step_number = step13_report.get(
            "step"
        )

        if step_number != 13:

            fail(
                "Deployment report is not a Step 13 report."
            )

        # --------------------------------------------------------------
        # Target
        # --------------------------------------------------------------

        report_target = step13_report.get(
            "target"
        )

        if report_target != TARGET_COLUMN:

            fail(
                "Target mismatch.\n"
                f"Expected: {TARGET_COLUMN}\n"
                f"Found   : {report_target}"
            )

        # --------------------------------------------------------------
        # Forecast horizon
        # --------------------------------------------------------------

        report_horizon = step13_report.get(
            "forecast_horizon"
        )

        if report_horizon != FORECAST_HORIZON:

            fail(
                "Forecast horizon mismatch.\n"
                f"Expected: {FORECAST_HORIZON}\n"
                f"Found   : {report_horizon}"
            )

        # --------------------------------------------------------------
        # Selected configuration
        # --------------------------------------------------------------

        self.selected_configuration = (
            step13_report.get(
                "selected_configuration"
            )
        )

        print(
            f"Selected configuration : "
            f"{self.selected_configuration}"
        )

        # Step 15 must only serve the tuned deployment
        # candidate. It must never silently fall back.
        if self.selected_configuration != "tuned":

            fail(
                "Step 13 did not select the tuned configuration.\n"
                f"Selected: {self.selected_configuration}\n"
                "Production service will not override model selection."
            )

        # --------------------------------------------------------------
        # Parameters
        # --------------------------------------------------------------

        self.parameters = (
            step13_report.get(
                "best_parameters"
            )
        )

        if not self.parameters:

            self.parameters = (
                step13_report.get(
                    "best_tuning_parameters"
                )
            )

        if not self.parameters:

            fail(
                "Step 13 report does not contain "
                "best XGBoost parameters."
            )

        print(
            "Deployment parameters:"
        )

        print(
            json.dumps(
                self.parameters,
                indent=2,
            )
        )

        # --------------------------------------------------------------
        # Feature protocol
        # --------------------------------------------------------------

        self.feature_columns = (
            step13_report.get(
                "feature_columns"
            )
        )

        if not self.feature_columns:

            fail(
                "Step 13 report does not contain "
                "feature_columns."
            )

        print(
            f"\nStep 13 feature count : "
            f"{len(self.feature_columns)}"
        )

        print(
            f"Expected feature count: "
            f"{EXPECTED_FEATURE_COUNT}"
        )

        if len(self.feature_columns) != EXPECTED_FEATURE_COUNT:

            fail(
                "Feature protocol mismatch.\n"
                f"Expected: {EXPECTED_FEATURE_COUNT}\n"
                f"Found   : {len(self.feature_columns)}"
            )

        # --------------------------------------------------------------
        # Feature uniqueness
        # --------------------------------------------------------------

        if len(
            set(self.feature_columns)
        ) != len(
            self.feature_columns
        ):

            fail(
                "Duplicate feature names detected "
                "in the Step 13 feature protocol."
            )

        print(
            "Feature protocol     : PASS"
        )

        print(
            "Exact 101-feature set: LOCKED"
        )

        # --------------------------------------------------------------
        # Confirm Step 13 did not use test set for selection
        # --------------------------------------------------------------

        test_used_for_tuning = (
            step13_report.get(
                "test_used_for_tuning"
            )
        )

        if test_used_for_tuning is not False:

            fail(
                "Step 13 deployment contract does not "
                "confirm that test data was excluded "
                "from hyperparameter selection."
            )

        print(
            "Test-set selection check: PASS"
        )

        # --------------------------------------------------------------
        # Step 13 model directory
        # --------------------------------------------------------------

        report_model_dir = (
            step13_report.get(
                "model_directory"
            )
        )

        if report_model_dir:

            print(
                f"Step 13 model directory: "
                f"{report_model_dir}"
            )


    # ==================================================================
    # LOAD MODELS
    # ==================================================================

    def _load_models(self):

        print("\n" + "=" * 70)
        print("LOADING LOCKED XGBOOST MODELS")
        print("=" * 70)

        if not os.path.isdir(
            MODEL_DIR
        ):

            fail(
                f"Model directory does not exist:\n"
                f"{MODEL_DIR}"
            )

        for horizon in range(
            1,
            FORECAST_HORIZON + 1,
        ):

            path = model_path_for_horizon(
                horizon
            )

            if not os.path.exists(path):

                fail(
                    f"Missing model artifact for "
                    f"{horizon:02d}h:\n{path}"
                )

            model = XGBRegressor()

            model.load_model(
                path
            )

            self.models[
                horizon
            ] = model

            print(
                f"{horizon:02d}h model       : LOADED"
            )

        print(
            f"\nModels loaded        : "
            f"{len(self.models)}"
        )

        if len(self.models) != FORECAST_HORIZON:

            fail(
                "Not all required horizon models "
                "were loaded."
            )

        print(
            "Model artifact check : PASS"
        )


    # ==================================================================
    # INPUT VALIDATION
    # ==================================================================

    def validate_input_dataframe(
        self,
        df,
    ):
        """
        Validate a production feature container.

        The DataFrame must contain:

            time
            all 101 locked features

        Additional columns are allowed because the feature protocol
        explicitly selects only the locked feature set.
        """

        if not isinstance(
            df,
            pd.DataFrame,
        ):

            fail(
                "Production input must be a pandas DataFrame."
            )

        if df.empty:

            fail(
                "Production input DataFrame is empty."
            )

        print(
            f"Input rows          : {len(df):,}"
        )

        print(
            f"Input columns       : {len(df.columns)}"
        )

        # --------------------------------------------------------------
        # Timestamp
        # --------------------------------------------------------------

        if "time" not in df.columns:

            fail(
                "Required 'time' column is missing."
            )

        # Work on a copy so the caller's DataFrame
        # is never unexpectedly modified.
        df = df.copy()

        df["time"] = pd.to_datetime(
            df["time"],
            errors="raise",
        )

        if df["time"].isna().any():

            fail(
                "Production input contains invalid timestamps."
            )

        if not df["time"].is_monotonic_increasing:

            fail(
                "Production input timestamps are not "
                "chronologically ordered."
            )

        if not df["time"].is_unique:

            fail(
                "Production input contains duplicate timestamps."
            )

        # --------------------------------------------------------------
        # Feature count
        # --------------------------------------------------------------

        missing_features = [
            feature
            for feature in self.feature_columns
            if feature not in df.columns
        ]

        if missing_features:

            fail(
                "Production input is missing required "
                "Step 13 features.\n"
                f"Missing count: {len(missing_features)}\n"
                f"Missing features: {missing_features}"
            )

        print(
            "All required features FOUND"
        )

        # --------------------------------------------------------------
        # Numeric feature validation
        # --------------------------------------------------------------

        feature_frame = df[
            self.feature_columns
        ]

        for feature in self.feature_columns:

            if not pd.api.types.is_numeric_dtype(
                feature_frame[feature]
            ):

                fail(
                    f"Feature '{feature}' is not numeric."
                )

        if feature_frame.isna().any().any():

            missing = feature_frame.columns[
                feature_frame.isna().any()
            ].tolist()

            fail(
                "Production feature matrix contains "
                f"missing values.\n{missing}"
            )

        values = feature_frame.to_numpy(
            dtype=np.float32
        )

        if not np.isfinite(values).all():

            fail(
                "Production feature matrix contains "
                "infinite or non-finite values."
            )

        print(
            "Feature numeric check : PASS"
        )

        print(
            "Missing values        : NONE"
        )

        print(
            "Infinite values       : NONE"
        )

        return df


    # ==================================================================
    # FEATURE MATRIX
    # ==================================================================

    def create_feature_matrix(
        self,
        df,
    ):
        """
        Create the exact Step 13 feature matrix.
        """

        X = df[
            self.feature_columns
        ].astype(
            np.float32
        )

        if X.shape[1] != EXPECTED_FEATURE_COUNT:

            fail(
                "Production feature matrix does not contain "
                f"{EXPECTED_FEATURE_COUNT} features."
            )

        print(
            f"X_production shape : {X.shape}"
        )

        return X


    # ==================================================================
    # SINGLE FORECAST
    # ==================================================================

    def predict_from_dataframe(
        self,
        df,
    ):
        """
        Generate a 72-hour forecast from a production
        feature container.

        The latest timestamp is used as the forecast origin.

        Returns
        -------
        pandas.DataFrame
            Columns:

                forecast_origin
                forecast_timestamp
                horizon
                predicted_us_aqi
        """

        if not self.loaded:

            fail(
                "Production service has not been loaded."
            )

        print("\n" + "=" * 70)
        print("VALIDATING PRODUCTION INPUT")
        print("=" * 70)

        df = self.validate_input_dataframe(
            df
        )

        # --------------------------------------------------------------
        # Latest production row
        # --------------------------------------------------------------

        latest_row = df.iloc[
            [-1]
        ].copy()

        forecast_origin = (
            latest_row[
                "time"
            ].iloc[0]
        )

        print(
            f"\nForecast origin : "
            f"{forecast_origin}"
        )

        # --------------------------------------------------------------
        # Feature matrix
        # --------------------------------------------------------------

        print("\n" + "=" * 70)
        print("CREATING PRODUCTION FEATURE MATRIX")
        print("=" * 70)

        X_production = (
            self.create_feature_matrix(
                latest_row
            )
        )

        # --------------------------------------------------------------
        # Inference
        # --------------------------------------------------------------

        print("\n" + "=" * 70)
        print("RUNNING PRODUCTION INFERENCE")
        print("=" * 70)

        predictions = []

        inference_start = time.time()

        for horizon in range(
            1,
            FORECAST_HORIZON + 1,
        ):

            model = self.models[
                horizon
            ]

            start = time.time()

            prediction = model.predict(
                X_production
            )

            elapsed = (
                time.time()
                - start
            )

            if len(prediction) != 1:

                fail(
                    f"{horizon:02d}h model returned "
                    f"{len(prediction)} predictions; "
                    "expected exactly 1."
                )

            predicted_aqi = float(
                prediction[0]
            )

            if not np.isfinite(
                predicted_aqi
            ):

                fail(
                    f"{horizon:02d}h model returned "
                    "a non-finite AQI prediction."
                )

            if predicted_aqi < 0:

                fail(
                    f"{horizon:02d}h model returned "
                    f"negative AQI: {predicted_aqi}"
                )

            forecast_timestamp = (
                forecast_origin
                + pd.Timedelta(
                    hours=horizon
                )
            )

            predictions.append(
                {
                    "forecast_origin":
                        forecast_origin,

                    "forecast_timestamp":
                        forecast_timestamp,

                    "horizon":
                        horizon,

                    "predicted_us_aqi":
                        predicted_aqi,
                }
            )

            print(
                f"{horizon:02d}h | "
                f"Predicted AQI = "
                f"{predicted_aqi:8.3f} | "
                f"Time = {elapsed:6.2f}s"
            )

        total_inference_time = (
            time.time()
            - inference_start
        )

        forecast_df = pd.DataFrame(
            predictions
        )

        forecast_df[
            "forecast_origin"
        ] = pd.to_datetime(
            forecast_df[
                "forecast_origin"
            ]
        )

        forecast_df[
            "forecast_timestamp"
        ] = pd.to_datetime(
            forecast_df[
                "forecast_timestamp"
            ]
        )

        return forecast_df, total_inference_time


    # ==================================================================
    # FORECAST VALIDATION
    # ==================================================================

    def validate_forecast(
        self,
        forecast_df,
    ):
        """
        Validate the generated production forecast.
        """

        print("\n" + "=" * 70)
        print("VALIDATING PRODUCTION FORECAST")
        print("=" * 70)

        expected_rows = FORECAST_HORIZON

        actual_rows = len(
            forecast_df
        )

        print(
            f"Forecast rows          : "
            f"{actual_rows}"
        )

        print(
            f"Expected rows          : "
            f"{expected_rows}"
        )

        if actual_rows != expected_rows:

            fail(
                "Production forecast row count mismatch."
            )

        # --------------------------------------------------------------
        # Required columns
        # --------------------------------------------------------------

        required_columns = [
            "forecast_origin",
            "forecast_timestamp",
            "horizon",
            "predicted_us_aqi",
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in forecast_df.columns
        ]

        if missing_columns:

            fail(
                "Production forecast missing columns:\n"
                f"{missing_columns}"
            )

        # --------------------------------------------------------------
        # Horizon sequence
        # --------------------------------------------------------------

        expected_horizons = list(
            range(
                1,
                FORECAST_HORIZON + 1,
            )
        )

        actual_horizons = (
            forecast_df[
                "horizon"
            ].tolist()
        )

        print(
            f"Horizon range          : "
            f"{min(actual_horizons)} -> "
            f"{max(actual_horizons)}"
        )

        if actual_horizons != expected_horizons:

            fail(
                "Horizon sequence is invalid."
            )

        print(
            "Horizon sequence       : PASS"
        )

        # --------------------------------------------------------------
        # Duplicate timestamps
        # --------------------------------------------------------------

        duplicate_count = (
            forecast_df[
                "forecast_timestamp"
            ].duplicated().sum()
        )

        print(
            f"Duplicate timestamps   : "
            f"{duplicate_count}"
        )

        if duplicate_count != 0:

            fail(
                "Duplicate forecast timestamps detected."
            )

        # --------------------------------------------------------------
        # Timestamp continuity
        # --------------------------------------------------------------

        timestamp_differences = (
            forecast_df[
                "forecast_timestamp"
            ]
            .sort_values()
            .diff()
            .dropna()
        )

        expected_delta = pd.Timedelta(
            hours=1
        )

        continuity_pass = all(
            delta == expected_delta
            for delta
            in timestamp_differences
        )

        print(
            "Timestamp continuity   : "
            + (
                "PASS"
                if continuity_pass
                else "FAIL"
            )
        )

        if not continuity_pass:

            fail(
                "Forecast timestamps are not "
                "continuous hourly timestamps."
            )

        # --------------------------------------------------------------
        # Prediction finite validation
        # --------------------------------------------------------------

        prediction_values = (
            forecast_df[
                "predicted_us_aqi"
            ].to_numpy(
                dtype=float
            )
        )

        finite = np.isfinite(
            prediction_values
        ).all()

        print(
            f"Prediction values finite: "
            f"{finite}"
        )

        if not finite:

            fail(
                "Production forecast contains "
                "non-finite values."
            )

        # --------------------------------------------------------------
        # Negative AQI
        # --------------------------------------------------------------

        negative_count = int(
            (
                prediction_values < 0
            ).sum()
        )

        print(
            f"Invalid negative AQI   : "
            f"{negative_count}"
        )

        if negative_count != 0:

            fail(
                "Negative AQI predictions detected."
            )

        print(
            "Production forecast validation: PASS"
        )


    # ==================================================================
    # SAVE FORECAST
    # ==================================================================

    def save_forecast(
        self,
        forecast_df,
    ):
        """
        Save the operational production forecast.
        """

        print("\n" + "=" * 70)
        print("SAVING PRODUCTION FORECAST")
        print("=" * 70)

        os.makedirs(
            PREDICTION_DIR,
            exist_ok=True,
        )

        output_df = forecast_df.copy()

        output_df[
            "forecast_origin"
        ] = output_df[
            "forecast_origin"
        ].dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        output_df[
            "forecast_timestamp"
        ] = output_df[
            "forecast_timestamp"
        ].dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        output_df[
            "predicted_us_aqi"
        ] = output_df[
            "predicted_us_aqi"
        ].round(
            3
        )

        output_df.to_csv(
            PREDICTION_FILE,
            index=False,
        )

        print(
            "Prediction file saved:"
        )

        print(
            PREDICTION_FILE
        )


    # ==================================================================
    # SAVE SERVICE REPORT
    # ==================================================================

    def save_service_report(
        self,
        forecast_df,
        inference_time,
    ):
        """
        Save a machine-readable operational service report.
        """

        print("\n" + "=" * 70)
        print("SAVING PRODUCTION SERVICE REPORT")
        print("=" * 70)

        os.makedirs(
            os.path.dirname(
                SERVICE_REPORT_FILE
            ),
            exist_ok=True,
        )

        predictions = (
            forecast_df[
                "predicted_us_aqi"
            ].to_numpy(
                dtype=float
            )
        )

        report = {

            "step": 15,

            "description":
                "XGBoost production serving",

            "service_status":
                "READY",

            "deployment_candidate":
                "XGBoost tuned",

            "target":
                TARGET_COLUMN,

            "forecast_horizon":
                FORECAST_HORIZON,

            "features_used":
                len(
                    self.feature_columns
                ),

            "feature_protocol":
                "Exact Step 13 101-feature list",

            "feature_columns":
                self.feature_columns,

            "selected_configuration":
                self.selected_configuration,

            "deployment_parameters":
                self.parameters,

            "models_loaded":
                len(
                    self.models
                ),

            "model_directory":
                MODEL_DIR,

            "forecast_origin":
                str(
                    forecast_df[
                        "forecast_origin"
                    ].iloc[0]
                ),

            "forecast_start":
                str(
                    forecast_df[
                        "forecast_timestamp"
                    ].min()
                ),

            "forecast_end":
                str(
                    forecast_df[
                        "forecast_timestamp"
                    ].max()
                ),

            "minimum_predicted_aqi":
                float(
                    predictions.min()
                ),

            "maximum_predicted_aqi":
                float(
                    predictions.max()
                ),

            "mean_predicted_aqi":
                float(
                    predictions.mean()
                ),

            "inference_time_seconds":
                round(
                    inference_time,
                    3,
                ),

            "prediction_file":
                PREDICTION_FILE,

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

            "production_validation":
                "PASS",
        }

        with open(
            SERVICE_REPORT_FILE,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                report,
                f,
                indent=2,
            )

        print(
            "Service report saved:"
        )

        print(
            SERVICE_REPORT_FILE
        )

        return report


# ======================================================================
# MAIN PRODUCTION EXECUTION
# ======================================================================

def main():

    overall_start = time.time()

    # ------------------------------------------------------------------
    # Initialize service
    # ------------------------------------------------------------------

    service = (
        XGBoostProductionService()
    )

    # ------------------------------------------------------------------
    # Load production input
    # ------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("LOADING PRODUCTION INPUT")
    print("=" * 70)

    if not os.path.exists(
        PRODUCTION_INPUT_FILE
    ):

        fail(
            "Production input file does not exist:\n"
            f"{PRODUCTION_INPUT_FILE}"
        )

    print(
        f"Input source : "
        "Latest available processed feature container"
    )

    print(
        f"Input file   : "
        f"{PRODUCTION_INPUT_FILE}"
    )

    production_df = pd.read_csv(
        PRODUCTION_INPUT_FILE
    )

    # ------------------------------------------------------------------
    # Generate forecast
    # ------------------------------------------------------------------

    forecast_df, inference_time = (
        service.predict_from_dataframe(
            production_df
        )
    )

    # ------------------------------------------------------------------
    # Validate forecast
    # ------------------------------------------------------------------

    service.validate_forecast(
        forecast_df
    )

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------

    service.save_forecast(
        forecast_df
    )

    report = (
        service.save_service_report(
            forecast_df,
            inference_time,
        )
    )

    total_time = (
        time.time()
        - overall_start
    )

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("STEP 15 — XGBOOST PRODUCTION SERVICE COMPLETE")
    print("=" * 70)

    print(
        f"Service status       : "
        f"{report['service_status']}"
    )

    print(
        f"Deployment candidate : "
        f"{report['deployment_candidate']}"
    )

    print(
        f"Models loaded        : "
        f"{report['models_loaded']}"
    )

    print(
        f"Features used        : "
        f"{report['features_used']}"
    )

    print(
        f"Forecast horizon     : "
        f"{report['forecast_horizon']} hours"
    )

    print(
        f"Forecast origin      : "
        f"{report['forecast_origin']}"
    )

    print(
        f"Forecast start       : "
        f"{report['forecast_start']}"
    )

    print(
        f"Forecast end         : "
        f"{report['forecast_end']}"
    )

    print(
        f"Minimum predicted AQI: "
        f"{report['minimum_predicted_aqi']:.3f}"
    )

    print(
        f"Maximum predicted AQI: "
        f"{report['maximum_predicted_aqi']:.3f}"
    )

    print(
        f"Mean predicted AQI   : "
        f"{report['mean_predicted_aqi']:.3f}"
    )

    print(
        f"Inference time       : "
        f"{report['inference_time_seconds']:.3f}s"
    )

    print(
        f"Total service time   : "
        f"{total_time:.3f}s"
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
        "Test-set usage       : NOT PERFORMED"
    )

    print()

    print(
        "Production service   : READY"
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
        "Service report:"
    )

    print(
        SERVICE_REPORT_FILE
    )

    print("=" * 70)


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":

    main()