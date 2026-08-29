"""
PEARLS AQI PREDICTOR
STEP 14 — MODEL COMPARISON / SELECTION

Purpose
-------
Compare the available forecasting models and select the final model
using the chronological validation set.

Models compared
---------------
1. Seasonal Persistence Baseline       — Step 10
2. Original XGBoost                   — Step 12
3. Tuned XGBoost                      — Step 13

Methodology
-----------
- Validation MAE is the primary model-selection metric.
- Validation data is used for model selection.
- Test data is NEVER used for model selection.
- Test metrics are reported only after the winning model is selected.
- Forecast horizons remain independent: t+1 ... t+72.
- All models are compared using the same 72 forecast horizons.

Inputs
------
data/processed/splits/train.csv
data/processed/splits/validation.csv
data/processed/splits/test.csv

reports/baseline_results.json
reports/xgboost_results.json
reports/xgboost_tuning_results.json

Outputs
-------
Report:
    reports/model_comparison_results.json

Selection:
    Final selected model based on mean validation MAE.

IMPORTANT
---------
Step 12 and Step 13 use different report schemas.

Step 12:
    validation.by_horizon
    test.by_horizon

Step 13:
    final_results

This script normalizes both formats into one common structure.
"""

import json
import os
import warnings

import numpy as np
import pandas as pd

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

BASELINE_REPORT_FILE = os.path.join(
    BASE_DIR,
    "reports",
    "baseline_results.json",
)

STEP12_REPORT_FILE = os.path.join(
    BASE_DIR,
    "reports",
    "xgboost_results.json",
)

STEP13_REPORT_FILE = os.path.join(
    BASE_DIR,
    "reports",
    "xgboost_tuning_results.json",
)

REPORT_FILE = os.path.join(
    BASE_DIR,
    "reports",
    "model_comparison_results.json",
)

TARGET_COLUMN = "us_aqi"

FORECAST_HORIZON = 72

SEASONAL_LAG = 24

SELECTION_METRIC = "validation_mae"


# ======================================================================
# HEADER
# ======================================================================

print("=" * 70)
print("PEARLS AQI PREDICTOR")
print("STEP 14 — MODEL COMPARISON / SELECTION")
print("=" * 70)

print(
    f"Target              : {TARGET_COLUMN}"
)

print(
    f"Forecast horizon    : {FORECAST_HORIZON} hours"
)

print(
    f"Seasonal lag        : {SEASONAL_LAG} hours"
)

print(
    f"Selection metric    : Validation MAE"
)

print(
    f"Test used for       : Final evaluation ONLY"
)


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
    ("baseline_results.json", BASELINE_REPORT_FILE),
    ("xgboost_results.json", STEP12_REPORT_FILE),
    (
        "xgboost_tuning_results.json",
        STEP13_REPORT_FILE,
    ),
]

for name, path in required_files:

    if os.path.exists(path):

        print(
            f"{name:<35}: FOUND"
        )

    else:

        print(
            f"{name:<35}: NOT FOUND"
        )

        raise FileNotFoundError(
            f"Required file not found: {path}"
        )


# ======================================================================
# LOAD DATA
# ======================================================================

print("\n" + "=" * 70)
print("LOADING DATA")
print("=" * 70)

train_df = pd.read_csv(
    TRAIN_FILE
)

validation_df = pd.read_csv(
    VALIDATION_FILE
)

test_df = pd.read_csv(
    TEST_FILE
)

print(
    f"Train rows          : {len(train_df):,}"
)

print(
    f"Validation rows     : {len(validation_df):,}"
)

print(
    f"Test rows           : {len(test_df):,}"
)


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
        f"{df['time'].min()} -> "
        f"{df['time'].max()}"
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
# LOAD REPORTS
# ======================================================================

print("\n" + "=" * 70)
print("LOADING PREVIOUS RESULTS")
print("=" * 70)


def load_json_report(path, name):

    try:

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as f:

            report = json.load(f)

    except Exception as exc:

        raise ValueError(
            f"Unable to load {name}: {exc}"
        )

    print(
        f"{name:<25}: LOADED"
    )

    return report


baseline_report = load_json_report(
    BASELINE_REPORT_FILE,
    "Baseline report",
)

step12_report = load_json_report(
    STEP12_REPORT_FILE,
    "Step 12 report",
)

step13_report = load_json_report(
    STEP13_REPORT_FILE,
    "Step 13 report",
)


# ======================================================================
# RESULT NORMALIZATION
# ======================================================================

print("\n" + "=" * 70)
print("EXTRACTING MODEL RESULTS")
print("=" * 70)


def normalize_horizon_results(
    validation_results,
    test_results,
    report_name,
):
    """
    Convert validation/test horizon dictionaries into
    a common internal structure.

    Expected output:

    [
        {
            "horizon": 1,
            "validation_mae": ...,
            "validation_rmse": ...,
            "test_mae": ...,
            "test_rmse": ...
        },
        ...
    ]
    """

    if not isinstance(
        validation_results,
        list,
    ):

        raise ValueError(
            f"{report_name}: validation results "
            "must be a list."
        )

    if not isinstance(
        test_results,
        list,
    ):

        raise ValueError(
            f"{report_name}: test results "
            "must be a list."
        )


    validation_by_horizon = {
        int(result["horizon"]): result
        for result in validation_results
    }

    test_by_horizon = {
        int(result["horizon"]): result
        for result in test_results
    }


    normalized = []


    for horizon in range(
        1,
        FORECAST_HORIZON + 1,
    ):

        if horizon not in validation_by_horizon:

            raise ValueError(
                f"{report_name}: validation results "
                f"missing horizon {horizon}."
            )

        if horizon not in test_by_horizon:

            raise ValueError(
                f"{report_name}: test results "
                f"missing horizon {horizon}."
            )


        validation = (
            validation_by_horizon[horizon]
        )

        test = (
            test_by_horizon[horizon]
        )


        # Support either:
        #
        # validation_mae / validation_rmse
        #
        # or:
        #
        # mae / rmse

        validation_mae = validation.get(
            "validation_mae",
            validation.get("mae"),
        )

        validation_rmse = validation.get(
            "validation_rmse",
            validation.get("rmse"),
        )

        test_mae = test.get(
            "test_mae",
            test.get("mae"),
        )

        test_rmse = test.get(
            "test_rmse",
            test.get("rmse"),
        )


        if validation_mae is None:

            raise ValueError(
                f"{report_name}: missing validation MAE "
                f"for horizon {horizon}."
            )

        if validation_rmse is None:

            raise ValueError(
                f"{report_name}: missing validation RMSE "
                f"for horizon {horizon}."
            )

        if test_mae is None:

            raise ValueError(
                f"{report_name}: missing test MAE "
                f"for horizon {horizon}."
            )

        if test_rmse is None:

            raise ValueError(
                f"{report_name}: missing test RMSE "
                f"for horizon {horizon}."
            )


        normalized.append(
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
            }
        )


    return normalized


def extract_final_results(
    report,
    report_name,
):
    """
    Extract model results from either:

    Step 12 format:
        validation.by_horizon
        test.by_horizon

    or Step 13 format:
        final_results
    """

    # --------------------------------------------------------------
    # FORMAT 1 — Step 13
    # --------------------------------------------------------------

    if "final_results" in report:

        final_results = report[
            "final_results"
        ]

        if not isinstance(
            final_results,
            list,
        ):

            raise ValueError(
                f"{report_name}: 'final_results' "
                "must be a list."
            )

        validation_results = []
        test_results = []


        for result in final_results:

            horizon = int(
                result["horizon"]
            )

            validation_results.append(
                {
                    "horizon": horizon,

                    "validation_mae": result[
                        "validation_mae"
                    ],

                    "validation_rmse": result[
                        "validation_rmse"
                    ],
                }
            )

            test_results.append(
                {
                    "horizon": horizon,

                    "test_mae": result[
                        "test_mae"
                    ],

                    "test_rmse": result[
                        "test_rmse"
                    ],
                }
            )


        return normalize_horizon_results(
            validation_results,
            test_results,
            report_name,
        )


    # --------------------------------------------------------------
    # FORMAT 2 — Step 12
    # --------------------------------------------------------------

    validation_section = report.get(
        "validation"
    )

    test_section = report.get(
        "test"
    )


    if (
        isinstance(
            validation_section,
            dict,
        )
        and isinstance(
            test_section,
            dict,
        )
    ):

        validation_results = (
            validation_section.get(
                "by_horizon"
            )
        )

        test_results = (
            test_section.get(
                "by_horizon"
            )
        )


        if (
            validation_results is not None
            and test_results is not None
        ):

            return normalize_horizon_results(
                validation_results,
                test_results,
                report_name,
            )


    raise ValueError(
        f"{report_name} does not contain a supported "
        "model-results format."
    )


# ======================================================================
# EXTRACT XGBOOST RESULTS
# ======================================================================

step12_results = extract_final_results(
    step12_report,
    "Step 12 report",
)

print(
    "Step 12 results      : "
    f"{len(step12_results)} horizons"
)


step13_results = extract_final_results(
    step13_report,
    "Step 13 report",
)

print(
    "Step 13 results      : "
    f"{len(step13_results)} horizons"
)


# ======================================================================
# BASELINE RESULT EXTRACTION
# ======================================================================

print("\n" + "=" * 70)
print("EXTRACTING SEASONAL BASELINE RESULTS")
print("=" * 70)


def extract_baseline_results(
    report,
):
    """
    Extract Seasonal Persistence results from the Step 10
    baseline report.

    The function supports the expected structure:

        results.validation
        results.test

    where each section contains horizon-level results.
    """

    results_section = report.get(
        "results"
    )

    if not isinstance(
        results_section,
        dict,
    ):

        raise ValueError(
            "Baseline report does not contain "
            "a valid 'results' section."
        )


    validation_section = (
        results_section.get(
            "validation",
            []
        )
    )

    test_section = (
        results_section.get(
            "test",
            []
        )
    )


    # --------------------------------------------------------------
    # Find Seasonal Persistence model
    # --------------------------------------------------------------

    seasonal_validation = None
    seasonal_test = None


    if isinstance(
        validation_section,
        list,
    ):

        for result in validation_section:

            model_name = str(
                result.get(
                    "model",
                    ""
                )
            ).strip().lower()

            if model_name == "seasonal persistence":

                seasonal_validation = result

                break


    if isinstance(
        test_section,
        list,
    ):

        for result in test_section:

            model_name = str(
                result.get(
                    "model",
                    ""
                )
            ).strip().lower()

            if model_name == "seasonal persistence":

                seasonal_test = result

                break


    # --------------------------------------------------------------
    # Case A — Overall metrics only
    # --------------------------------------------------------------

    if (
        seasonal_validation is not None
        and seasonal_test is not None
        and isinstance(
            seasonal_validation.get(
                "overall"
            ),
            dict,
        )
        and isinstance(
            seasonal_test.get(
                "overall"
            ),
            dict,
        )
    ):

        validation_overall = (
            seasonal_validation[
                "overall"
            ]
        )

        test_overall = (
            seasonal_test[
                "overall"
            ]
        )


        overall_validation_mae = (
            validation_overall.get(
                "mae"
            )
        )

        overall_validation_rmse = (
            validation_overall.get(
                "rmse"
            )
        )

        overall_test_mae = (
            test_overall.get(
                "mae"
            )
        )

        overall_test_rmse = (
            test_overall.get(
                "rmse"
            )
        )


        if (
            overall_validation_mae is None
            or overall_test_mae is None
        ):

            raise ValueError(
                "Seasonal Persistence baseline report "
                "does not contain required MAE values."
            )


        # If horizon-level values are not available,
        # create an overall representation. This is sufficient
        # for model-level comparison but cannot provide exact
        # per-horizon seasonal metrics.

        return {
            "overall_only": True,

            "mean_validation_mae": float(
                overall_validation_mae
            ),

            "mean_validation_rmse": (
                float(overall_validation_rmse)
                if overall_validation_rmse is not None
                else None
            ),

            "mean_test_mae": float(
                overall_test_mae
            ),

            "mean_test_rmse": (
                float(overall_test_rmse)
                if overall_test_rmse is not None
                else None
            ),

            "horizon_results": [],
        }


    # --------------------------------------------------------------
    # Case B — Horizon-level baseline results
    # --------------------------------------------------------------

    if (
        seasonal_validation is None
        or seasonal_test is None
    ):

        raise ValueError(
            "Could not find Seasonal Persistence "
            "results in baseline report."
        )


    validation_horizons = (
        seasonal_validation.get(
            "by_horizon",
            seasonal_validation.get(
                "horizons",
                []
            ),
        )
    )

    test_horizons = (
        seasonal_test.get(
            "by_horizon",
            seasonal_test.get(
                "horizons",
                []
            ),
        )
    )


    if not validation_horizons:

        raise ValueError(
            "Seasonal Persistence validation "
            "horizon results not found."
        )

    if not test_horizons:

        raise ValueError(
            "Seasonal Persistence test "
            "horizon results not found."
        )


    return {
        "overall_only": False,

        "horizon_results":
            normalize_horizon_results(
                validation_horizons,
                test_horizons,
                "Seasonal Persistence baseline",
            ),
    }


seasonal_baseline = extract_baseline_results(
    baseline_report
)


if seasonal_baseline[
    "overall_only"
]:

    print(
        "Seasonal Persistence : "
        "OVERALL METRICS FOUND"
    )

else:

    print(
        "Seasonal Persistence : "
        f"{len(seasonal_baseline['horizon_results'])} "
        "horizons FOUND"
    )


# ======================================================================
# MODEL SUMMARY CALCULATION
# ======================================================================

print("\n" + "=" * 70)
print("CALCULATING MODEL METRICS")
print("=" * 70)


def summarize_results(
    results,
):
    """
    Calculate mean validation/test MAE and RMSE.
    """

    validation_mae = np.array(
        [
            result["validation_mae"]
            for result in results
        ],
        dtype=float,
    )

    validation_rmse = np.array(
        [
            result["validation_rmse"]
            for result in results
        ],
        dtype=float,
    )

    test_mae = np.array(
        [
            result["test_mae"]
            for result in results
        ],
        dtype=float,
    )

    test_rmse = np.array(
        [
            result["test_rmse"]
            for result in results
        ],
        dtype=float,
    )


    return {
        "mean_validation_mae": float(
            np.mean(
                validation_mae
            )
        ),

        "mean_validation_rmse": float(
            np.mean(
                validation_rmse
            )
        ),

        "mean_test_mae": float(
            np.mean(
                test_mae
            )
        ),

        "mean_test_rmse": float(
            np.mean(
                test_rmse
            )
        ),

        "best_validation_horizon": int(
            results[
                int(
                    np.argmin(
                        validation_mae
                    )
                )
            ]["horizon"]
        ),

        "best_validation_mae": float(
            np.min(
                validation_mae
            )
        ),

        "best_test_horizon": int(
            results[
                int(
                    np.argmin(
                        test_mae
                    )
                )
            ]["horizon"]
        ),

        "best_test_mae": float(
            np.min(
                test_mae
            )
        ),
    }


step12_summary = summarize_results(
    step12_results
)

step13_summary = summarize_results(
    step13_results
)


# ======================================================================
# SEASONAL BASELINE SUMMARY
# ======================================================================

if seasonal_baseline[
    "overall_only"
]:

    seasonal_summary = {

        "mean_validation_mae":
            seasonal_baseline[
                "mean_validation_mae"
            ],

        "mean_validation_rmse":
            seasonal_baseline[
                "mean_validation_rmse"
            ],

        "mean_test_mae":
            seasonal_baseline[
                "mean_test_mae"
            ],

        "mean_test_rmse":
            seasonal_baseline[
                "mean_test_rmse"
            ],

        "best_validation_horizon":
            None,

        "best_validation_mae":
            None,

        "best_test_horizon":
            None,

        "best_test_mae":
            None,
    }

else:

    seasonal_summary = summarize_results(
        seasonal_baseline[
            "horizon_results"
        ]
    )


# ======================================================================
# PRINT MODEL METRICS
# ======================================================================

print(
    f"\n{'Model':<25}"
    f"{'Validation MAE':>18}"
    f"{'Validation RMSE':>20}"
    f"{'Test MAE':>15}"
    f"{'Test RMSE':>17}"
)

print("-" * 95)


print(
    f"{'Seasonal Persistence':<25}"
    f"{seasonal_summary['mean_validation_mae']:>18.3f}"
    f"{seasonal_summary['mean_validation_rmse']:>20.3f}"
    f"{seasonal_summary['mean_test_mae']:>15.3f}"
    f"{seasonal_summary['mean_test_rmse']:>17.3f}"
)


print(
    f"{'Original XGBoost':<25}"
    f"{step12_summary['mean_validation_mae']:>18.3f}"
    f"{step12_summary['mean_validation_rmse']:>20.3f}"
    f"{step12_summary['mean_test_mae']:>15.3f}"
    f"{step12_summary['mean_test_rmse']:>17.3f}"
)


print(
    f"{'Tuned XGBoost':<25}"
    f"{step13_summary['mean_validation_mae']:>18.3f}"
    f"{step13_summary['mean_validation_rmse']:>20.3f}"
    f"{step13_summary['mean_test_mae']:>15.3f}"
    f"{step13_summary['mean_test_rmse']:>17.3f}"
)


# ======================================================================
# MODEL SELECTION
# ======================================================================

print("\n" + "=" * 70)
print("MODEL SELECTION")
print("=" * 70)

# IMPORTANT:
#
# Only validation MAE is used here.
#
# Test metrics are deliberately NOT used to select the model.

selection_candidates = [
    {
        "model": "Seasonal Persistence",
        "validation_mae":
            seasonal_summary[
                "mean_validation_mae"
            ],
    },

    {
        "model": "Original XGBoost",
        "validation_mae":
            step12_summary[
                "mean_validation_mae"
            ],
    },

    {
        "model": "Tuned XGBoost",
        "validation_mae":
            step13_summary[
                "mean_validation_mae"
            ],
    },
]


selection_candidates = sorted(
    selection_candidates,
    key=lambda x: x[
        "validation_mae"
    ],
)


print(
    f"{'Rank':<8}"
    f"{'Model':<25}"
    f"{'Validation MAE':>18}"
)

print("-" * 55)


for rank, candidate in enumerate(
    selection_candidates,
    start=1,
):

    print(
        f"{rank:<8}"
        f"{candidate['model']:<25}"
        f"{candidate['validation_mae']:>18.3f}"
    )


winner = selection_candidates[0]

selected_model = winner[
    "model"
]

selected_validation_mae = winner[
    "validation_mae"
]


# ======================================================================
# WINNER DETAILS
# ======================================================================

print("\n" + "=" * 70)
print("SELECTED FINAL MODEL")
print("=" * 70)

print(
    f"Selected model       : "
    f"{selected_model}"
)

print(
    f"Validation MAE       : "
    f"{selected_validation_mae:.3f}"
)

print(
    "\nSelection basis      : "
    "Lowest mean validation MAE"
)

print(
    "Test set used for    : "
    "Final evaluation only"
)


# ======================================================================
# COMPARE AGAINST SEASONAL BASELINE
# ======================================================================

print("\n" + "=" * 70)
print("SEASONAL BASELINE COMPARISON")
print("=" * 70)


seasonal_validation_mae = (
    seasonal_summary[
        "mean_validation_mae"
    ]
)


selected_vs_seasonal_difference = (
    seasonal_validation_mae
    - selected_validation_mae
)


selected_vs_seasonal_percent = (
    selected_vs_seasonal_difference
    / seasonal_validation_mae
    * 100
    if seasonal_validation_mae != 0
    else 0
)


print(
    f"Seasonal baseline MAE : "
    f"{seasonal_validation_mae:.3f}"
)

print(
    f"Selected model MAE    : "
    f"{selected_validation_mae:.3f}"
)

print(
    f"Difference            : "
    f"{selected_vs_seasonal_difference:.3f}"
)

print(
    f"Improvement           : "
    f"{selected_vs_seasonal_percent:.2f}%"
)


if selected_vs_seasonal_difference > 0:

    print(
        "\nSelected model: "
        "BETTER than Seasonal Persistence"
    )

elif selected_vs_seasonal_difference < 0:

    print(
        "\nSelected model: "
        "WORSE than Seasonal Persistence"
    )

else:

    print(
        "\nSelected model: "
        "EQUAL to Seasonal Persistence"
    )


# ======================================================================
# SELECT WINNING TEST RESULTS
# ======================================================================

print("\n" + "=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)

# Test metrics are accessed only AFTER model selection.

if selected_model == "Seasonal Persistence":

    selected_summary = seasonal_summary

elif selected_model == "Original XGBoost":

    selected_summary = step12_summary

elif selected_model == "Tuned XGBoost":

    selected_summary = step13_summary

else:

    raise ValueError(
        f"Unknown selected model: "
        f"{selected_model}"
    )


print(
    f"Selected model       : "
    f"{selected_model}"
)

print(
    f"Final validation MAE : "
    f"{selected_summary['mean_validation_mae']:.3f}"
)

print(
    f"Final validation RMSE: "
    f"{selected_summary['mean_validation_rmse']:.3f}"
)

print(
    f"Final test MAE       : "
    f"{selected_summary['mean_test_mae']:.3f}"
)

print(
    f"Final test RMSE      : "
    f"{selected_summary['mean_test_rmse']:.3f}"
)


# ======================================================================
# HORIZON COMPARISON
# ======================================================================

print("\n" + "=" * 70)
print("MODEL COMPARISON BY HORIZON")
print("=" * 70)


step12_by_horizon = {
    result["horizon"]: result
    for result in step12_results
}

step13_by_horizon = {
    result["horizon"]: result
    for result in step13_results
}


if seasonal_baseline[
    "overall_only"
]:

    seasonal_by_horizon = {}

else:

    seasonal_by_horizon = {
        result["horizon"]: result
        for result in seasonal_baseline[
            "horizon_results"
        ]
    }


print(
    f"{'Horizon':<10}"
    f"{'Seasonal V':>14}"
    f"{'XGB V':>14}"
    f"{'Tuned V':>14}"
    f"{'Seasonal T':>14}"
    f"{'XGB T':>14}"
    f"{'Tuned T':>14}"
)

print("-" * 90)


display_horizons = [
    1,
    6,
    12,
    24,
    48,
    72,
]


for horizon in display_horizons:

    if horizon in seasonal_by_horizon:

        seasonal_v = (
            seasonal_by_horizon[
                horizon
            ]["validation_mae"]
        )

        seasonal_t = (
            seasonal_by_horizon[
                horizon
            ]["test_mae"]
        )

        seasonal_v_text = (
            f"{seasonal_v:.3f}"
        )

        seasonal_t_text = (
            f"{seasonal_t:.3f}"
        )

    else:

        seasonal_v_text = "N/A"
        seasonal_t_text = "N/A"


    xgb_v = step12_by_horizon[
        horizon
    ]["validation_mae"]

    xgb_t = step12_by_horizon[
        horizon
    ]["test_mae"]


    tuned_v = step13_by_horizon[
        horizon
    ]["validation_mae"]

    tuned_t = step13_by_horizon[
        horizon
    ]["test_mae"]


    print(
        f"{horizon:<10}"
        f"{seasonal_v_text:>14}"
        f"{xgb_v:>14.3f}"
        f"{tuned_v:>14.3f}"
        f"{seasonal_t_text:>14}"
        f"{xgb_t:>14.3f}"
        f"{tuned_t:>14.3f}"
    )


# ======================================================================
# TUNED VS ORIGINAL XGBOOST
# ======================================================================

print("\n" + "=" * 70)
print("TUNED VS ORIGINAL XGBOOST")
print("=" * 70)


xgb_validation_mae = (
    step12_summary[
        "mean_validation_mae"
    ]
)

tuned_validation_mae = (
    step13_summary[
        "mean_validation_mae"
    ]
)


xgb_test_mae = (
    step12_summary[
        "mean_test_mae"
    ]
)

tuned_test_mae = (
    step13_summary[
        "mean_test_mae"
    ]
)


tuning_validation_difference = (
    xgb_validation_mae
    - tuned_validation_mae
)


tuning_validation_percent = (
    tuning_validation_difference
    / xgb_validation_mae
    * 100
    if xgb_validation_mae != 0
    else 0
)


tuning_test_difference = (
    xgb_test_mae
    - tuned_test_mae
)


tuning_test_percent = (
    tuning_test_difference
    / xgb_test_mae
    * 100
    if xgb_test_mae != 0
    else 0
)


print(
    f"Original XGBoost validation MAE : "
    f"{xgb_validation_mae:.3f}"
)

print(
    f"Tuned XGBoost validation MAE    : "
    f"{tuned_validation_mae:.3f}"
)

print(
    f"Validation difference            : "
    f"{tuning_validation_difference:.3f}"
)

print(
    f"Validation improvement           : "
    f"{tuning_validation_percent:.2f}%"
)

print()

print(
    f"Original XGBoost test MAE       : "
    f"{xgb_test_mae:.3f}"
)

print(
    f"Tuned XGBoost test MAE          : "
    f"{tuned_test_mae:.3f}"
)

print(
    f"Test difference                  : "
    f"{tuning_test_difference:.3f}"
)

print(
    f"Test improvement                 : "
    f"{tuning_test_percent:.2f}%"
)


# ======================================================================
# FINAL MODEL DIRECTORY
# ======================================================================

if selected_model == "Tuned XGBoost":

    final_model_directory = os.path.join(
        BASE_DIR,
        "models",
        "artifacts",
        "xgboost_tuned",
    )

elif selected_model == "Original XGBoost":

    final_model_directory = os.path.join(
        BASE_DIR,
        "models",
        "artifacts",
        "xgboost",
    )

else:

    final_model_directory = None


# ======================================================================
# BUILD COMPARISON TABLE
# ======================================================================

comparison_table = []


comparison_table.append(
    {
        "model": "Seasonal Persistence",

        "validation_mae":
            seasonal_summary[
                "mean_validation_mae"
            ],

        "validation_rmse":
            seasonal_summary[
                "mean_validation_rmse"
            ],

        "test_mae":
            seasonal_summary[
                "mean_test_mae"
            ],

        "test_rmse":
            seasonal_summary[
                "mean_test_rmse"
            ],

        "selection_rank":
            next(
                index + 1
                for index, candidate
                in enumerate(
                    selection_candidates
                )
                if candidate[
                    "model"
                ] == "Seasonal Persistence"
            ),
    }
)


comparison_table.append(
    {
        "model": "Original XGBoost",

        "validation_mae":
            step12_summary[
                "mean_validation_mae"
            ],

        "validation_rmse":
            step12_summary[
                "mean_validation_rmse"
            ],

        "test_mae":
            step12_summary[
                "mean_test_mae"
            ],

        "test_rmse":
            step12_summary[
                "mean_test_rmse"
            ],

        "selection_rank":
            next(
                index + 1
                for index, candidate
                in enumerate(
                    selection_candidates
                )
                if candidate[
                    "model"
                ] == "Original XGBoost"
            ),
    }
)


comparison_table.append(
    {
        "model": "Tuned XGBoost",

        "validation_mae":
            step13_summary[
                "mean_validation_mae"
            ],

        "validation_rmse":
            step13_summary[
                "mean_validation_rmse"
            ],

        "test_mae":
            step13_summary[
                "mean_test_mae"
            ],

        "test_rmse":
            step13_summary[
                "mean_test_rmse"
            ],

        "selection_rank":
            next(
                index + 1
                for index, candidate
                in enumerate(
                    selection_candidates
                )
                if candidate[
                    "model"
                ] == "Tuned XGBoost"
            ),
    }
)


# ======================================================================
# REPORT
# ======================================================================

print("\n" + "=" * 70)
print("SAVING MODEL COMPARISON REPORT")
print("=" * 70)


report = {

    "step": 14,

    "description":
        "Model comparison and final model selection",

    "target":
        TARGET_COLUMN,

    "forecast_horizon":
        FORECAST_HORIZON,

    "seasonal_lag":
        SEASONAL_LAG,

    "selection_metric":
        SELECTION_METRIC,

    "test_used_for_selection":
        False,

    "methodology": {

        "training_data":
            "Used by previous model-training steps",

        "validation_data":
            "Used for model selection",

        "test_data":
            "Used only for final evaluation",

        "selection_rule":
            "Lowest mean validation MAE",

        "models_compared": [
            "Seasonal Persistence",
            "Original XGBoost",
            "Tuned XGBoost",
        ],
    },

    "selected_model":
        selected_model,

    "selected_validation_mae":
        selected_validation_mae,

    "selected_model_summary":
        selected_summary,

    "selection_ranking":
        selection_candidates,

    "comparison": {

        "seasonal_persistence":
            seasonal_summary,

        "original_xgboost":
            step12_summary,

        "tuned_xgboost":
            step13_summary,
    },

    "comparison_table":
        comparison_table,

    "horizon_results": {

        "seasonal_persistence":
            (
                seasonal_baseline[
                    "horizon_results"
                ]
                if not seasonal_baseline[
                    "overall_only"
                ]
                else []
            ),

        "original_xgboost":
            step12_results,

        "tuned_xgboost":
            step13_results,
    },

    "selected_model_test_evaluation": {

        "mean_test_mae":
            selected_summary[
                "mean_test_mae"
            ],

        "mean_test_rmse":
            selected_summary[
                "mean_test_rmse"
            ],

        "best_test_horizon":
            selected_summary[
                "best_test_horizon"
            ],

        "best_test_mae":
            selected_summary[
                "best_test_mae"
            ],
    },

    "seasonal_baseline_comparison": {

        "seasonal_validation_mae":
            seasonal_validation_mae,

        "selected_model_validation_mae":
            selected_validation_mae,

        "difference":
            selected_vs_seasonal_difference,

        "improvement_percent":
            selected_vs_seasonal_percent,

        "selected_model_beats_baseline":
            (
                selected_vs_seasonal_difference
                > 0
            ),
    },

    "tuned_vs_original_xgboost": {

        "original_validation_mae":
            xgb_validation_mae,

        "tuned_validation_mae":
            tuned_validation_mae,

        "validation_difference":
            tuning_validation_difference,

        "validation_improvement_percent":
            tuning_validation_percent,

        "original_test_mae":
            xgb_test_mae,

        "tuned_test_mae":
            tuned_test_mae,

        "test_difference":
            tuning_test_difference,

        "test_improvement_percent":
            tuning_test_percent,
    },

    "selected_model_directory":
        final_model_directory,

    "input_reports": {

        "baseline":
            BASELINE_REPORT_FILE,

        "step12":
            STEP12_REPORT_FILE,

        "step13":
            STEP13_REPORT_FILE,
    },
}


# ======================================================================
# SAVE REPORT
# ======================================================================

os.makedirs(
    os.path.dirname(
        REPORT_FILE
    ),
    exist_ok=True,
)


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
    f"Report saved to:"
)

print(
    REPORT_FILE
)


# ======================================================================
# COMPLETE
# ======================================================================

print("\n" + "=" * 70)
print("STEP 14 COMPLETE")
print("=" * 70)

print(
    f"Models compared       : 3"
)

print(
    f"Forecast horizons     : "
    f"{FORECAST_HORIZON}"
)

print(
    f"Selection metric      : "
    f"Mean validation MAE"
)

print(
    f"Selected model        : "
    f"{selected_model}"
)

print(
    f"Validation MAE        : "
    f"{selected_validation_mae:.3f}"
)

print(
    f"Final test MAE        : "
    f"{selected_summary['mean_test_mae']:.3f}"
)

print(
    f"Final test RMSE       : "
    f"{selected_summary['mean_test_rmse']:.3f}"
)

print()

print(
    "Test set was NOT used "
    "for model selection."
)

print()

print(
    "Model comparison report:"
)

print(
    REPORT_FILE
)

if final_model_directory:

    print()

    print(
        "Selected model directory:"
    )

    print(
        final_model_directory
    )

print("\n" + "=" * 70)