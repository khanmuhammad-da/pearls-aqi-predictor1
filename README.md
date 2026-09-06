# AQI Multi-Horizon Forecasting

A leakage-aware machine learning project for forecasting Air Quality Index (AQI) at multiple future horizons:

- **1 hour ahead**
- **6 hours ahead**
- **24 hours ahead**
- **72 hours ahead**

The project is designed as an end-to-end forecasting system rather than a single machine-learning experiment. It covers data ingestion, data validation, exploratory analysis, feature engineering, forecasting baselines, machine-learning models, multi-horizon evaluation, error analysis, explainability, and eventual deployment.

> **Project status:** Research / development  
> **Primary objective:** Build a reliable and generalising multi-horizon AQI forecasting system  
> **Important:** Long-horizon forecasting (24h and 72h) is treated as a separate modelling problem and is not assumed to be solvable simply by extending a 1-hour model.

---

## 1. Project Overview

Air quality changes over time because of interactions between pollutant concentrations, weather conditions, atmospheric processes, time-of-day effects, and longer-term temporal patterns.

A model that performs well one hour ahead may perform poorly 24 or 72 hours ahead. Therefore, this project explicitly investigates the predictability of AQI at different forecast horizons.

The system will answer questions such as:

1. How predictable is AQI one hour into the future?
2. How does forecast accuracy deteriorate at 6h, 24h, and 72h?
3. Which historical AQI, pollutant, weather, and temporal features provide useful predictive information?
4. Does a machine-learning model actually outperform simple persistence/statistical baselines?
5. Which forecasting strategy generalises best across horizons?
6. Are long-horizon failures caused by insufficient information, poor feature representation, model limitations, or future-covariate availability?
7. Can the final model be deployed as a reproducible forecasting application?

---

## 2. Forecasting Contract

For a forecast origin time `t`, the system predicts AQI at:

```text
t + 1 hour
t + 6 hours
t + 24 hours
t + 72 hours
```

Formally:

```text
ŷ(t+h) = f_h(X_t)
```

where:

- `t` = forecast origin
- `h` = forecast horizon
- `X_t` = information available at time `t`
- `ŷ(t+h)` = predicted AQI at the future target timestamp

The four horizons are treated as explicit forecasting tasks.

### Forecasting principle

At forecast origin `t`, the model must not use information that would only become available after `t`.

This is the central leakage-control principle of the project.

---

## 3. Main Objectives

### Primary objectives

- Build a clean and reproducible AQI forecasting pipeline.
- Forecast AQI at 1h, 6h, 24h, and 72h horizons.
- Establish strong non-ML baselines before complex modelling.
- Compare several forecasting strategies.
- Prevent temporal leakage.
- Evaluate generalisation using chronological validation and testing.
- Analyse model failures rather than relying only on aggregate metrics.
- Determine whether long-horizon deterministic forecasting is practically predictable.
- Prepare the system for deployment.

### Secondary objectives

- Identify important predictors.
- Investigate forecast uncertainty.
- Provide explainable predictions.
- Build a reusable data/feature pipeline.
- Optionally integrate a feature store and model registry.
- Provide an interactive forecasting interface.

---

## 4. Core Design Principles

This project follows several rules.

### Rule 1 — No random train/test split

Time-series data must be split chronologically.

Do not use:

```python
train_test_split(...)
```

with random shuffling for the final forecasting evaluation.

### Rule 2 — No future information

A feature at time `t` may only use information available at or before `t`.

For example:

```text
AQI(t-1)       valid
AQI(t-6)       valid
AQI(t-24)      valid
rolling_mean   valid if calculated only from past/current values

AQI(t+1)       invalid
AQI(t+6)       invalid
future target  invalid
```

### Rule 3 — Test data is not for tuning

The test set should be treated as unseen future data.

Do not repeatedly tune the model against test performance.

### Rule 4 — Baselines come first

Before claiming that a machine-learning model is useful, compare it against simple forecasting strategies such as persistence.

### Rule 5 — Do not optimise for R² alone

A high R² does not automatically mean a forecast is operationally useful.

The project evaluates:

- MAE
- RMSE
- R²
- Bias
- Improvement over baseline
- Error distribution
- Performance during rapid AQI changes
- Performance across AQI regimes

### Rule 6 — Do not assume XGBoost is the final model

Tree-based models are part of the model ladder, not the predetermined answer.

### Rule 7 — Treat 24h and 72h as distinct problems

Poor long-horizon performance may indicate a genuine information/predictability limitation rather than simply an inadequate algorithm.

---

## 5. Expected Project Architecture

The intended high-level architecture is:

```text
                ┌──────────────────────┐
                │   Data Sources / API  │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Data Ingestion        │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Data Validation       │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ EDA / Data Profiling  │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Feature Engineering   │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Forecast Targets      │
                │ 1h / 6h / 24h / 72h  │
                └──────────┬───────────┘
                           │
                           ▼
          ┌──────────────────────────────────┐
          │ Chronological Validation / Split │
          └────────────────┬─────────────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Forecasting Models    │
                │                      │
                │ Persistence          │
                │ Statistical           │
                │ Linear                │
                │ Tree-based            │
                │ Deep/Temporal*        │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Evaluation & Error     │
                │ Analysis               │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Explainability        │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Forecast Service/API  │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Streamlit Dashboard   │
                └──────────────────────┘
```

`*` Advanced/deep temporal models will only be introduced if simpler models and predictability analysis justify them.

---

## 6. Recommended Repository Structure

The repository should evolve toward a structure similar to:

```text
aqi-multi-horizon-forecasting/
│
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_analysis.ipynb
│   └── 04_model_analysis.ipynb
│
├── src/
│   ├── data/
│   │   ├── api_client.py
│   │   ├── ingestion.py
│   │   └── validation.py
│   │
│   ├── features/
│   │   ├── build_features.py
│   │   └── create_targets.py
│   │
│   ├── models/
│   │   ├── baselines.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── error_analysis.py
│   │
│   └── utils/
│       └── config.py
│
├── models/
│   ├── artifacts/
│   └── metrics/
│
├── reports/
│   ├── figures/
│   └── project_report/
│
├── app/
│   └── streamlit_app.py
│
└── tests/
    ├── test_data.py
    ├── test_features.py
    └── test_targets.py
```

The exact structure may change during development.

---

## 7. Data Strategy

The project requires time-indexed AQI and environmental variables.

Potential data categories include:

### Air-quality variables

- AQI
- PM2.5
- PM10
- O3
- NO2
- SO2
- CO

### Weather variables

Depending on data availability:

- Temperature
- Relative humidity
- Precipitation
- Wind speed
- Wind direction
- Surface pressure
- Cloud cover
- Radiation-related variables

### Temporal variables

Examples:

- Hour
- Day of week
- Day of year
- Month
- Weekend/weekday
- Cyclic hour encoding
- Cyclic day-of-year encoding

Future weather forecasts may become important for 24h and 72h forecasting.

---

## 8. Historical vs Future Covariates

One of the most important design decisions is whether future environmental information is actually available at forecast time.

### Historical/current observations

At time `t`, the system may use observations such as:

```text
AQI(t)
PM2.5(t)
PM10(t)
temperature(t)
wind_speed(t)
...
```

and historical lags:

```text
AQI(t-1)
AQI(t-6)
AQI(t-24)
PM2.5(t-1)
...
```

### Future forecast covariates

For a 24h or 72h forecast, future weather forecasts may be available:

```text
temperature forecast(t+24)
wind forecast(t+24)
precipitation forecast(t+24)
...
```

These are fundamentally different from observed future values.

The model must never use the actual future weather observation simply because it exists in the historical dataset.

---

## 9. Target Engineering

For each forecast origin:

```text
target_aqi_1h  = AQI(t + 1h)
target_aqi_6h  = AQI(t + 6h)
target_aqi_24h = AQI(t + 24h)
target_aqi_72h = AQI(t + 72h)
```

Target timestamps should also be retained for auditing.

Example:

```text
origin_timestamp = 2026-01-01 00:00

target_timestamp_1h  = 2026-01-01 01:00
target_timestamp_6h  = 2026-01-01 06:00
target_timestamp_24h = 2026-01-02 00:00
target_timestamp_72h = 2026-01-04 00:00
```

---

## 10. Feature Engineering

Feature engineering must be explicitly causal.

Potential feature groups:

### Lag features

```text
AQI_lag_1
AQI_lag_3
AQI_lag_6
AQI_lag_12
AQI_lag_24
AQI_lag_48
AQI_lag_72
```

### Rolling statistics

Examples:

```text
AQI rolling mean 3h
AQI rolling mean 6h
AQI rolling mean 24h
AQI rolling std 6h
AQI rolling std 24h
```

These must be calculated using only historical/current information.

### Change features

Examples:

```text
AQI_change_1h
AQI_change_6h
AQI_change_24h
PM25_change_1h
PM25_change_6h
```

### Temporal features

Examples:

```text
hour_sin
hour_cos
dow_sin
dow_cos
month_sin
month_cos
```

### Interaction features

Potential interactions between:

- pollutants
- temperature
- humidity
- wind
- pressure
- time-of-day

Feature engineering will be validated experimentally rather than assuming every feature is beneficial.

---

## 11. Forecasting Strategies to Compare

The project should not rely on one forecasting strategy.

### Strategy A — Persistence

The forecast is:

```text
ŷ(t+h) = AQI(t)
```

This is a critical benchmark.

### Strategy B — Statistical forecasting

Potential candidates:

- Seasonal Naive
- Moving average
- Exponential smoothing
- ARIMA/SARIMA
- SARIMAX where appropriate

### Strategy C — Direct horizon-specific models

Train separate models:

```text
Model_1h
Model_6h
Model_24h
Model_72h
```

This is an important initial machine-learning strategy.

### Strategy D — Multi-output model

One model predicts:

```text
[AQI(t+1), AQI(t+6), AQI(t+24), AQI(t+72)]
```

### Strategy E — Horizon-conditioned model

A model receives the forecast horizon as an input:

```text
X_t + horizon
```

and learns:

```text
f(X_t, h)
```

### Strategy F — Change/residual forecasting

Instead of directly predicting AQI:

```text
ΔAQI_h = AQI(t+h) - AQI(t)
```

The final forecast becomes:

```text
ŷ(t+h) = AQI(t) + predicted_ΔAQI_h
```

This will be tested rather than assumed to be superior.

### Strategy G — Hybrid approach

For example:

```text
Persistence baseline
        +
ML correction
        ↓
Final forecast
```

This is particularly interesting for long horizons.

---

## 12. Model Development Ladder

Complexity should increase only when justified.

### Level 0 — Naive baselines

- Persistence
- Seasonal persistence
- Moving average

### Level 1 — Statistical models

- Exponential smoothing
- ARIMA/SARIMA
- SARIMAX

### Level 2 — Classical ML

- Linear Regression
- Ridge
- Random Forest
- Gradient Boosting
- XGBoost

### Level 3 — Advanced temporal models

Only if required:

- LSTM
- GRU
- Temporal CNN
- Temporal Fusion Transformer
- Transformer-based time-series models
- Time-series foundation models

The objective is not to use the most sophisticated model. The objective is to establish the best reliable forecasting approach supported by evidence.

---

## 13. Evaluation Metrics

Every horizon should be evaluated independently.

### MAE

Mean Absolute Error:

```text
MAE = mean(|y - ŷ|)
```

Interpretation:

> Average absolute forecasting error in AQI units.

### RMSE

Root Mean Squared Error:

```text
RMSE = sqrt(mean((y - ŷ)^2))
```

RMSE penalises large errors more heavily.

### R²

Coefficient of determination.

Useful for understanding explained variance, but not sufficient for model selection by itself.

### Bias

Mean signed error:

```text
Bias = mean(ŷ - y)
```

Interpretation:

- Positive bias → systematic overprediction
- Negative bias → systematic underprediction

### Baseline improvement

For MAE:

```text
Improvement (%) =
((Baseline MAE - Model MAE) / Baseline MAE) × 100
```

Positive values indicate improvement over the baseline.

---

## 14. Error Analysis

Aggregate metrics are not enough.

The project should investigate:

### Error by horizon

```text
1h → expected strongest predictability
6h → intermediate
24h → substantially harder
72h → long-range uncertainty
```

### Error by AQI regime

Examples:

- Good
- Moderate
- Poor
- Very poor
- Severe

The exact categories depend on the selected AQI standard.

### Error during rapid changes

Identify periods where:

```text
|AQI(t+h) - AQI(t)| 
```

is large.

These cases are especially important because persistence may perform poorly during rapid transitions.

### Residual distribution

Inspect:

- mean
- standard deviation
- skew
- extreme errors
- systematic bias

### Prediction vs actual

Plot:

```text
Actual AQI
Predicted AQI
```

over time.

### Worst-error cases

For each horizon, inspect the largest errors and determine:

- What happened?
- Which variables changed?
- Was there a sudden event?
- Did the model lag behind?
- Was relevant future information unavailable?
- Was the event outside the training distribution?

---

## 15. Long-Horizon Predictability Investigation

24h and 72h forecasts require special attention.

Poor performance can result from:

1. insufficient historical context;
2. missing future weather information;
3. weak target formulation;
4. distribution shift;
5. abrupt pollution events;
6. seasonal changes;
7. model underfitting;
8. excessive model complexity;
9. intrinsic forecast uncertainty.

Therefore, a poor 72h score should not automatically lead to:

> "Use a bigger neural network."

Instead, investigate the information available to the model.

---

## 16. Leakage Prevention

The following practices are prohibited:

### Do not randomly shuffle time-series rows

```python
shuffle=True
```

should not be used for the final temporal evaluation.

### Do not fit preprocessing on the full dataset

Incorrect:

```python
scaler.fit(all_data)
```

Correct:

```text
fit on training data
transform training data
transform validation data
transform test data
```

### Do not calculate future-aware rolling features

Incorrect:

```text
rolling centered around t
```

Correct:

```text
rolling ending at t
```

### Do not use target information

Features must not contain:

```text
AQI(t+1)
AQI(t+6)
AQI(t+24)
AQI(t+72)
```

when predicting from origin `t`.

### Do not use actual future weather observations

If a future weather variable is used, it must represent information that would genuinely have been available at forecast time, such as a weather forecast.

---

## 17. Validation Strategy

The preferred evaluation design is chronological.

A basic structure is:

```text
|---------------- Training ----------------|---- Validation ----|---- Test ----|
Past                                                                  Future
```

For stronger validation, use walk-forward / rolling-origin evaluation:

```text
Train → Validate
Train expands → Validate
Train expands → Validate
...
Final untouched test period
```

This better represents real forecasting deployment.

---

## 18. Experiment Tracking

Each experiment should record:

- experiment ID
- date
- data version
- feature version
- target definition
- forecast horizon
- model
- hyperparameters
- training period
- validation period
- test period
- preprocessing
- MAE
- RMSE
- R²
- bias
- baseline improvement
- notes
- failure observations

Never rely solely on memory or notebook output.

---

## 19. Explainability

Potential explainability methods include:

- Feature importance
- Permutation importance
- SHAP
- Partial dependence
- Local explanations for individual forecasts

For tree-based models, SHAP can help answer:

> Why did the model predict this AQI?

Explainability should be added after the forecasting pipeline is stable.

---

## 20. Deployment Roadmap

The project may eventually provide:

### API

A FastAPI service can expose endpoints such as:

```text
GET /health
POST /forecast
```

Example conceptual request:

```json
{
  "timestamp": "2026-09-01T12:00:00",
  "horizons": [1, 6, 24, 72]
}
```

### Streamlit interface

Potential UI components:

- Current AQI
- 1h forecast
- 6h forecast
- 24h forecast
- 72h forecast
- Forecast chart
- Confidence/uncertainty interval
- Historical AQI
- Model information
- Feature importance
- Data quality status

### Feature store / model registry

A platform such as Hopsworks may be introduced later for:

- feature management
- point-in-time feature retrieval
- model versioning
- model registry
- reproducibility

This should not be added merely for technology demonstration. It should solve an actual project requirement.

---

## 21. Automation

Possible future automation:

```text
Scheduled job
     ↓
Fetch latest data
     ↓
Validate data
     ↓
Build features
     ↓
Generate forecast
     ↓
Quality checks
     ↓
Store prediction
     ↓
Update dashboard/API
```

Potential CI/CD tooling:

- GitHub Actions
- scheduled workflows
- automated tests
- model validation
- deployment pipeline

---

## 22. Testing

Important tests include:

### Data tests

- Timestamp uniqueness
- Hourly frequency
- Missing values
- Unexpected gaps
- Invalid ranges
- Duplicate records

### Target tests

Verify:

```text
target_1h(t) == AQI(t+1h)
target_6h(t) == AQI(t+6h)
target_24h(t) == AQI(t+24h)
target_72h(t) == AQI(t+72h)
```

### Feature tests

Verify that features do not accidentally reference future observations.

### Model tests

- Input schema
- Missing-value handling
- Prediction shape
- Output range sanity checks

---

## 23. What We Must NOT Do

The following anti-patterns are explicitly avoided:

- Do not start with deep learning.
- Do not assume XGBoost will be the final model.
- Do not optimise only for R².
- Do not randomly split the data.
- Do not tune against the test set.
- Do not use future observations as features.
- Do not silently delete difficult forecast periods.
- Do not clip predictions simply to improve metrics.
- Do not remove outliers merely because they hurt model performance.
- Do not repeatedly change the dataset until the metrics look good.
- Do not compare models trained/evaluated on different data definitions without documenting it.
- Do not report only the best horizon.
- Do not hide poor 24h/72h results.
- Do not introduce unnecessary infrastructure before the modelling problem is understood.

---

## 24. Project Success Criteria

A successful project is not necessarily the model with the highest R².

The project should demonstrate:

### Data quality

A reproducible and validated time-series dataset.

### Leakage control

A demonstrably causal forecasting pipeline.

### Baseline comparison

Evidence that proposed models outperform appropriate baselines where possible.

### Generalisation

Performance on genuinely unseen future periods.

### Horizon analysis

Clear understanding of why forecast quality changes from:

```text
1h → 6h → 24h → 72h
```

### Failure analysis

Transparent identification of situations where the model fails.

### Reproducibility

Another user should be able to clone the repository, install dependencies, and reproduce the major experiments.

### Deployment readiness

A working inference pipeline and, eventually, an API/dashboard.

---

## 25. Development Workflow

The recommended development sequence is:

```text
1. Repository setup
        ↓
2. Data source verification
        ↓
3. Data audit
        ↓
4. Data validation
        ↓
5. EDA
        ↓
6. Forecast target construction
        ↓
7. Leakage-safe feature engineering
        ↓
8. Chronological split
        ↓
9. Persistence baseline
        ↓
10. Statistical baselines
        ↓
11. Linear models
        ↓
12. Tree-based models
        ↓
13. Horizon strategy comparison
        ↓
14. Error analysis
        ↓
15. Long-horizon predictability investigation
        ↓
16. Advanced temporal models if justified
        ↓
17. Explainability
        ↓
18. Final model selection
        ↓
19. API
        ↓
20. Streamlit application
        ↓
21. Automation / CI-CD
```

---

## 26. Current Research Direction

The key research question is:

> **How can we build a leakage-free, generalising AQI forecasting system whose forecasting strategy is appropriate for each horizon from 1 hour to 72 hours?**

Sub-questions:

1. Does persistence remain a strong benchmark at each horizon?
2. Which features improve forecasts beyond persistence?
3. Does direct forecasting outperform other multi-horizon strategies?
4. Does predicting AQI change improve long-horizon stability?
5. Do future weather forecasts materially improve 24h/72h performance?
6. When does deterministic forecasting become unreliable?
7. Can uncertainty estimates provide more useful long-horizon forecasts?
8. Which model provides the best balance between accuracy, stability, explainability, and operational simplicity?

---

## 27. Reproducibility

Clone the repository:

```powershell
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd aqi-multi-horizon-forecasting
```

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run the project scripts according to the current development stage.

---

## 28. Environment Variables

Secrets and API credentials must never be committed to GitHub.

Use:

```text
.env
```

for local development and:

```text
.env.example
```

for documenting required variables.

Example:

```text
API_KEY=your_key_here
```

The real `.env` file should be included in `.gitignore`.

---

## 29. Git Workflow

Recommended workflow:

```powershell
git status
git add .
git commit -m "Add project README"
git push origin main
```

Before pushing, verify that sensitive files are not staged:

```powershell
git status
```

If `.env`, credentials, large datasets, model binaries, or virtual environments appear in the staged changes, stop and fix `.gitignore` before pushing.

---

## 30. License

Add the project's chosen license here when the repository is ready for public release.

For example:

```text
MIT License
```

Do not claim a license until the repository actually contains the corresponding license file.

---

## 31. Acknowledgements

Potential data and infrastructure sources should be documented here once finalised.

Examples may include:

- Open-Meteo
- CAMS / atmospheric reanalysis or forecast sources
- Hopsworks
- Scikit-learn
- XGBoost
- Pandas
- NumPy
- Matplotlib
- Streamlit
- FastAPI

The final README should document the exact data source, API, dataset version, and licensing terms actually used by the project.

---

## 32. Project Status

The project is under active development.

Current emphasis:

```text
Data quality
      ↓
Forecasting contract
      ↓
Leakage prevention
      ↓
Baseline establishment
      ↓
Multi-horizon modelling
      ↓
Failure analysis
      ↓
Long-horizon strategy
```

Deployment and advanced infrastructure will follow only after the forecasting methodology is validated.

---

## 33. Author

**Muhammad Khan**

Electrical Engineer | MBA | Data Analytics | Power BI | Machine Learning | AI

This project is being developed as a practical machine-learning/time-series forecasting project focused on solving a real-world environmental forecasting problem.
