"""
PEARLS AQI PREDICTOR
STEP 26 — PRODUCTION DEPLOYMENT PREPARATION

Purpose
-------
Prepare the EXISTING Step 23 Pearl Intelligence Dashboard and
Step 25 Live AQI Intelligence API for deployment.

IMPORTANT
---------
This step does NOT:
- redesign the dashboard
- regenerate the dashboard HTML
- change the Step 23 UI
- retrain models
- select models
- retrain XGBoost
- modify forecast data
- modify pollution intelligence

It ONLY:
1. Verifies Step 23 dashboard assets.
2. Verifies Step 25 live intelligence assets.
3. Creates a clean deployment package.
4. Copies the existing Step 23 dashboard unchanged.
5. Copies the existing Step 25 API/intelligence data.
6. Creates a production FastAPI server.
7. Creates a Windows launcher.
8. Creates deployment metadata.
9. Creates a health/status endpoint.
10. Validates the deployment package.

Deployment package
------------------
reports/
    production_deployment/
        app/
            main.py
            dashboard/
                pearl_intelligence_dashboard.html
                pearl_intelligence_dashboard_data.json
                pearl_forecast.csv
                ...
            data/
                pearl_live_aqi_intelligence.json
                pearl_live_aqi_pollution.csv
                pearl_live_aqi_alerts.csv
                pearl_live_aqi_summary.csv
        start_server.bat
        start_server.ps1
        deployment_manifest.json
        deployment_report.json
        README_DEPLOYMENT.txt

Run
---
python scripts\\26_production_deployment.py

Then:
    cd reports\\production_deployment
    start_server.bat

Dashboard:
    http://127.0.0.1:8000/

API:
    http://127.0.0.1:8000/api/intelligence

Health:
    http://127.0.0.1:8000/api/health

Docs:
    http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

SCRIPT_PATH = Path(__file__).resolve()
BASE_DIR = SCRIPT_PATH.parent.parent

REPORTS_DIR = BASE_DIR / "reports"

STEP23_DIR = REPORTS_DIR / "pearl_intelligence_dashboard"

DEPLOYMENT_DIR = REPORTS_DIR / "production_deployment"

APP_DIR = DEPLOYMENT_DIR / "app"
DASHBOARD_DIR = APP_DIR / "dashboard"
DATA_DIR = APP_DIR / "data"

MAIN_PY = APP_DIR / "main.py"

MANIFEST_FILE = DEPLOYMENT_DIR / "deployment_manifest.json"
REPORT_FILE = DEPLOYMENT_DIR / "deployment_report.json"
README_FILE = DEPLOYMENT_DIR / "README_DEPLOYMENT.txt"

START_BAT = DEPLOYMENT_DIR / "start_server.bat"
START_PS1 = DEPLOYMENT_DIR / "start_server.ps1"


# ---------------------------------------------------------------------
# DISPLAY
# ---------------------------------------------------------------------

def banner(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def status(label: str, value: Any) -> None:
    print(f"{label:<32}: {value}")


# ---------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def safe_json_value(value: Any) -> Any:
    """
    Convert numpy/pandas-like values into JSON-safe values without
    requiring pandas or numpy.
    """
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            str(k): safe_json_value(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            safe_json_value(v)
            for v in value
        ]

    try:
        return value.item()
    except Exception:
        pass

    return str(value)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            safe_json_value(data),
            f,
            indent=2,
            ensure_ascii=False,
        )


def copy_file(
    source: Path,
    destination: Path,
) -> bool:
    if not source.exists():
        return False

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(source, destination)

    return True


def clean_deployment_directory() -> None:
    """
    Remove ONLY the Step 26 deployment package.

    Existing reports, models, dashboard files and data are untouched.
    """

    if DEPLOYMENT_DIR.exists():
        shutil.rmtree(DEPLOYMENT_DIR)

    DASHBOARD_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ---------------------------------------------------------------------
# STEP 23 FILES
# ---------------------------------------------------------------------

STEP23_REQUIRED_FILES = [
    "pearl_intelligence_dashboard.html",
    "pearl_intelligence_dashboard_data.json",
    "pearl_forecast.csv",
    "pearl_dashboard_summary.csv",
]

STEP23_OPTIONAL_FILES = [
    "pearl_intelligence_dashboard_events.csv",
    "pearl_intelligence_dashboard_forecast.csv",
    "pearl_intelligence_dashboard_pollution.csv",
    "pearl_intelligence_dashboard_summary.csv",
    "pearl_intelligence_dashboard_results.json",
    "pearl_intelligence_forecast.csv",
    "pearl_intelligence_pollution.csv",
    "pearl_intelligence_summary.csv",
    "pearl_pollution.csv",
]


# ---------------------------------------------------------------------
# STEP 25 FILES
# ---------------------------------------------------------------------

STEP25_FILES = [
    "pearl_live_aqi_intelligence.json",
    "pearl_live_aqi_pollution.csv",
    "pearl_live_aqi_alerts.csv",
    "pearl_live_aqi_summary.csv",
]


# ---------------------------------------------------------------------
# VALIDATE STEP 23
# ---------------------------------------------------------------------

def validate_step23() -> dict:
    banner("VERIFYING STEP 23 DASHBOARD")

    result = {
        "required": {},
        "optional": {},
        "all_required_present": True,
    }

    for filename in STEP23_REQUIRED_FILES:

        source = STEP23_DIR / filename
        found = source.exists() and source.is_file()

        result["required"][filename] = {
            "found": found,
            "path": str(source),
            "size_bytes": source.stat().st_size if found else 0,
        }

        status(
            filename,
            "FOUND" if found else "MISSING",
        )

        if not found:
            result["all_required_present"] = False

    for filename in STEP23_OPTIONAL_FILES:

        source = STEP23_DIR / filename
        found = source.exists() and source.is_file()

        result["optional"][filename] = {
            "found": found,
            "path": str(source),
            "size_bytes": source.stat().st_size if found else 0,
        }

        status(
            f"Optional {filename}",
            "FOUND" if found else "NOT FOUND",
        )

    return result


# ---------------------------------------------------------------------
# VALIDATE STEP 25
# ---------------------------------------------------------------------

def validate_step25() -> dict:
    banner("VERIFYING STEP 25 LIVE INTELLIGENCE")

    result = {
        "files": {},
        "all_present": True,
    }

    for filename in STEP25_FILES:

        source = STEP23_DIR / filename
        found = source.exists() and source.is_file()

        result["files"][filename] = {
            "found": found,
            "path": str(source),
            "size_bytes": source.stat().st_size if found else 0,
        }

        status(
            filename,
            "FOUND" if found else "NOT FOUND",
        )

        if not found:
            result["all_present"] = False

    return result


# ---------------------------------------------------------------------
# COPY STEP 23
# ---------------------------------------------------------------------

def package_step23() -> list[dict]:
    banner("PACKAGING EXISTING STEP 23 DASHBOARD")

    copied = []

    all_files = (
        STEP23_REQUIRED_FILES
        + STEP23_OPTIONAL_FILES
    )

    seen = set()

    for filename in all_files:

        if filename in seen:
            continue

        seen.add(filename)

        source = STEP23_DIR / filename

        if not source.exists():
            continue

        destination = DASHBOARD_DIR / filename

        copy_file(
            source,
            destination,
        )

        entry = {
            "filename": filename,
            "source": str(source),
            "destination": str(destination),
            "size_bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
        }

        copied.append(entry)

        status(
            filename,
            f"COPIED ({destination.stat().st_size:,} bytes)",
        )

    return copied


# ---------------------------------------------------------------------
# COPY STEP 25
# ---------------------------------------------------------------------

def package_step25() -> list[dict]:
    banner("PACKAGING STEP 25 LIVE INTELLIGENCE")

    copied = []

    for filename in STEP25_FILES:

        source = STEP23_DIR / filename

        if not source.exists():
            continue

        destination = DATA_DIR / filename

        copy_file(
            source,
            destination,
        )

        entry = {
            "filename": filename,
            "source": str(source),
            "destination": str(destination),
            "size_bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
        }

        copied.append(entry)

        status(
            filename,
            f"COPIED ({destination.stat().st_size:,} bytes)",
        )

    return copied


# ---------------------------------------------------------------------
# SERVER
# ---------------------------------------------------------------------

SERVER_CODE = r'''
"""
PEARLS AQI PREDICTOR
STEP 26 — PRODUCTION SERVER

This server serves the EXISTING Step 23 dashboard.

It does NOT generate or redesign the dashboard.

It exposes:

    /
    /dashboard

    /api/health
    /api/intelligence
    /api/pollution
    /api/alerts
    /api/summary
    /api/forecast

    /docs
"""

from __future__ import annotations

import json
import mimetypes
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware


BASE_DIR = Path(__file__).resolve().parent

DASHBOARD_DIR = BASE_DIR / "dashboard"
DATA_DIR = BASE_DIR / "data"

DASHBOARD_HTML = DASHBOARD_DIR / "pearl_intelligence_dashboard.html"

INTELLIGENCE_FILE = DATA_DIR / "pearl_live_aqi_intelligence.json"
POLLUTION_FILE = DATA_DIR / "pearl_live_aqi_pollution.csv"
ALERTS_FILE = DATA_DIR / "pearl_live_aqi_alerts.csv"
SUMMARY_FILE = DATA_DIR / "pearl_live_aqi_summary.csv"

FORECAST_FILE = DASHBOARD_DIR / "pearl_forecast.csv"


app = FastAPI(
    title="PEARLS AQI Predictor",
    description="Production API for the existing Pearl Intelligence Dashboard.",
    version="26.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def read_json(path: Path):
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"File not found: {path.name}",
        )

    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


def read_text(path: Path):
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"File not found: {path.name}",
        )

    return path.read_text(
        encoding="utf-8",
        errors="replace",
    )


@app.get("/")
def dashboard():
    if not DASHBOARD_HTML.exists():
        raise HTTPException(
            status_code=404,
            detail="Step 23 dashboard HTML not found.",
        )

    return FileResponse(
        DASHBOARD_HTML,
        media_type="text/html",
    )


@app.get("/dashboard")
def dashboard_alias():
    return dashboard()


@app.get("/api/health")
def health():

    files = {
        "dashboard": DASHBOARD_HTML,
        "intelligence": INTELLIGENCE_FILE,
        "pollution": POLLUTION_FILE,
        "alerts": ALERTS_FILE,
        "summary": SUMMARY_FILE,
        "forecast": FORECAST_FILE,
    }

    availability = {
        name: path.exists()
        for name, path in files.items()
    }

    return {
        "status": "healthy"
        if all(availability.values())
        else "degraded",

        "service": "PEARLS AQI Predictor",

        "step": 26,

        "dashboard": "Step 23 existing dashboard",

        "dashboard_redesign": False,

        "model_training": False,

        "model_selection": False,

        "model_retraining": False,

        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "files": availability,
    }


@app.get("/api/intelligence")
def intelligence():
    return read_json(INTELLIGENCE_FILE)


@app.get("/api/pollution")
def pollution():

    return JSONResponse(
        content={
            "filename": POLLUTION_FILE.name,
            "csv": read_text(POLLUTION_FILE),
        }
    )


@app.get("/api/alerts")
def alerts():

    return JSONResponse(
        content={
            "filename": ALERTS_FILE.name,
            "csv": read_text(ALERTS_FILE),
        }
    )


@app.get("/api/summary")
def summary():

    return JSONResponse(
        content={
            "filename": SUMMARY_FILE.name,
            "csv": read_text(SUMMARY_FILE),
        }
    )


@app.get("/api/forecast")
def forecast():

    return JSONResponse(
        content={
            "filename": FORECAST_FILE.name,
            "csv": read_text(FORECAST_FILE),
        }
    )


@app.get("/api/files")
def files():

    return {
        "dashboard": str(
            DASHBOARD_HTML.relative_to(BASE_DIR)
        ),

        "intelligence": str(
            INTELLIGENCE_FILE.relative_to(BASE_DIR)
        ),

        "pollution": str(
            POLLUTION_FILE.relative_to(BASE_DIR)
        ),

        "alerts": str(
            ALERTS_FILE.relative_to(BASE_DIR)
        ),

        "summary": str(
            SUMMARY_FILE.relative_to(BASE_DIR)
        ),

        "forecast": str(
            FORECAST_FILE.relative_to(BASE_DIR)
        ),
    }


if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
'''


# ---------------------------------------------------------------------
# WRITE SERVER
# ---------------------------------------------------------------------

def create_server() -> None:
    banner("CREATING PRODUCTION SERVER")

    MAIN_PY.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    MAIN_PY.write_text(
        SERVER_CODE.strip() + "\n",
        encoding="utf-8",
    )

    status(
        "Production API server",
        f"CREATED -> {MAIN_PY}",
    )


# ---------------------------------------------------------------------
# WINDOWS BAT
# ---------------------------------------------------------------------

BAT_CODE = r'''@echo off
setlocal

cd /d "%~dp0"

echo ================================================================
echo PEARLS AQI PREDICTOR
echo STEP 26 - PRODUCTION SERVER
echo ================================================================
echo.
echo Dashboard:
echo http://127.0.0.1:8000/
echo.
echo API:
echo http://127.0.0.1:8000/api/intelligence
echo.
echo Health:
echo http://127.0.0.1:8000/api/health
echo.
echo Swagger:
echo http://127.0.0.1:8000/docs
echo.
echo Press CTRL+C to stop.
echo ================================================================
echo.

python app\main.py

pause
'''


def create_bat() -> None:
    START_BAT.write_text(
        BAT_CODE,
        encoding="utf-8",
    )

    status(
        "Windows launcher",
        f"CREATED -> {START_BAT}",
    )


# ---------------------------------------------------------------------
# POWERSHELL
# ---------------------------------------------------------------------

PS1_CODE = r'''Set-Location $PSScriptRoot

Write-Host "================================================================"
Write-Host "PEARLS AQI PREDICTOR"
Write-Host "STEP 26 - PRODUCTION SERVER"
Write-Host "================================================================"
Write-Host ""
Write-Host "Dashboard:"
Write-Host "http://127.0.0.1:8000/"
Write-Host ""
Write-Host "API:"
Write-Host "http://127.0.0.1:8000/api/intelligence"
Write-Host ""
Write-Host "Health:"
Write-Host "http://127.0.0.1:8000/api/health"
Write-Host ""
Write-Host "Swagger:"
Write-Host "http://127.0.0.1:8000/docs"
Write-Host ""

python app\main.py
'''


def create_ps1() -> None:
    START_PS1.write_text(
        PS1_CODE,
        encoding="utf-8",
    )

    status(
        "PowerShell launcher",
        f"CREATED -> {START_PS1}",
    )


# ---------------------------------------------------------------------
# README
# ---------------------------------------------------------------------

README_TEXT = r'''
================================================================
PEARLS AQI PREDICTOR
STEP 26 — PRODUCTION DEPLOYMENT
================================================================

PURPOSE
-------

This directory is the deployment package for the existing
Pearl Intelligence Dashboard.

The dashboard UI comes directly from STEP 23.

STEP 26 DOES NOT REDESIGN THE UI.


DIRECTORY
---------

app/
    main.py

    dashboard/
        Existing Step 23 dashboard files

    data/
        Existing Step 25 live intelligence files


START SERVER
------------

Windows CMD:

    start_server.bat


PowerShell:

    .\start_server.ps1


Or directly:

    python app\main.py


DASHBOARD
---------

    http://127.0.0.1:8000/


ALTERNATIVE DASHBOARD
---------------------

    http://127.0.0.1:8000/dashboard


HEALTH
------

    http://127.0.0.1:8000/api/health


LIVE INTELLIGENCE
-----------------

    http://127.0.0.1:8000/api/intelligence


POLLUTION
---------

    http://127.0.0.1:8000/api/pollution


ALERTS
------

    http://127.0.0.1:8000/api/alerts


SUMMARY
-------

    http://127.0.0.1:8000/api/summary


FORECAST
--------

    http://127.0.0.1:8000/api/forecast


API DOCUMENTATION
-----------------

    http://127.0.0.1:8000/docs


IMPORTANT
---------

127.0.0.1 means the server is available only on this computer.

For Internet access, the next deployment step must place this
application on a public server/cloud platform.

Do NOT expose this development server directly to the Internet
without proper production hosting, HTTPS, authentication where
appropriate, monitoring, and security configuration.


UI PRESERVATION
---------------

The Step 23 HTML is copied byte-for-byte.

Step 26 does not regenerate it.

Step 26 does not alter CSS.

Step 26 does not alter JavaScript.

Step 26 does not alter dashboard layout.


MODEL
-----

No model training is performed.

No model selection is performed.

No model retraining is performed.


DATA
----

Step 25 live intelligence is packaged as deployment data.

The deployment API reads these files.

This means the deployment layer is separated from model
training and dashboard generation.


NEXT STAGE
----------

After Step 26 is validated, the next stage is actual hosting /
cloud deployment.

Possible deployment targets include:

    - Replit
    - Render
    - Railway
    - Azure
    - AWS
    - Google Cloud
    - VPS

The choice depends on the required production architecture.
'''


def create_readme() -> None:
    README_FILE.write_text(
        README_TEXT.strip() + "\n",
        encoding="utf-8",
    )

    status(
        "Deployment README",
        f"CREATED -> {README_FILE}",
    )


# ---------------------------------------------------------------------
# MANIFEST
# ---------------------------------------------------------------------

def create_manifest(
    step23_validation: dict,
    step25_validation: dict,
    dashboard_files: list[dict],
    data_files: list[dict],
) -> dict:

    return {
        "project": "PEARLS AQI Predictor",

        "step": 26,

        "step_name": "Production Deployment",

        "created_utc": utc_now(),

        "base_directory": str(BASE_DIR),

        "deployment_directory": str(
            DEPLOYMENT_DIR
        ),

        "dashboard_source": str(
            STEP23_DIR
        ),

        "dashboard_source_step": 23,

        "live_intelligence_source_step": 25,

        "ui_changed": False,

        "dashboard_redesigned": False,

        "html_regenerated": False,

        "css_changed": False,

        "javascript_changed": False,

        "model_training": False,

        "model_selection": False,

        "model_retraining": False,

        "step23_validation": step23_validation,

        "step25_validation": step25_validation,

        "dashboard_files": dashboard_files,

        "live_data_files": data_files,

        "server": {
            "framework": "FastAPI",
            "host": "0.0.0.0",
            "port": 8000,
            "entrypoint": "app/main.py",
        },

        "endpoints": {
            "dashboard": "/",
            "dashboard_alias": "/dashboard",
            "health": "/api/health",
            "intelligence": "/api/intelligence",
            "pollution": "/api/pollution",
            "alerts": "/api/alerts",
            "summary": "/api/summary",
            "forecast": "/api/forecast",
            "files": "/api/files",
            "docs": "/docs",
        },
    }


# ---------------------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------------------

def validate_deployment(
    step23_validation: dict,
    step25_validation: dict,
) -> dict:

    banner("VALIDATING PRODUCTION DEPLOYMENT")

    checks = {}

    # Step 23 required files
    checks["step23_required"] = (
        step23_validation["all_required_present"]
    )

    status(
        "Step 23 source",
        "PASS"
        if checks["step23_required"]
        else "FAIL",
    )

    # Dashboard HTML
    dashboard_html = (
        DASHBOARD_DIR
        / "pearl_intelligence_dashboard.html"
    )

    checks["dashboard_html"] = (
        dashboard_html.exists()
        and dashboard_html.stat().st_size > 1000
    )

    status(
        "Dashboard HTML",
        "PASS" if checks["dashboard_html"] else "FAIL",
    )

    # Dashboard JSON
    dashboard_json = (
        DASHBOARD_DIR
        / "pearl_intelligence_dashboard_data.json"
    )

    json_valid = False

    if dashboard_json.exists():

        try:

            with dashboard_json.open(
                "r",
                encoding="utf-8",
            ) as f:
                json.load(f)

            json_valid = True

        except Exception:
            json_valid = False

    checks["dashboard_json"] = json_valid

    status(
        "Dashboard JSON",
        "PASS" if json_valid else "FAIL",
    )

    # Forecast
    forecast = (
        DASHBOARD_DIR
        / "pearl_forecast.csv"
    )

    checks["forecast"] = (
        forecast.exists()
        and forecast.stat().st_size > 0
    )

    status(
        "Forecast",
        "PASS" if checks["forecast"] else "FAIL",
    )

    # Step 25
    checks["step25_live_data"] = True

    for filename in STEP25_FILES:

        destination = DATA_DIR / filename

        if not destination.exists():
            checks["step25_live_data"] = False

    status(
        "Step 25 live data",
        "PASS"
        if checks["step25_live_data"]
        else "FAIL",
    )

    # Server
    checks["server"] = (
        MAIN_PY.exists()
        and MAIN_PY.stat().st_size > 1000
    )

    status(
        "Production server",
        "PASS" if checks["server"] else "FAIL",
    )

    # Launchers
    checks["bat"] = START_BAT.exists()
    checks["ps1"] = START_PS1.exists()

    status(
        "Windows launcher",
        "PASS" if checks["bat"] else "FAIL",
    )

    status(
        "PowerShell launcher",
        "PASS" if checks["ps1"] else "FAIL",
    )

    # README
    checks["readme"] = README_FILE.exists()

    status(
        "Deployment README",
        "PASS" if checks["readme"] else "FAIL",
    )

    checks["overall"] = all(
        checks.values()
    )

    status(
        "Overall deployment validation",
        "PASS"
        if checks["overall"]
        else "FAIL",
    )

    return checks


# ---------------------------------------------------------------------
# REPORT
# ---------------------------------------------------------------------

def create_report(
    validation: dict,
    manifest: dict,
    elapsed: float,
) -> dict:

    report = {
        "project": "PEARLS AQI Predictor",

        "step": 26,

        "step_name": "Production Deployment",

        "status": (
            "PASS"
            if validation["overall"]
            else "FAIL"
        ),

        "timestamp_utc": utc_now(),

        "execution_seconds": round(
            elapsed,
            3,
        ),

        "deployment_directory": str(
            DEPLOYMENT_DIR
        ),

        "dashboard": {
            "source_step": 23,
            "redesigned": False,
            "html_regenerated": False,
            "ui_changed": False,
        },

        "live_intelligence": {
            "source_step": 25,
            "packaged": True,
        },

        "model": {
            "training": False,
            "selection": False,
            "retraining": False,
        },

        "validation": validation,

        "manifest": manifest,
    }

    write_json(
        REPORT_FILE,
        report,
    )

    return report


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main() -> int:

    started = time.perf_counter()

    banner("PEARLS AQI PREDICTOR")
    banner("STEP 26 — PRODUCTION DEPLOYMENT")

    status(
        "Base directory",
        BASE_DIR,
    )

    status(
        "Step 23 dashboard",
        STEP23_DIR,
    )

    status(
        "Deployment directory",
        DEPLOYMENT_DIR,
    )

    status(
        "Dashboard redesign",
        "NO",
    )

    status(
        "HTML regeneration",
        "NO",
    )

    status(
        "CSS modification",
        "NO",
    )

    status(
        "JavaScript modification",
        "NO",
    )

    status(
        "Model training",
        "NO",
    )

    status(
        "Model selection",
        "NO",
    )

    status(
        "Model retraining",
        "NO",
    )

    # -------------------------------------------------------------
    # STEP 23
    # -------------------------------------------------------------

    step23_validation = validate_step23()

    if not step23_validation["all_required_present"]:

        banner("STEP 26 FAILED")

        print(
            "Required Step 23 dashboard files are missing."
        )

        print(
            f"Expected directory:\n{STEP23_DIR}"
        )

        return 1

    # -------------------------------------------------------------
    # STEP 25
    # -------------------------------------------------------------

    step25_validation = validate_step25()

    # -------------------------------------------------------------
    # CLEAN DEPLOYMENT DIRECTORY
    # -------------------------------------------------------------

    banner("CREATING CLEAN DEPLOYMENT PACKAGE")

    clean_deployment_directory()

    status(
        "Deployment directory",
        f"CREATED -> {DEPLOYMENT_DIR}",
    )

    # -------------------------------------------------------------
    # PACKAGE STEP 23
    # -------------------------------------------------------------

    dashboard_files = package_step23()

    # -------------------------------------------------------------
    # PACKAGE STEP 25
    # -------------------------------------------------------------

    data_files = package_step25()

    # -------------------------------------------------------------
    # CREATE SERVER
    # -------------------------------------------------------------

    create_server()

    # -------------------------------------------------------------
    # CREATE LAUNCHERS
    # -------------------------------------------------------------

    create_bat()
    create_ps1()

    # -------------------------------------------------------------
    # README
    # -------------------------------------------------------------

    create_readme()

    # -------------------------------------------------------------
    # MANIFEST
    # -------------------------------------------------------------

    banner("CREATING DEPLOYMENT MANIFEST")

    manifest = create_manifest(
        step23_validation=step23_validation,
        step25_validation=step25_validation,
        dashboard_files=dashboard_files,
        data_files=data_files,
    )

    write_json(
        MANIFEST_FILE,
        manifest,
    )

    status(
        "Deployment manifest",
        f"SAVED -> {MANIFEST_FILE}",
    )

    # -------------------------------------------------------------
    # VALIDATE
    # -------------------------------------------------------------

    validation = validate_deployment(
        step23_validation,
        step25_validation,
    )

    # -------------------------------------------------------------
    # REPORT
    # -------------------------------------------------------------

    elapsed = (
        time.perf_counter()
        - started
    )

    report = create_report(
        validation=validation,
        manifest=manifest,
        elapsed=elapsed,
    )

    # -------------------------------------------------------------
    # FINAL
    # -------------------------------------------------------------

    if not validation["overall"]:

        banner("STEP 26 FAILED")

        status(
            "Deployment status",
            "FAIL",
        )

        status(
            "Report",
            REPORT_FILE,
        )

        return 1

    banner("STEP 26 COMPLETE")

    status(
        "Deployment status",
        "PASS",
    )

    status(
        "Step 23 UI",
        "PRESERVED — NOT CHANGED",
    )

    status(
        "Step 25 intelligence",
        "PACKAGED",
    )

    status(
        "Dashboard HTML",
        f"{DASHBOARD_DIR / 'pearl_intelligence_dashboard.html'}",
    )

    status(
        "Production server",
        MAIN_PY,
    )

    status(
        "Windows launcher",
        START_BAT,
    )

    status(
        "PowerShell launcher",
        START_PS1,
    )

    status(
        "Deployment manifest",
        MANIFEST_FILE,
    )

    status(
        "Deployment README",
        README_FILE,
    )

    status(
        "Deployment report",
        REPORT_FILE,
    )

    print()
    print("URLs after starting the server:")
    print()
    print("Dashboard:")
    print("http://127.0.0.1:8000/")
    print()
    print("Live intelligence:")
    print("http://127.0.0.1:8000/api/intelligence")
    print()
    print("Health:")
    print("http://127.0.0.1:8000/api/health")
    print()
    print("Swagger:")
    print("http://127.0.0.1:8000/docs")
    print()

    status(
        "Execution time",
        f"{elapsed:.3f}s",
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(
            main()
        )

    except KeyboardInterrupt:

        print()
        print("STEP 26 interrupted.")

        raise SystemExit(130)

    except Exception as exc:

        banner("STEP 26 FAILED")

        print(
            f"{type(exc).__name__}: {exc}"
        )

        raise SystemExit(1)