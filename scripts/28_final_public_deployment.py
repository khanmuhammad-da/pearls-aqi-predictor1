"""
PEARLS AQI PREDICTOR
STEP 28 — FINAL PUBLIC PRODUCTION DEPLOYMENT

Purpose
-------
Deploy the EXISTING Step 23 Pearl Intelligence Dashboard together with
the EXISTING Step 25 Live AQI Intelligence API.

IMPORTANT:
- Does NOT redesign the dashboard.
- Does NOT regenerate HTML.
- Does NOT modify CSS.
- Does NOT modify JavaScript.
- Does NOT retrain models.
- Does NOT perform model selection.
- Does NOT create another intelligence layer.
- Does NOT create another dashboard.

It prepares a single production application and can launch it locally
for verification. For actual Internet exposure, use the deployment
platform's public service/port configuration.

Endpoints
---------
/                  -> Step 23 dashboard
/dashboard         -> Step 23 dashboard
/api/health        -> health check
/api/intelligence  -> Step 25 live intelligence
/docs              -> API documentation
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

STEP23_DIR = (
    PROJECT_ROOT
    / "reports"
    / "pearl_intelligence_dashboard"
)

DEPLOYMENT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "final_public_deployment"
)

APP_DIR = DEPLOYMENT_DIR / "app"
DASHBOARD_DIR = APP_DIR / "dashboard"
DATA_DIR = APP_DIR / "data"

DASHBOARD_HTML = (
    STEP23_DIR
    / "pearl_intelligence_dashboard.html"
)

DASHBOARD_JSON = (
    STEP23_DIR
    / "pearl_intelligence_dashboard_data.json"
)

FORECAST_CSV = (
    STEP23_DIR
    / "pearl_forecast.csv"
)

LIVE_INTELLIGENCE = (
    STEP23_DIR
    / "pearl_live_aqi_intelligence.json"
)

LIVE_POLLUTION = (
    STEP23_DIR
    / "pearl_live_aqi_pollution.csv"
)

LIVE_ALERTS = (
    STEP23_DIR
    / "pearl_live_aqi_alerts.csv"
)

LIVE_SUMMARY = (
    STEP23_DIR
    / "pearl_live_aqi_summary.csv"
)

HOST = os.environ.get("PEARLS_HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))


# ============================================================================
# DISPLAY
# ============================================================================

def banner(title: str):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def info(label: str, value):
    print(f"{label:<34}: {value}")


# ============================================================================
# UTILITIES
# ============================================================================

def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def require_file(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(
            f"{label} not found:\n{path}"
        )

    if path.stat().st_size == 0:
        raise RuntimeError(
            f"{label} is empty:\n{path}"
        )


def copy_file(source: Path, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


# ============================================================================
# VERIFY STEP 23
# ============================================================================

def verify_step23():

    banner("VERIFYING STEP 23 DASHBOARD")

    required = {
        "Dashboard HTML": DASHBOARD_HTML,
        "Dashboard data": DASHBOARD_JSON,
        "Forecast": FORECAST_CSV,
    }

    for label, path in required.items():
        require_file(path, label)
        info(label, f"PASS -> {path}")

    print()
    info(
        "Dashboard modification",
        "NONE"
    )

    return True


# ============================================================================
# VERIFY STEP 25
# ============================================================================

def verify_step25():

    banner("VERIFYING STEP 25 LIVE INTELLIGENCE")

    files = {
        "Live intelligence": LIVE_INTELLIGENCE,
        "Live pollution": LIVE_POLLUTION,
        "Live alerts": LIVE_ALERTS,
        "Live summary": LIVE_SUMMARY,
    }

    for label, path in files.items():

        if path.exists():
            info(
                label,
                f"FOUND -> {path}"
            )
        else:
            info(
                label,
                "NOT FOUND"
            )

    return True


# ============================================================================
# CREATE CLEAN DEPLOYMENT
# ============================================================================

def prepare_deployment():

    banner("PREPARING FINAL PUBLIC DEPLOYMENT")

    if DEPLOYMENT_DIR.exists():
        shutil.rmtree(DEPLOYMENT_DIR)

    DASHBOARD_DIR.mkdir(parents=True)
    DATA_DIR.mkdir(parents=True)

    # ------------------------------------------------------------
    # EXISTING STEP 23 DASHBOARD
    # ------------------------------------------------------------

    copy_file(
        DASHBOARD_HTML,
        DASHBOARD_DIR / DASHBOARD_HTML.name
    )

    copy_file(
        DASHBOARD_JSON,
        DATA_DIR / DASHBOARD_JSON.name
    )

    copy_file(
        FORECAST_CSV,
        DATA_DIR / FORECAST_CSV.name
    )

    # ------------------------------------------------------------
    # EXISTING STEP 25 LIVE DATA
    # ------------------------------------------------------------

    for source in [
        LIVE_INTELLIGENCE,
        LIVE_POLLUTION,
        LIVE_ALERTS,
        LIVE_SUMMARY,
    ]:

        if source.exists():
            copy_file(
                source,
                DATA_DIR / source.name
            )

    info(
        "Dashboard",
        "STEP 23 HTML COPIED UNCHANGED"
    )

    info(
        "Live intelligence",
        "STEP 25 DATA COPIED"
    )


# ============================================================================
# PRODUCTION SERVER
# ============================================================================

def create_server():

    banner("CREATING FINAL PRODUCTION SERVER")

    main_py = APP_DIR / "main.py"

    code = r'''
from pathlib import Path
import json
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles


BASE_DIR = Path(__file__).resolve().parent

DASHBOARD_DIR = BASE_DIR / "dashboard"
DATA_DIR = BASE_DIR / "data"

DASHBOARD = (
    DASHBOARD_DIR
    / "pearl_intelligence_dashboard.html"
)

app = FastAPI(
    title="Pearls AQI Predictor",
    description="Pearl Intelligence Dashboard and Live AQI Intelligence API",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# STATIC DATA
# ---------------------------------------------------------------------------

app.mount(
    "/data",
    StaticFiles(directory=str(DATA_DIR)),
    name="data",
)


# ---------------------------------------------------------------------------
# DASHBOARD
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def root():

    return FileResponse(
        DASHBOARD,
        media_type="text/html",
    )


@app.get("/dashboard", include_in_schema=False)
def dashboard():

    return FileResponse(
        DASHBOARD,
        media_type="text/html",
    )


# ---------------------------------------------------------------------------
# HEALTH
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():

    return {
        "status": "healthy",
        "service": "pearls-aqi-predictor",
        "dashboard": "step_23",
        "intelligence": "step_25",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# LIVE INTELLIGENCE
# ---------------------------------------------------------------------------

@app.get("/api/intelligence")
def intelligence():

    path = (
        DATA_DIR
        / "pearl_live_aqi_intelligence.json"
    )

    if not path.exists():

        return JSONResponse(
            status_code=404,
            content={
                "status": "unavailable",
                "message": "Live intelligence data not found.",
            },
        )

    try:

        with path.open(
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        return data

    except Exception as exc:

        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(exc),
            },
        )


# ---------------------------------------------------------------------------
# API INFO
# ---------------------------------------------------------------------------

@app.get("/api")
def api_info():

    return {
        "service": "Pearls AQI Predictor",
        "dashboard": "/",
        "health": "/api/health",
        "intelligence": "/api/intelligence",
        "documentation": "/docs",
    }
'''

    main_py.write_text(
        code.strip() + "\n",
        encoding="utf-8",
    )

    info(
        "Production server",
        f"CREATED -> {main_py}"
    )

    return main_py


# ============================================================================
# REQUIREMENTS
# ============================================================================

def create_requirements():

    requirements = APP_DIR / "requirements.txt"

    requirements.write_text(
        "fastapi>=0.110\n"
        "uvicorn[standard]>=0.29\n",
        encoding="utf-8",
    )

    info(
        "Requirements",
        f"CREATED -> {requirements}"
    )


# ============================================================================
# START SCRIPTS
# ============================================================================

def create_launchers():

    bat = DEPLOYMENT_DIR / "start_public_server.bat"

    bat.write_text(
        "@echo off\n"
        "cd /d \"%~dp0\"\n"
        "python -m uvicorn app.main:app "
        "--host 0.0.0.0 --port %PORT%\n",
        encoding="utf-8",
    )

    ps1 = DEPLOYMENT_DIR / "start_public_server.ps1"

    ps1.write_text(
        '$env:PORT = if ($env:PORT) { $env:PORT } else { "8000" }'
        "\n"
        'Set-Location $PSScriptRoot'
        "\n"
        'python -m uvicorn app.main:app --host 0.0.0.0 --port $env:PORT'
        "\n",
        encoding="utf-8",
    )

    info(
        "Windows launcher",
        f"CREATED -> {bat}"
    )

    info(
        "PowerShell launcher",
        f"CREATED -> {ps1}"
    )


# ============================================================================
# MANIFEST
# ============================================================================

def create_manifest():

    banner("CREATING DEPLOYMENT MANIFEST")

    files = []

    for path in DEPLOYMENT_DIR.rglob("*"):

        if path.is_file():

            files.append(
                {
                    "path": str(
                        path.relative_to(DEPLOYMENT_DIR)
                    ),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )

    manifest = {
        "project": "Pearls AQI Predictor",
        "step": 28,
        "deployment_type": "final_public_production",
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),

        "dashboard": {
            "source": "Step 23",
            "modified": False,
        },

        "live_intelligence": {
            "source": "Step 25",
            "modified": False,
        },

        "model_training": False,
        "model_selection": False,
        "model_retraining": False,

        "host": HOST,
        "port": PORT,

        "files": files,
    }

    path = (
        DEPLOYMENT_DIR
        / "deployment_manifest.json"
    )

    path.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    info(
        "Deployment manifest",
        f"SAVED -> {path}"
    )


# ============================================================================
# README
# ============================================================================

def create_readme():

    readme = DEPLOYMENT_DIR / "README.txt"

    text = f"""
PEARLS AQI PREDICTOR
STEP 28 — FINAL PUBLIC PRODUCTION DEPLOYMENT

This package contains the existing Pearl Intelligence Dashboard.

NO dashboard redesign was performed.

Dashboard:
    /

Dashboard:
    /dashboard

Health:
    /api/health

Live intelligence:
    /api/intelligence

API documentation:
    /docs

Local verification:

    python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

Then open:

    http://127.0.0.1:8000/

PUBLIC DEPLOYMENT

For a public Internet URL, deploy this directory to a hosting
provider that supports Python/FastAPI/ASGI applications.

The hosting platform must expose:

    0.0.0.0:$PORT

The application reads PORT from the environment.

IMPORTANT:

Do not use a file:// URL for production.

Do not expose private API keys in the dashboard.

The public URL is assigned by the hosting provider.
"""

    readme.write_text(
        text.strip() + "\n",
        encoding="utf-8",
    )

    info(
        "Deployment README",
        f"CREATED -> {readme}"
    )


# ============================================================================
# VALIDATION
# ============================================================================

def validate():

    banner("VALIDATING FINAL DEPLOYMENT")

    checks = {}

    checks["dashboard_html"] = (
        DASHBOARD_DIR
        / DASHBOARD_HTML.name
    ).exists()

    checks["dashboard_data"] = (
        DATA_DIR
        / DASHBOARD_JSON.name
    ).exists()

    checks["forecast"] = (
        DATA_DIR
        / FORECAST_CSV.name
    ).exists()

    checks["live_intelligence"] = (
        DATA_DIR
        / LIVE_INTELLIGENCE.name
    ).exists()

    checks["production_server"] = (
        APP_DIR / "main.py"
    ).exists()

    checks["requirements"] = (
        APP_DIR / "requirements.txt"
    ).exists()

    checks["manifest"] = (
        DEPLOYMENT_DIR
        / "deployment_manifest.json"
    ).exists()

    checks["readme"] = (
        DEPLOYMENT_DIR
        / "README.txt"
    ).exists()

    for name, passed in checks.items():

        info(
            name,
            "PASS" if passed else "FAIL"
        )

    return all(checks.values())


# ============================================================================
# MAIN
# ============================================================================

def main():

    started = time.time()

    banner("PEARLS AQI PREDICTOR")

    banner("STEP 28 — FINAL PUBLIC PRODUCTION DEPLOYMENT")

    info(
        "Base directory",
        PROJECT_ROOT
    )

    info(
        "Step 23 dashboard",
        STEP23_DIR
    )

    info(
        "Deployment directory",
        DEPLOYMENT_DIR
    )

    info(
        "Dashboard redesign",
        "NO"
    )

    info(
        "HTML regeneration",
        "NO"
    )

    info(
        "CSS modification",
        "NO"
    )

    info(
        "JavaScript modification",
        "NO"
    )

    info(
        "Model training",
        "NO"
    )

    info(
        "Model selection",
        "NO"
    )

    info(
        "Model retraining",
        "NO"
    )

    verify_step23()
    verify_step25()

    prepare_deployment()

    create_server()
    create_requirements()
    create_launchers()
    create_readme()
    create_manifest()

    passed = validate()

    elapsed = time.time() - started

    banner("STEP 28 COMPLETE")

    info(
        "Deployment status",
        "PASS" if passed else "FAIL"
    )

    info(
        "Step 23 UI",
        "PRESERVED — NOT CHANGED"
    )

    info(
        "Step 25 API",
        "PRESERVED — PACKAGED"
    )

    info(
        "Deployment directory",
        DEPLOYMENT_DIR
    )

    info(
        "Dashboard",
        "http://127.0.0.1:8000/"
    )

    info(
        "Live intelligence",
        "http://127.0.0.1:8000/api/intelligence"
    )

    info(
        "Health",
        "http://127.0.0.1:8000/api/health"
    )

    info(
        "Swagger",
        "http://127.0.0.1:8000/docs"
    )

    info(
        "Execution time",
        f"{elapsed:.3f}s"
    )

    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()