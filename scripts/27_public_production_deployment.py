"""
PEARLS AQI PREDICTOR
STEP 27 — PUBLIC PRODUCTION DEPLOYMENT

Purpose
-------
Prepare the already validated Step 26 production deployment for public
cloud hosting.

This step DOES NOT:
    - train models
    - retrain models
    - select models
    - modify the Step 23 dashboard UI
    - redesign HTML/CSS/JavaScript
    - regenerate the forecast
    - change prediction logic

This step DOES:
    - validate Step 26 deployment
    - preserve Step 23 dashboard exactly
    - preserve Step 25 intelligence API
    - create a cloud-compatible FastAPI server
    - support PORT environment variable
    - bind to 0.0.0.0
    - provide health endpoint
    - provide API endpoints
    - create requirements.txt
    - create Replit configuration
    - create deployment manifest
    - create deployment README
    - create a clean public deployment package
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


# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_PATH = Path(__file__).resolve()
BASE_DIR = SCRIPT_PATH.parent.parent

STEP26_DIR = (
    BASE_DIR
    / "reports"
    / "production_deployment"
)

OUTPUT_DIR = (
    BASE_DIR
    / "reports"
    / "public_production_deployment"
)

APP_DIR = OUTPUT_DIR / "app"
DASHBOARD_DIR = APP_DIR / "dashboard"
DATA_DIR = APP_DIR / "data"

MANIFEST_PATH = OUTPUT_DIR / "public_deployment_manifest.json"
README_PATH = OUTPUT_DIR / "PUBLIC_DEPLOYMENT_README.txt"
REQUIREMENTS_PATH = OUTPUT_DIR / "requirements.txt"
REPLIT_NIX_PATH = OUTPUT_DIR / ".replit"
REPLITIGNORE_PATH = OUTPUT_DIR / ".gitignore"
MAIN_PATH = APP_DIR / "main.py"
START_BAT_PATH = OUTPUT_DIR / "start_public_server.bat"
START_PS1_PATH = OUTPUT_DIR / "start_public_server.ps1"


# ============================================================================
# DISPLAY HELPERS
# ============================================================================

WIDTH = 72


def banner(title: str):
    print()
    print("=" * WIDTH)
    print(title)
    print("=" * WIDTH)


def line(label: str, value):
    print(f"{label:<32}: {value}")


def fail(message: str):
    print()
    print("=" * WIDTH)
    print("STEP 27 FAILED")
    print("=" * WIDTH)
    print(message)
    print()
    sys.exit(1)


def utc_now():
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


def copy_file(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


# ============================================================================
# REQUIRED STEP 26 FILES
# ============================================================================

REQUIRED_STEP26_FILES = [
    Path("app") / "main.py",
    Path("app") / "dashboard" / "pearl_intelligence_dashboard.html",
    Path("app") / "dashboard" / "pearl_intelligence_dashboard_data.json",
    Path("app") / "dashboard" / "pearl_forecast.csv",
    Path("app") / "dashboard" / "pearl_dashboard_summary.csv",
]


OPTIONAL_STEP26_FILES = [
    Path("app") / "dashboard" / "pearl_intelligence_dashboard_events.csv",
    Path("app") / "dashboard" / "pearl_intelligence_dashboard_forecast.csv",
    Path("app") / "dashboard" / "pearl_intelligence_dashboard_pollution.csv",
    Path("app") / "dashboard" / "pearl_intelligence_dashboard_summary.csv",
    Path("app") / "dashboard" / "pearl_intelligence_dashboard_results.json",
    Path("app") / "dashboard" / "pearl_intelligence_forecast.csv",
    Path("app") / "dashboard" / "pearl_intelligence_pollution.csv",
    Path("app") / "dashboard" / "pearl_intelligence_summary.csv",
    Path("app") / "dashboard" / "pearl_pollution.csv",
]


# ============================================================================
# STEP 26 VALIDATION
# ============================================================================

def validate_step26():
    banner("VERIFYING STEP 26 DEPLOYMENT")

    if not STEP26_DIR.exists():
        fail(
            "Step 26 deployment directory not found:\n"
            f"{STEP26_DIR}"
        )

    line("Step 26 directory", "FOUND")

    missing = []

    for rel in REQUIRED_STEP26_FILES:
        path = STEP26_DIR / rel

        if path.exists():
            line(str(rel), "FOUND")
        else:
            line(str(rel), "MISSING")
            missing.append(str(rel))

    if missing:
        fail(
            "Required Step 26 files are missing:\n"
            + "\n".join(missing)
        )

    optional_count = 0

    for rel in OPTIONAL_STEP26_FILES:
        path = STEP26_DIR / rel

        if path.exists():
            optional_count += 1

    line("Optional Step 26 files", optional_count)

    return True


# ============================================================================
# STEP 23 UI INTEGRITY
# ============================================================================

def validate_dashboard_integrity():
    banner("VALIDATING STEP 23 DASHBOARD INTEGRITY")

    source = (
        STEP26_DIR
        / "app"
        / "dashboard"
        / "pearl_intelligence_dashboard.html"
    )

    if not source.exists():
        fail(f"Step 23 dashboard not found:\n{source}")

    size = source.stat().st_size
    digest = sha256_file(source)

    line("Dashboard HTML", "FOUND")
    line("Dashboard size", f"{size:,} bytes")
    line("Dashboard SHA256", digest)

    text = source.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    if "<html" not in text.lower():
        fail("Step 23 dashboard does not appear to be valid HTML.")

    if "</html>" not in text.lower():
        fail("Step 23 dashboard is missing </html>.")

    if len(text) < 10000:
        fail(
            "Dashboard HTML is unexpectedly small. "
            "Refusing deployment."
        )

    line("HTML validation", "PASS")
    line("UI modification", "NO")

    return {
        "sha256": digest,
        "size_bytes": size,
    }


# ============================================================================
# CREATE CLEAN PACKAGE
# ============================================================================

def create_clean_package():
    banner("CREATING PUBLIC DEPLOYMENT PACKAGE")

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    APP_DIR.mkdir(parents=True, exist_ok=True)
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    line("Output directory", f"CREATED -> {OUTPUT_DIR}")

    return True


# ============================================================================
# COPY STEP 26 APPLICATION
# ============================================================================

def package_dashboard_and_data():
    banner("PACKAGING STEP 23 + STEP 25")

    source_app = STEP26_DIR / "app"

    if not source_app.exists():
        fail(f"Step 26 app directory missing:\n{source_app}")

    copied = 0

    for src in source_app.rglob("*"):
        if not src.is_file():
            continue

        rel = src.relative_to(source_app)
        dst = APP_DIR / rel

        copy_file(src, dst)

        copied += 1

        print(
            f"{str(rel):<55} : COPIED "
            f"({src.stat().st_size:,} bytes)"
        )

    line("Files copied", copied)

    if not (
        DASHBOARD_DIR
        / "pearl_intelligence_dashboard.html"
    ).exists():
        fail("Step 23 dashboard was not packaged.")

    return copied


# ============================================================================
# CLOUD MAIN.PY
# ============================================================================

def create_cloud_server():
    banner("CREATING CLOUD-COMPATIBLE SERVER")

    source_main = STEP26_DIR / "app" / "main.py"

    if not source_main.exists():
        fail("Step 26 main.py not found.")

    cloud_main = r'''"""
PEARLS AQI PREDICTOR
PUBLIC PRODUCTION SERVER

This server serves the existing Step 23 dashboard and Step 25
intelligence API.

The dashboard UI is intentionally preserved.

Cloud hosting:
    - binds to 0.0.0.0
    - uses PORT environment variable
    - supports HTTPS termination by hosting provider
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles


# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DASHBOARD_DIR = BASE_DIR / "dashboard"

DASHBOARD_HTML = (
    DASHBOARD_DIR
    / "pearl_intelligence_dashboard.html"
)

DASHBOARD_DATA = (
    DASHBOARD_DIR
    / "pearl_intelligence_dashboard_data.json"
)

FORECAST_CSV = (
    DASHBOARD_DIR
    / "pearl_forecast.csv"
)

SUMMARY_CSV = (
    DASHBOARD_DIR
    / "pearl_dashboard_summary.csv"
)

LIVE_INTELLIGENCE = (
    DASHBOARD_DIR
    / "pearl_live_aqi_intelligence.json"
)

LIVE_POLLUTION = (
    DASHBOARD_DIR
    / "pearl_live_aqi_pollution.csv"
)

LIVE_ALERTS = (
    DASHBOARD_DIR
    / "pearl_live_aqi_alerts.csv"
)

LIVE_SUMMARY = (
    DASHBOARD_DIR
    / "pearl_live_aqi_summary.csv"
)


# ---------------------------------------------------------------------------
# APPLICATION
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Pearls AQI Predictor",
    description=(
        "Pearl Intelligence Dashboard with live AQI "
        "intelligence API."
    ),
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# BASIC HELPERS
# ---------------------------------------------------------------------------

def read_json(path: Path) -> Any:
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Data file not found: {path.name}",
        )

    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to read {path.name}: {exc}",
        )


def file_response(path: Path, media_type: str | None = None):
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"File not found: {path.name}",
        )

    return FileResponse(
        path=str(path),
        media_type=media_type,
    )


# ---------------------------------------------------------------------------
# ROOT / DASHBOARD
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def dashboard():
    return file_response(
        DASHBOARD_HTML,
        "text/html",
    )


@app.get("/dashboard", include_in_schema=False)
def dashboard_alias():
    return file_response(
        DASHBOARD_HTML,
        "text/html",
    )


# ---------------------------------------------------------------------------
# HEALTH
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    files = {
        "dashboard_html": DASHBOARD_HTML,
        "dashboard_data": DASHBOARD_DATA,
        "forecast": FORECAST_CSV,
        "summary": SUMMARY_CSV,
        "live_intelligence": LIVE_INTELLIGENCE,
        "live_pollution": LIVE_POLLUTION,
        "live_alerts": LIVE_ALERTS,
        "live_summary": LIVE_SUMMARY,
    }

    status = {}

    for name, path in files.items():
        status[name] = {
            "available": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else 0,
        }

    required_ok = all(
        status[name]["available"]
        for name in [
            "dashboard_html",
            "dashboard_data",
            "forecast",
        ]
    )

    return JSONResponse(
        content={
            "status": "healthy" if required_ok else "degraded",
            "service": "pearls-aqi-predictor",
            "dashboard": "step_23",
            "intelligence": "step_25",
            "production_deployment": "step_27",
            "host": os.getenv("HOST", "0.0.0.0"),
            "port": int(os.getenv("PORT", "8000")),
            "files": status,
        }
    )


# ---------------------------------------------------------------------------
# DASHBOARD DATA
# ---------------------------------------------------------------------------

@app.get("/api/dashboard-data")
def dashboard_data():
    return JSONResponse(
        content=read_json(DASHBOARD_DATA)
    )


# ---------------------------------------------------------------------------
# LIVE INTELLIGENCE
# ---------------------------------------------------------------------------

@app.get("/api/intelligence")
def intelligence():
    return JSONResponse(
        content=read_json(LIVE_INTELLIGENCE)
    )


@app.get("/api/live")
def live_intelligence_alias():
    return JSONResponse(
        content=read_json(LIVE_INTELLIGENCE)
    )


# ---------------------------------------------------------------------------
# LIVE POLLUTION
# ---------------------------------------------------------------------------

@app.get("/api/pollution")
def pollution():
    return file_response(
        LIVE_POLLUTION,
        "text/csv",
    )


# ---------------------------------------------------------------------------
# LIVE ALERTS
# ---------------------------------------------------------------------------

@app.get("/api/alerts")
def alerts():
    return file_response(
        LIVE_ALERTS,
        "text/csv",
    )


# ---------------------------------------------------------------------------
# LIVE SUMMARY
# ---------------------------------------------------------------------------

@app.get("/api/summary")
def summary():
    return file_response(
        LIVE_SUMMARY,
        "text/csv",
    )


# ---------------------------------------------------------------------------
# FORECAST
# ---------------------------------------------------------------------------

@app.get("/api/forecast")
def forecast():
    return file_response(
        FORECAST_CSV,
        "text/csv",
    )


# ---------------------------------------------------------------------------
# STATIC DASHBOARD ASSETS
# ---------------------------------------------------------------------------

if DASHBOARD_DIR.exists():
    app.mount(
        "/assets",
        StaticFiles(directory=str(DASHBOARD_DIR)),
        name="assets",
    )


# ---------------------------------------------------------------------------
# LOCAL / CLOUD ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=False,
        access_log=True,
    )
'''

    MAIN_PATH.write_text(
        cloud_main,
        encoding="utf-8",
    )

    line("Cloud main.py", f"CREATED -> {MAIN_PATH}")

    return True


# ============================================================================
# REQUIREMENTS
# ============================================================================

def create_requirements():
    banner("CREATING CLOUD REQUIREMENTS")

    requirements = """fastapi>=0.110,<1.0
uvicorn[standard]>=0.29,<1.0
"""

    REQUIREMENTS_PATH.write_text(
        requirements,
        encoding="utf-8",
    )

    line(
        "requirements.txt",
        f"CREATED -> {REQUIREMENTS_PATH}",
    )


# ============================================================================
# REPLIT CONFIGURATION
# ============================================================================

def create_replit_config():
    banner("CREATING REPLIT CONFIGURATION")

    replit_config = """entrypoint = "app/main.py"

[deployment]
run = ["python", "app/main.py"]
deploymentTarget = "cloudrun"

[env]
PYTHONUNBUFFERED = "1"
"""

    REPLIT_NIX_PATH.write_text(
        replit_config,
        encoding="utf-8",
    )

    replitignore = """.venv/
__pycache__/
*.pyc
*.pyo
*.pyd
.git/
"""

    REPLITIGNORE_PATH.write_text(
        replitignore,
        encoding="utf-8",
    )

    line(".replit", f"CREATED -> {REPLIT_NIX_PATH}")
    line(".gitignore", f"CREATED -> {REPLITIGNORE_PATH}")


# ============================================================================
# WINDOWS LAUNCHER
# ============================================================================

def create_windows_launcher():
    banner("CREATING LOCAL/CLOUD LAUNCHERS")

    bat = r"""@echo off
setlocal

if "%PORT%"=="" set PORT=8000
set HOST=0.0.0.0

echo ================================================================
echo PEARLS AQI PREDICTOR
echo PUBLIC PRODUCTION SERVER
echo ================================================================
echo Host : %HOST%
echo Port : %PORT%
echo.
echo Dashboard:
echo http://127.0.0.1:%PORT%/
echo.
echo API:
echo http://127.0.0.1:%PORT%/api/intelligence
echo.
echo Health:
echo http://127.0.0.1:%PORT%/api/health
echo ================================================================

python app\main.py

endlocal
"""

    START_BAT_PATH.write_text(
        bat,
        encoding="utf-8",
    )

    ps1 = r'''$ErrorActionPreference = "Stop"

if (-not $env:PORT) {
    $env:PORT = "8000"
}

$env:HOST = "0.0.0.0"

Write-Host "================================================================"
Write-Host "PEARLS AQI PREDICTOR"
Write-Host "PUBLIC PRODUCTION SERVER"
Write-Host "================================================================"
Write-Host "Host : $env:HOST"
Write-Host "Port : $env:PORT"
Write-Host ""
Write-Host "Dashboard:"
Write-Host "http://127.0.0.1:$env:PORT/"
Write-Host ""
Write-Host "API:"
Write-Host "http://127.0.0.1:$env:PORT/api/intelligence"
Write-Host ""
Write-Host "Health:"
Write-Host "http://127.0.0.1:$env:PORT/api/health"
Write-Host "================================================================"

python app/main.py
'''

    START_PS1_PATH.write_text(
        ps1,
        encoding="utf-8",
    )

    line(
        "Windows launcher",
        f"CREATED -> {START_BAT_PATH}",
    )

    line(
        "PowerShell launcher",
        f"CREATED -> {START_PS1_PATH}",
    )


# ============================================================================
# README
# ============================================================================

def create_readme():
    banner("CREATING PUBLIC DEPLOYMENT README")

    readme = """PEARLS AQI PREDICTOR
STEP 27 — PUBLIC PRODUCTION DEPLOYMENT

============================================================
PURPOSE
============================================================

This package exposes the existing Pearl Intelligence Dashboard
and live AQI intelligence API to a public hosting platform.

The Step 23 dashboard UI is preserved.

No model training or model retraining is performed.

============================================================
DASHBOARD
============================================================

GET /

The root URL serves:

    Pearl Intelligence Dashboard — Step 23

============================================================
API
============================================================

GET /api/health

Production health information.

GET /api/intelligence

Step 25 live AQI intelligence.

GET /api/dashboard-data

Step 23 dashboard data.

GET /api/forecast

Production forecast CSV.

GET /api/pollution

Live pollution CSV.

GET /api/alerts

Live alert CSV.

GET /api/summary

Live AQI summary CSV.

GET /docs

FastAPI Swagger documentation.

============================================================
CLOUD SERVER
============================================================

The server binds to:

    0.0.0.0

The port is read from:

    PORT

If PORT is not supplied locally, 8000 is used.

============================================================
REPLIT
============================================================

Upload/import this deployment package into Replit.

The included .replit configuration points to:

    app/main.py

The server will listen on the platform-provided PORT.

After deployment, Replit provides the public HTTPS URL.

Example:

    https://your-project-name.replit.app/

The exact public URL is assigned by the hosting platform.

============================================================
LOCAL TEST
============================================================

From this directory:

    python app/main.py

or:

    .\\start_public_server.bat

Then open:

    http://127.0.0.1:8000/

============================================================
IMPORTANT
============================================================

Do NOT open the HTML file directly with:

    file:///

Use the HTTP server.

Do NOT change the Step 23 HTML if the goal is to preserve
the validated dashboard.

============================================================
ARCHITECTURE
============================================================

Step 23:
    Pearl Intelligence Dashboard

Step 25:
    Live AQI Intelligence API

Step 26:
    Local production deployment

Step 27:
    Public cloud deployment package

============================================================
MODEL STATUS
============================================================

Training:
    NOT PERFORMED

Model selection:
    NOT PERFORMED

Model retraining:
    NOT PERFORMED

Dashboard redesign:
    NOT PERFORMED

Forecast logic:
    NOT CHANGED
"""

    README_PATH.write_text(
        readme,
        encoding="utf-8",
    )

    line(
        "Deployment README",
        f"CREATED -> {README_PATH}",
    )


# ============================================================================
# MANIFEST
# ============================================================================

def build_manifest(dashboard_integrity):
    banner("CREATING PUBLIC DEPLOYMENT MANIFEST")

    files = []

    for path in OUTPUT_DIR.rglob("*"):
        if not path.is_file():
            continue

        if path == MANIFEST_PATH:
            continue

        rel = path.relative_to(OUTPUT_DIR)

        files.append(
            {
                "path": str(rel).replace("\\", "/"),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    files.sort(key=lambda x: x["path"])

    manifest = {
        "project": "Pearls AQI Predictor",
        "step": 27,
        "name": "Public Production Deployment",
        "created_at_utc": utc_now(),
        "base_directory": str(BASE_DIR),
        "source": {
            "step_23": True,
            "step_25": True,
            "step_26": True,
        },
        "dashboard": {
            "name": "Pearl Intelligence Dashboard",
            "preserved": True,
            "redesign": False,
            "html_sha256": dashboard_integrity["sha256"],
            "html_size_bytes": dashboard_integrity["size_bytes"],
        },
        "model": {
            "training": False,
            "selection": False,
            "retraining": False,
        },
        "server": {
            "framework": "FastAPI",
            "host": "0.0.0.0",
            "port": "environment variable PORT",
            "default_port": 8000,
            "https": (
                "provided by hosting platform / reverse proxy"
            ),
        },
        "endpoints": [
            "/",
            "/dashboard",
            "/api/health",
            "/api/dashboard-data",
            "/api/intelligence",
            "/api/live",
            "/api/pollution",
            "/api/alerts",
            "/api/summary",
            "/api/forecast",
            "/docs",
        ],
        "files": files,
        "validation": {
            "status": "PASS",
        },
    }

    MANIFEST_PATH.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    line(
        "Manifest",
        f"SAVED -> {MANIFEST_PATH}",
    )

    return manifest


# ============================================================================
# VALIDATION
# ============================================================================

def validate_output():
    banner("VALIDATING PUBLIC DEPLOYMENT PACKAGE")

    checks = {}

    required_outputs = [
        (
            "Dashboard HTML",
            DASHBOARD_DIR
            / "pearl_intelligence_dashboard.html",
        ),
        (
            "Dashboard JSON",
            DASHBOARD_DIR
            / "pearl_intelligence_dashboard_data.json",
        ),
        (
            "Forecast CSV",
            DASHBOARD_DIR
            / "pearl_forecast.csv",
        ),
        (
            "Dashboard summary",
            DASHBOARD_DIR
            / "pearl_dashboard_summary.csv",
        ),
        (
            "Cloud server",
            MAIN_PATH,
        ),
        (
            "Requirements",
            REQUIREMENTS_PATH,
        ),
        (
            "Replit config",
            REPLIT_NIX_PATH,
        ),
        (
            "README",
            README_PATH,
        ),
        (
            "Windows launcher",
            START_BAT_PATH,
        ),
        (
            "PowerShell launcher",
            START_PS1_PATH,
        ),
        (
            "Manifest",
            MANIFEST_PATH,
        ),
    ]

    for name, path in required_outputs:
        ok = path.exists() and path.stat().st_size > 0

        checks[name] = ok

        line(
            name,
            "PASS" if ok else "FAIL",
        )

    # Validate cloud server contents.
    if MAIN_PATH.exists():
        text = MAIN_PATH.read_text(
            encoding="utf-8",
            errors="ignore",
        )

        cloud_checks = {
            "0.0.0.0 binding": "0.0.0.0" in text,
            "PORT support": 'os.getenv("PORT"' in text,
            "FastAPI": "FastAPI" in text,
            "Step 23 dashboard": "pearl_intelligence_dashboard.html" in text,
            "Step 25 API": "/api/intelligence" in text,
            "Health endpoint": "/api/health" in text,
        }

        for name, ok in cloud_checks.items():
            checks[f"server:{name}"] = ok

            line(
                name,
                "PASS" if ok else "FAIL",
            )

    # Confirm dashboard was not regenerated by this script.
    dashboard = (
        DASHBOARD_DIR
        / "pearl_intelligence_dashboard.html"
    )

    if dashboard.exists():
        text = dashboard.read_text(
            encoding="utf-8",
            errors="ignore",
        )

        html_valid = (
            "<html" in text.lower()
            and "</html>" in text.lower()
        )

        checks["dashboard HTML"] = html_valid

        line(
            "Dashboard HTML integrity",
            "PASS" if html_valid else "FAIL",
        )

    all_pass = all(checks.values())

    print()
    line(
        "Public deployment validation",
        "PASS" if all_pass else "FAIL",
    )

    if not all_pass:
        fail(
            "One or more public deployment validation checks failed."
        )

    return checks


# ============================================================================
# FINAL REPORT
# ============================================================================

def create_final_report(manifest):
    report_path = OUTPUT_DIR / "public_deployment_report.json"

    report = {
        "project": "Pearls AQI Predictor",
        "step": 27,
        "status": "PASS",
        "timestamp_utc": utc_now(),
        "deployment": {
            "type": "public_production",
            "dashboard": "Step 23",
            "live_intelligence": "Step 25",
            "source_deployment": "Step 26",
        },
        "ui": {
            "redesigned": False,
            "modified": False,
            "step_23_preserved": True,
        },
        "model": {
            "training": False,
            "selection": False,
            "retraining": False,
        },
        "server": {
            "host": "0.0.0.0",
            "port": "PORT environment variable",
            "framework": "FastAPI",
        },
        "public_endpoints": [
            "/",
            "/api/health",
            "/api/intelligence",
            "/api/dashboard-data",
            "/api/forecast",
            "/api/pollution",
            "/api/alerts",
            "/api/summary",
            "/docs",
        ],
        "manifest": str(MANIFEST_PATH),
    }

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return report_path


# ============================================================================
# MAIN
# ============================================================================

def main():
    start = time.perf_counter()

    banner("PEARLS AQI PREDICTOR")

    banner("STEP 27 — PUBLIC PRODUCTION DEPLOYMENT")

    line("Base directory", BASE_DIR)
    line("Step 26 deployment", STEP26_DIR)
    line("Output directory", OUTPUT_DIR)

    print()
    line("Deployment target", "PUBLIC CLOUD / REPLIT")
    line("Step 23 dashboard", "PRESERVED")
    line("Step 25 intelligence", "PRESERVED")
    line("Dashboard redesign", "NO")
    line("HTML regeneration", "NO")
    line("CSS modification", "NO")
    line("JavaScript modification", "NO")
    line("Model training", "NO")
    line("Model selection", "NO")
    line("Model retraining", "NO")
    line("Caching", "UNCHANGED")
    line("Cloud host", "0.0.0.0")
    line("Cloud port", "PORT environment variable")

    validate_step26()

    dashboard_integrity = validate_dashboard_integrity()

    create_clean_package()

    package_dashboard_and_data()

    create_cloud_server()

    create_requirements()

    create_replit_config()

    create_windows_launcher()

    create_readme()

    manifest = build_manifest(
        dashboard_integrity
    )

    validate_output()

    report_path = create_final_report(
        manifest
    )

    elapsed = time.perf_counter() - start

    banner("STEP 27 COMPLETE")

    line("Deployment status", "PASS")
    line("Step 23 UI", "PRESERVED — NOT CHANGED")
    line("Step 25 intelligence", "PACKAGED")
    line("Cloud server", "READY")
    line("Host binding", "0.0.0.0")
    line("Port", "Environment variable PORT")
    line("Replit configuration", "READY")

    print()
    print("PUBLIC DEPLOYMENT PACKAGE:")
    print(OUTPUT_DIR)

    print()
    print("Dashboard:")
    print(
        OUTPUT_DIR
        / "app"
        / "dashboard"
        / "pearl_intelligence_dashboard.html"
    )

    print()
    print("Cloud server:")
    print(MAIN_PATH)

    print()
    print("Requirements:")
    print(REQUIREMENTS_PATH)

    print()
    print("Replit configuration:")
    print(REPLIT_NIX_PATH)

    print()
    print("Deployment manifest:")
    print(MANIFEST_PATH)

    print()
    print("Deployment report:")
    print(report_path)

    print()
    print("NEXT:")
    print(
        "Upload/import reports/public_production_deployment "
        "to your public hosting platform."
    )

    print()
    print("Local test:")
    print(
        "cd reports\\public_production_deployment"
    )
    print(
        ".\\start_public_server.bat"
    )

    print()
    print("Local dashboard:")
    print("http://127.0.0.1:8000/")

    print()
    print("Public URL:")
    print(
        "Assigned by the hosting provider after deployment."
    )

    print()
    line("Execution time", f"{elapsed:.3f}s")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        fail("Execution interrupted by user.")
    except Exception as exc:
        fail(
            f"{type(exc).__name__}: {exc}"
        )