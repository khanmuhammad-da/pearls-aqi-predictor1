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
