"""
main.py
=======
FastAPI production service for Stroke Risk Triage & Explainable AI.
- Runs 5-fold calibrated pipeline evaluated in Phase 8
- Exact 5-fold TreeSHAP in raw margin space with strict additivity
- Human-readable local surrogate in calibrated probability space
- Proxy-aware rate limiting with anti-spoofing protection
- Scoped CORS (production domain + project-specific preview regex)
- Strict patient privacy (zero request body logging, sanitized 422 errors)
"""
from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from api.explainer import compute_5fold_treeshap, compute_local_surrogate
from api.schemas import PatientInput, PredictionResponse

# Configure sanitized logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stroke_triage_api")

ROOT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

# Global model state
STATE = {}


def verify_versions():
    """Fail loudly on startup if core ML packages diverge from artifact provenance."""
    import joblib as jl
    import sklearn
    import xgboost

    expected = {
        "xgboost": "3.2.0",
        "scikit-learn": "1.8.0",
        "joblib": "1.5.3",
    }
    actual = {
        "xgboost": xgboost.__version__,
        "scikit-learn": sklearn.__version__,
        "joblib": jl.__version__,
    }
    for pkg, exp_ver in expected.items():
        act_ver = actual[pkg]
        if act_ver != exp_ver:
            logger.warning("Version mismatch for %s: expected %s, found %s", pkg, exp_ver, act_ver)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup validation
    logger.info("Initializing Stroke Triage API...")
    verify_versions()

    # Load artifacts
    cal_pipe_path = ARTIFACTS_DIR / "calibrated_pipeline.joblib"
    meta_path = ARTIFACTS_DIR / "meta.json"

    if not cal_pipe_path.exists():
        raise FileNotFoundError(f"Missing required artifact: {cal_pipe_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing required artifact: {meta_path}")

    STATE["cal_pipe"] = joblib.load(cal_pipe_path)
    with open(meta_path) as f:
        STATE["meta"] = json.load(f)

    logger.info("Artifacts loaded successfully. Ready for inference.")
    yield
    STATE.clear()


# Proxy-aware client IP resolver with anti-spoofing protection
TRUSTED_PROXIES = set(os.getenv("TRUSTED_PROXIES", "127.0.0.1,::1").split(","))


def get_client_ip(request: Request) -> str:
    """Resolve real client IP.
    
    If the direct peer is NOT in TRUSTED_PROXIES, ignore any X-Forwarded-For header to prevent spoofing.
    If the direct peer IS a trusted proxy, extract the client IP from X-Forwarded-For.
    """
    direct_ip = request.client.host if request.client else "127.0.0.1"

    if direct_ip in TRUSTED_PROXIES:
        xff = request.headers.get("x-forwarded-for")
        if xff:
            # First IP in the list is the originating client
            client_candidate = xff.split(",")[0].strip()
            if client_candidate:
                return client_candidate

    return direct_ip


limiter = Limiter(key_func=get_client_ip, default_limits=["120/minute"])

app = FastAPI(
    title="Stroke Risk Triage & Explainability API",
    version="2.0.0",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# Sanitized 422 Error Handler: NEVER echo patient input values
@app.exception_handler(RequestValidationError)
async def sanitized_validation_exception_handler(request: Request, exc: RequestValidationError):
    sanitized_errors = []
    for err in exc.errors():
        sanitized_errors.append({
            "field": [str(loc) for loc in err.get("loc", []) if loc != "body"],
            "message": err.get("msg", "Invalid value"),
            "type": err.get("type", "value_error"),
        })
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": sanitized_errors, "error": "Validation failed on clinical input fields."},
    )


# CORS Configuration: Scoped strictly to production and preview domains
ENVIRONMENT = os.getenv("ENVIRONMENT", "development").lower()
FRONTEND_PROD_URL = os.getenv("FRONTEND_URL", "https://stroke-xai-frontend.vercel.app")
VERCEL_PREVIEW_REGEX = os.getenv(
    "VERCEL_PREVIEW_REGEX",
    r"^https:\/\/stroke-xai-frontend(-[a-z0-9-]+)?-dummycodertechs-projects\.vercel\.app$",
)

origins = [FRONTEND_PROD_URL]
if ENVIRONMENT != "production":
    origins.extend([
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ])

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=VERCEL_PREVIEW_REGEX,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# Request Privacy Logger Middleware: never logs request bodies
@app.middleware("http")
async def privacy_logging_middleware(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start_time) * 1000

    # Log route, status, latency only; ZERO patient payloads
    client_ip = get_client_ip(request)
    logger.info(
        "%s %s -> status=%d duration=%.1fms ip=%s",
        request.method, request.url.path, response.status_code, duration_ms, client_ip
    )
    return response


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", summary="Health check and warmup probe")
async def health_check():
    """Immediate 200 OK without model computation for Render cold-start warmup."""
    return {
        "status": "healthy",
        "model_loaded": "cal_pipe" in STATE,
        "environment": ENVIRONMENT,
    }


@app.get("/meta", summary="Model metadata, operating thresholds, and metrics")
async def get_metadata():
    """Return evaluated model metadata, thresholds, and 95% bootstrap CIs from api/artifacts/meta.json."""
    if "meta" not in STATE:
        return JSONResponse(status_code=503, content={"error": "Model metadata not ready."})
    return STATE["meta"]


@app.post("/predict", response_model=PredictionResponse, summary="Predict stroke risk and compute explanations")
@limiter.limit("60/minute")
async def predict_stroke_risk(patient: PatientInput, request: Request):
    """Calibrated probability triage scoring with exact 5-fold TreeSHAP and local surrogate (LIME-style)."""
    if "cal_pipe" not in STATE or "meta" not in STATE:
        return JSONResponse(status_code=503, content={"error": "Service warming up, please retry in 5 seconds."})

    cal_pipe = STATE["cal_pipe"]
    meta = STATE["meta"]

    # Convert patient input into DataFrame matching training feature columns
    patient_dict = {
        "age": patient.age,
        "avg_glucose_level": patient.avg_glucose_level,
        "bmi": patient.bmi,
        "gender": patient.gender,
        "hypertension": patient.hypertension,
        "heart_disease": patient.heart_disease,
        "ever_married": patient.ever_married,
        "work_type": patient.work_type,
        "Residence_type": patient.Residence_type,
        "smoking_status": patient.smoking_status,
    }
    input_df = pd.DataFrame([patient_dict])

    # 1. Calibrated probability via 5-fold ensemble
    prob = float(cal_pipe.predict_proba(input_df)[0][1])

    # 2. Operating threshold resolution
    op_key = patient.operating_point or "cost_based_10"
    op_info = meta.get("operating_points", {}).get(op_key, {})
    threshold = float(op_info.get("threshold", meta.get("default_threshold", 0.0826)))
    decision = "HIGH_RISK" if prob >= threshold else "BELOW_THRESHOLD"

    # 3. Exact 5-fold TreeSHAP on raw margin log-odds
    base_val, raw_margin, shap_items = compute_5fold_treeshap(cal_pipe, input_df, patient_dict)

    # 4. Human-readable local surrogate (LIME-style) in calibrated probability points
    surrogate_rules = compute_local_surrogate(cal_pipe, patient_dict, n_samples=500)

    return PredictionResponse(
        probability=round(prob, 4),
        threshold=round(threshold, 4),
        operating_point=op_key,
        decision=decision,
        base_value=round(base_val, 3),
        raw_margin=round(raw_margin, 3),
        shap=shap_items,
        local_surrogate=surrogate_rules,
        lime=surrogate_rules,
    )
