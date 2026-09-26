"""KrishiDisha standalone FastAPI service.

A thin, documented (Swagger at /docs) HTTP layer over the same service objects
the Flask site uses: ML models, knowledge base, disease detector, assistant,
weather, mandi prices and the marketplace catalogue. Useful for mobile apps,
IVR / SMS / WhatsApp gateways, or when you only want the API without the UI.

    uvicorn api.main:app --reload --port 8000

The Flask app is created once at import time purely to reuse its configuration,
database and service singletons (no Flask request handling happens here).
"""
from __future__ import annotations

import io
import logging
from typing import Any, Optional

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

from api.models import (ChatRequest, ChatResponse, CropRecommendationInput, CropRecommendationOutput, CropYieldInput,
                        CropYieldOutput, DiseaseDetectionOutput, FertilizerCalculatorInput,
                        FertilizerRecommendationInput, FertilizerRecommendationOutput, HealthCheck)
from krishidisha import create_app
from krishidisha.services import market, weather
from krishidisha.services.knowledge import FERTILIZER_REQUIREMENTS, fertilizer_calculator
from krishidisha.services.llm import strip_markdown
from krishidisha.services.ml import FERT_CROP_TYPES, SOIL_TYPES

log = logging.getLogger("krishidisha.api")

flask_app = create_app()
ml, kb, detector, assistant = flask_app.ml, flask_app.kb, flask_app.detector, flask_app.assistant
CFG = flask_app.config

app = FastAPI(
    title="KrishiDisha Agriculture API",
    description="Crop / fertilizer / yield models, leaf disease detection, agricultural assistant, weather, mandi "
                "prices, schemes and the input marketplace - the same engine behind the KrishiDisha website.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.exception_handler(ValueError)
async def value_error_handler(_: Request, exc: ValueError):
    return JSONResponse(status_code=400, content={"error": str(exc)})


@app.exception_handler(Exception)
async def generic_error_handler(_: Request, exc: Exception):
    log.exception("unhandled error")
    return JSONResponse(status_code=500, content={"error": "internal error", "detail": str(exc)[:200]})


def _products(**kwargs) -> list[dict[str, Any]]:
    from krishidisha.blueprints.marketplace import search_products

    with flask_app.app_context():
        return [dict(p.to_dict(), url=f"/marketplace/product/{p.slug}") for p in search_products(**kwargs)]


# --------------------------------------------------------------------- meta
@app.get("/", tags=["Meta"])
def root():
    return {"name": "KrishiDisha API", "version": app.version, "docs": "/docs",
            "endpoints": sorted({r.path for r in app.routes if r.path.startswith("/") and "{" not in r.path})}


@app.get("/health", response_model=HealthCheck, tags=["Meta"])
def health():
    return {"status": "ok", "assistant": assistant.describe(), "models": ml.status(), "disease_model": detector.info()}


@app.get("/reference", tags=["Meta"])
def reference():
    meta = ml.yield_meta
    return {"soil_types": SOIL_TYPES, "fertilizer_crop_types": FERT_CROP_TYPES, "yield_crops": meta["crops"],
            "yield_states": meta["states"], "yield_seasons": meta["seasons"],
            "fertilizer_calculator_crops": sorted(FERTILIZER_REQUIREMENTS)}


# ------------------------------------------------------------------- models
@app.post("/crop/recommend", response_model=CropRecommendationOutput, tags=["Models"])
def crop_recommend(body: CropRecommendationInput):
    result = ml.recommend_crop(**body.model_dump())
    result["guide"] = kb.crop_guide(result["recommended_crop"])
    return result


@app.post("/fertilizer/recommend", response_model=FertilizerRecommendationOutput, tags=["Models"])
def fertilizer_recommend(body: FertilizerRecommendationInput):
    if body.soil_type not in SOIL_TYPES:
        raise HTTPException(400, f"soil_type must be one of {SOIL_TYPES}")
    if body.crop_type not in FERT_CROP_TYPES:
        raise HTTPException(400, f"crop_type must be one of {FERT_CROP_TYPES}")
    return ml.recommend_fertilizer(temperature=body.temperature, humidity=body.humidity, moisture=body.moisture,
                                   soil_type=body.soil_type, crop_type=body.crop_type, N=body.N, K=body.K, P=body.P,
                                   area=body.area, unit=body.unit)


@app.post("/fertilizer/calculator", tags=["Models"])
def fert_calc(body: FertilizerCalculatorInput):
    result = fertilizer_calculator(body.crop, body.area, body.unit, body.soil_n, body.soil_p, body.soil_k)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


@app.post("/yield/predict", response_model=CropYieldOutput, tags=["Models"])
def yield_predict(body: CropYieldInput):
    return ml.predict_yield(**body.model_dump())


@app.post("/disease/detect", response_model=DiseaseDetectionOutput, tags=["Models"])
async def disease_detect(image: UploadFile = File(...)):
    raw = await image.read()
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "not a valid image")
    result = detector.predict(img)
    if not result.get("available"):
        raise HTTPException(503, result.get("message", "disease model unavailable"))
    top = result["top"]
    result["info"] = kb.disease_by_label(top["label"])
    result["products"] = _products(disease=top["label"], crop=top["crop"], limit=4)
    return result


# ---------------------------------------------------------------- assistant
@app.post("/chat", response_model=ChatResponse, tags=["Assistant"])
def chat(body: ChatRequest):
    result = assistant.chat(body.message, history=body.history, farmer_context=body.farmer_context,
                            language=body.language)
    out = {"reply": result["reply"], "provider": result.get("provider"), "model": result.get("model"),
           "tools_used": result.get("tools_used", []), "sources": result.get("sources", [])}
    if body.plain:
        out["reply_plain"] = strip_markdown(result["reply"])
    return out


@app.post("/chat/image", response_model=ChatResponse, tags=["Assistant"])
async def chat_image(image: UploadFile = File(...), message: str = "", language: str = "en"):
    raw = await image.read()
    mime = image.content_type if image.content_type in ("image/png", "image/jpeg", "image/webp") else "image/jpeg"
    result = assistant.chat(message or "Please analyse this leaf photo and tell me what to do.", language=language,
                            image_bytes=raw, image_mime=mime)
    return {"reply": result["reply"], "provider": result.get("provider"), "model": result.get("model"),
            "tools_used": result.get("tools_used", []), "sources": result.get("sources", [])}


# ---------------------------------------------------------------- open data
@app.get("/weather", tags=["Data"])
def weather_api(place: str = Query(..., min_length=2), days: int = Query(7, ge=1, le=14)):
    fc = weather.forecast_for_place(place, days)
    if "error" in fc:
        raise HTTPException(404, fc["error"])
    return fc


@app.get("/mandi", tags=["Data"])
def mandi_api(commodity: str = Query(..., min_length=2), state: Optional[str] = None, district: Optional[str] = None,
              limit: int = Query(20, ge=1, le=100)):
    return market.mandi_prices(CFG["DATA_GOV_API_KEY"], CFG["DATA_DIR"], commodity, state, district, limit=limit)


@app.get("/msp", tags=["Data"])
def msp_api(commodity: Optional[str] = None):
    return {"prices": market.msp_reference(CFG["DATA_DIR"], commodity)}


@app.get("/schemes", tags=["Data"])
def schemes_api(q: Optional[str] = None):
    return {"schemes": kb.scheme_lookup(q)}


@app.get("/crop-guide/{crop}", tags=["Data"])
def crop_guide_api(crop: str):
    guide = kb.crop_guide(crop)
    if not guide:
        raise HTTPException(404, f"no guide for {crop}; available: {sorted(kb.crop_guides)}")
    return {"crop": crop.lower(), "guide": guide, "calendar": kb.calendar_for(crop)}


@app.get("/crop-calendar", tags=["Data"])
def calendar_api(crop: Optional[str] = None, season: Optional[str] = None, month: Optional[str] = None):
    return {"rows": kb.calendar_for(crop, season, month)}


@app.get("/diseases", tags=["Data"])
def diseases_api():
    return {"diseases": kb.all_diseases()}


@app.get("/knowledge/search", tags=["Data"])
def knowledge_search(q: str = Query(..., min_length=2), k: int = Query(5, ge=1, le=20)):
    return {"results": kb.search(q, k=k)}


@app.get("/products", tags=["Marketplace"])
def products_api(q: Optional[str] = None, category: Optional[str] = None, disease: Optional[str] = None,
                 crop: Optional[str] = None, limit: int = Query(12, ge=1, le=60)):
    return {"products": _products(query=q, category=category, disease=disease, crop=crop, limit=limit)}
