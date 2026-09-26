"""Versioned JSON API (``/api/v1``) for mobile apps, SMS/WhatsApp gateways and integrations.

Every endpoint mirrors a web page but speaks JSON only. Authentication is the
same cookie session as the website; endpoints that do not need a farmer work
anonymously.
"""
from __future__ import annotations

import io

from flask import Blueprint, current_app, jsonify, request
from PIL import Image, UnidentifiedImageError

from ..services import market, weather
from ..services.knowledge import FERTILIZER_REQUIREMENTS, fertilizer_calculator
from ..services.llm import strip_markdown
from ..services.ml import FERT_CROP_TYPES, SOIL_TYPES
from ..utils import current_farmer, log_activity

bp = Blueprint("api", __name__, url_prefix="/api/v1")


def _payload() -> dict:
    if request.is_json:
        return request.get_json(silent=True) or {}
    return {k: v for k, v in request.form.items()} or {k: v for k, v in request.args.items()}


def _floats(data: dict, keys: tuple[str, ...]) -> dict[str, float]:
    out = {}
    missing = [k for k in keys if data.get(k) in (None, "")]
    if missing:
        raise ValueError(f"missing fields: {', '.join(missing)}")
    for k in keys:
        try:
            out[k] = float(data[k])
        except (TypeError, ValueError):
            raise ValueError(f"{k} must be a number")
    return out


@bp.errorhandler(ValueError)
def bad_request(exc):
    return jsonify(error=str(exc)), 400


@bp.route("/health")
def health():
    return jsonify(status="ok", assistant=current_app.assistant.describe(), models=current_app.ml.status(),
                   disease_model=current_app.detector.info())


@bp.route("/reference")
def reference():
    meta = current_app.ml.yield_meta
    return jsonify(soil_types=SOIL_TYPES, fertilizer_crop_types=FERT_CROP_TYPES, yield_crops=meta["crops"],
                   yield_states=meta["states"], yield_seasons=meta["seasons"],
                   fertilizer_calculator_crops=sorted(FERTILIZER_REQUIREMENTS))


# ------------------------------------------------------------------ ML models
@bp.route("/crop/recommend", methods=["POST"])
def crop_recommend():
    data = _payload()
    inputs = _floats(data, ("N", "P", "K", "temperature", "humidity", "ph", "rainfall"))
    result = current_app.ml.recommend_crop(**inputs)
    result["guide"] = current_app.kb.crop_guide(result["recommended_crop"])
    log_activity("Crop Recommendation", inputs, {"crop": result["recommended_crop"], "confidence": result["confidence"]})
    return jsonify(result)


@bp.route("/fertilizer/recommend", methods=["POST"])
def fertilizer_recommend():
    data = _payload()
    nums = _floats(data, ("temperature", "humidity", "moisture", "N", "P", "K"))
    soil, crop = data.get("soil_type", ""), data.get("crop_type", "")
    if soil not in SOIL_TYPES:
        raise ValueError(f"soil_type must be one of {SOIL_TYPES}")
    if crop not in FERT_CROP_TYPES:
        raise ValueError(f"crop_type must be one of {FERT_CROP_TYPES}")
    result = current_app.ml.recommend_fertilizer(
        soil_type=soil, crop_type=crop, **nums,
        area=float(data["area"]) if data.get("area") else None, unit=data.get("unit", "acre"))
    log_activity("Fertilizer Recommendation", {**nums, "soil_type": soil, "crop_type": crop},
                 {"fertilizer": result["recommended_fertilizer"], "confidence": result["confidence"]})
    return jsonify(result)


@bp.route("/fertilizer/calculator", methods=["GET", "POST"])
def fert_calc():
    data = _payload()
    if not data.get("crop") or not data.get("area"):
        raise ValueError("crop and area are required")

    def opt(k):
        return float(data[k]) if data.get(k) not in (None, "") else None
    result = fertilizer_calculator(data["crop"], float(data["area"]), data.get("unit", "acre"), opt("soil_n"),
                                   opt("soil_p"), opt("soil_k"))
    return jsonify(result), (400 if "error" in result else 200)


@bp.route("/yield/predict", methods=["POST"])
def yield_predict():
    data = _payload()
    nums = _floats(data, ("area", "annual_rainfall"))
    # fertilizer/pesticide are optional: omitted values fall back to regional medians.
    # "production" is accepted and ignored - it used to leak the target.
    for k in ("fertilizer", "pesticide"):
        nums[k] = float(data[k]) if data.get(k) not in (None, "") else None
    for k in ("crop", "season", "state", "crop_year"):
        if not data.get(k):
            raise ValueError(f"{k} is required")
    result = current_app.ml.predict_yield(crop=data["crop"], crop_year=int(data["crop_year"]), season=data["season"],
                                          state=data["state"], **nums)
    log_activity("Crop Yield Prediction", {**nums, "crop": data["crop"], "state": data["state"]},
                 {"predicted_yield": result["predicted_yield"]})
    return jsonify(result)


@bp.route("/disease/detect", methods=["POST"])
def disease_detect():
    file = request.files.get("image")
    if not file or not file.filename:
        raise ValueError("upload an image file in the 'image' field")
    try:
        img = Image.open(io.BytesIO(file.read()))
        img.load()
    except (UnidentifiedImageError, OSError):
        raise ValueError("not a valid image")
    result = current_app.detector.predict(img)
    if not result.get("available"):
        return jsonify(result), 503
    top = result["top"]
    is_plant = result.get("is_plant", True)
    result["info"] = current_app.kb.disease_by_label(top["label"]) if is_plant else None
    from .marketplace import search_products

    result["products"] = [dict(p.to_dict(), url=f"/marketplace/product/{p.slug}")
                          for p in search_products(disease=top["label"], crop=top["crop"], limit=4)] \
        if is_plant and not top["is_healthy"] else []
    log_activity("Crop Disease Detection", {"image": file.filename}, {"disease": top["name"], "confidence": top["confidence"]})
    return jsonify(result)


# ------------------------------------------------------------------- assistant
@bp.route("/chat", methods=["POST"])
def chat():
    data = _payload()
    message = (data.get("message") or "").strip()
    if not message:
        raise ValueError("message is required")
    history = data.get("history") if request.is_json else None
    farmer = current_farmer()
    result = current_app.assistant.chat(message, history=history if isinstance(history, list) else None,
                                        farmer_context=farmer.profile_summary() if farmer else None,
                                        language=(data.get("language") or "en")[:10])
    out = {"reply": result["reply"], "provider": result.get("provider"), "model": result.get("model"),
           "tools_used": result.get("tools_used", []), "sources": result.get("sources", [])}
    if data.get("plain") in ("1", "true", True):
        out["reply_plain"] = strip_markdown(result["reply"])
    return jsonify(out)


# ------------------------------------------------------------------- open data
@bp.route("/weather")
def weather_api():
    place = request.args.get("place", "").strip()
    if not place:
        raise ValueError("place is required")
    fc = weather.forecast_for_place(place, request.args.get("days", 7, type=int))
    return jsonify(fc), (404 if "error" in fc else 200)


@bp.route("/mandi")
def mandi_api():
    commodity = request.args.get("commodity", "").strip()
    if not commodity:
        raise ValueError("commodity is required")
    return jsonify(market.mandi_prices(current_app.config["DATA_GOV_API_KEY"], current_app.config["DATA_DIR"], commodity,
                                       request.args.get("state") or None, request.args.get("district") or None,
                                       limit=request.args.get("limit", 20, type=int)))


@bp.route("/msp")
def msp_api():
    return jsonify(prices=market.msp_reference(current_app.config["DATA_DIR"], request.args.get("commodity")))


@bp.route("/schemes")
def schemes_api():
    return jsonify(schemes=current_app.kb.scheme_lookup(request.args.get("q")))


@bp.route("/crop-guide/<crop>")
def crop_guide_api(crop):
    guide = current_app.kb.crop_guide(crop)
    if not guide:
        return jsonify(error="no guide", available=sorted(current_app.kb.crop_guides)), 404
    return jsonify(crop=crop.lower(), guide=guide, calendar=current_app.kb.calendar_for(crop))


@bp.route("/crop-calendar")
def calendar_api():
    return jsonify(rows=current_app.kb.calendar_for(request.args.get("crop"), request.args.get("season"),
                                                    request.args.get("month")))


@bp.route("/diseases")
def diseases_api():
    return jsonify(diseases=current_app.kb.all_diseases())


@bp.route("/knowledge/search")
def knowledge_search():
    q = request.args.get("q", "").strip()
    if not q:
        raise ValueError("q is required")
    return jsonify(results=current_app.kb.search(q, k=request.args.get("k", 5, type=int)))


@bp.route("/products")
def products_api():
    from .marketplace import search_products

    rows = search_products(request.args.get("q"), request.args.get("category"), request.args.get("disease"),
                           request.args.get("crop"), limit=request.args.get("limit", 12, type=int))
    return jsonify(products=[dict(p.to_dict(), url=f"/marketplace/product/{p.slug}") for p in rows])
