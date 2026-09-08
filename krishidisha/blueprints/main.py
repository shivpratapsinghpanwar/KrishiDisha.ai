"""Public pages: landing, weather, mandi prices, schemes, crop calendar, fertilizer calculator."""
from __future__ import annotations

from flask import Blueprint, current_app, jsonify, render_template, request

from ..services import market, weather
from ..services.knowledge import FERTILIZER_REQUIREMENTS, fertilizer_calculator
from ..utils import current_farmer, log_activity

bp = Blueprint("main", __name__)


@bp.route("/")
def index():
    from ..models import Product

    featured = Product.query.filter_by(is_active=True).order_by(Product.rating.desc()).limit(4).all()
    return render_template("index.html", featured=featured)


@bp.route("/about")
def about():
    return render_template("about.html", ml_status=current_app.ml.status(), detector=current_app.detector.info())


@bp.route("/health")
def health():
    return jsonify(status="ok", assistant=current_app.assistant.describe(), models=current_app.ml.status())


@bp.route("/weather")
def weather_page():
    place = request.args.get("place", "").strip()
    farmer = current_farmer()
    if not place and farmer and (farmer.district or farmer.state):
        place = farmer.district or farmer.state
    forecast = weather.forecast_for_place(place, 7) if place else None
    if forecast and "error" not in forecast:
        log_activity("Weather Advisory", {"place": place}, {"advisories": [a["title"] for a in forecast["advisories"]]})
    return render_template("weather.html", place=place, forecast=forecast)


@bp.route("/mandi")
def mandi_page():
    commodity = request.args.get("commodity", "").strip()
    state = request.args.get("state", "").strip()
    district = request.args.get("district", "").strip()
    result = None
    if commodity:
        result = market.mandi_prices(current_app.config["DATA_GOV_API_KEY"], current_app.config["DATA_DIR"],
                                     commodity, state or None, district or None, limit=30)
    msp = market.msp_reference(current_app.config["DATA_DIR"])
    return render_template("mandi.html", commodity=commodity, state=state, district=district, result=result, msp=msp)


@bp.route("/schemes")
def schemes_page():
    q = request.args.get("q", "").strip()
    kb = current_app.kb
    schemes = kb.scheme_lookup(q) if q else kb.schemes
    return render_template("schemes.html", schemes=schemes, q=q)


@bp.route("/crop-calendar")
def calendar_page():
    kb = current_app.kb
    crop = request.args.get("crop", "").strip()
    season = request.args.get("season", "").strip()
    rows = kb.calendar_for(crop or None, season or None)
    crops = sorted({r["crop"] for r in kb.crop_calendar})
    return render_template("crop_calendar.html", rows=rows, crops=crops, crop=crop, season=season)


@bp.route("/crop-guide/<crop>")
def crop_guide_page(crop):
    kb = current_app.kb
    guide = kb.crop_guide(crop)
    from ..blueprints.marketplace import search_products

    products = search_products(crop=crop, limit=4) if guide else []
    return render_template("crop_guide.html", crop=crop, guide=guide, products=products,
                           all_crops=sorted(kb.crop_guides), calendar=kb.calendar_for(crop))


@bp.route("/tools/fertilizer-calculator", methods=["GET", "POST"])
def fert_calc():
    result = None
    form = {"crop": "", "area": "", "unit": "acre", "soil_n": "", "soil_p": "", "soil_k": ""}
    if request.method == "POST":
        form.update({k: request.form.get(k, "") for k in form})
        try:
            def opt(v):
                return float(v) if v not in ("", None) else None
            result = fertilizer_calculator(form["crop"], float(form["area"]), form["unit"], opt(form["soil_n"]),
                                           opt(form["soil_p"]), opt(form["soil_k"]))
            if "error" not in result:
                log_activity("Fertilizer Calculator", form, result["bags_50kg"])
        except ValueError as exc:
            result = {"error": f"Invalid input: {exc}"}
    return render_template("fertilizer_calculator.html", result=result, form=form, crops=sorted(FERTILIZER_REQUIREMENTS))
