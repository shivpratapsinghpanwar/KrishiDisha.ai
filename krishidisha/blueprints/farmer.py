"""Farmer portal: dashboard, crop / fertilizer / yield / disease tools, PDF reports."""
from __future__ import annotations

import json
import os
import uuid
from collections import Counter

import pandas as pd
from flask import (Blueprint, abort, current_app, flash, redirect, render_template, request, send_file, session,
                   url_for)
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename

from ..extensions import db
from ..models import FarmerActivity, Order
from ..services import reports
from ..services.knowledge import fertilizer_calculator
from ..services.ml import FERT_CROP_TYPES, SOIL_TYPES
from ..utils import current_farmer, farmer_required, form_float, log_activity

bp = Blueprint("farmer", __name__)

ALLOWED_IMAGE = {"png", "jpg", "jpeg", "webp"}


@bp.route("/home_crop")
@farmer_required
def home():
    farmer = current_farmer()
    recent = FarmerActivity.query.filter_by(farmer_id=farmer.id, deleted=False) \
        .order_by(FarmerActivity.timestamp.desc()).limit(5).all()
    orders = Order.query.filter_by(farmer_id=farmer.id).order_by(Order.created_at.desc()).limit(3).all()
    place = farmer.district or farmer.state
    forecast = None
    if place:
        from ..services import weather

        try:
            fc = weather.forecast_for_place(place, 3)
            forecast = fc if "error" not in fc else None
        except Exception:  # noqa: BLE001
            forecast = None
    return render_template("home_crop.html", farmer=farmer, recent=recent, orders=orders, forecast=forecast,
                           place=place)


# ------------------------------------------------------------ crop rec
@bp.route("/crop_recommendation", methods=["GET", "POST"])
@farmer_required
def crop_recommendation():
    result = None
    inputs = session.get("crop_inputs") or {}
    if request.method == "POST":
        try:
            inputs = {k: form_float(k) for k in ("N", "P", "K", "temperature", "humidity", "ph", "rainfall")}
            result = current_app.ml.recommend_crop(**inputs)
            result["guide"] = current_app.kb.crop_guide(result["recommended_crop"])
            session["crop_inputs"] = inputs
            session["crop_result"] = {k: v for k, v in result.items() if k != "guide"}
            log_activity("Crop Recommendation", inputs, {"crop": result["recommended_crop"],
                                                          "confidence": result["confidence"],
                                                          "profit_per_acre": result.get("profit_per_acre")})
        except ValueError as exc:
            flash(f"Invalid input: {exc}", "danger")
    return render_template("crop_recommendation.html", result=result, inputs=inputs)


@bp.route("/download_report", methods=["GET", "POST"])
@farmer_required
def download_crop_report():
    inputs, result = session.get("crop_inputs"), session.get("crop_result")
    if not inputs or not result:
        flash("Run a crop recommendation first.", "warning")
        return redirect(url_for("farmer.crop_recommendation"))
    result = dict(result, guide=current_app.kb.crop_guide(result["recommended_crop"]))
    buf = reports.crop_report(inputs, result, current_app.static_folder, current_farmer().name)
    return send_file(buf, as_attachment=True, download_name="krishidisha_crop_report.pdf", mimetype="application/pdf")


# ------------------------------------------------------- fertilizer rec
@bp.route("/fertilizer_recommendation", methods=["GET", "POST"])
@farmer_required
def fertilizer_recommendation():
    result = None
    inputs = session.get("fert_inputs") or {}
    if request.method == "POST":
        try:
            inputs = {
                "temperature": form_float("temperature"), "humidity": form_float("humidity"),
                "moisture": form_float("moisture"), "soil_type": request.form["soil_type"],
                "crop_type": request.form["crop_type"], "N": form_float("N"), "K": form_float("K"), "P": form_float("P"),
            }
            if inputs["soil_type"] not in SOIL_TYPES or inputs["crop_type"] not in FERT_CROP_TYPES:
                raise ValueError("unknown soil or crop type")
            result = current_app.ml.recommend_fertilizer(**inputs)
            area = request.form.get("area")
            if area:
                crop_key = {"Paddy": "rice", "Ground Nuts": "groundnut", "Oil seeds": "mustard", "Pulses": "chickpea"} \
                    .get(inputs["crop_type"], inputs["crop_type"].lower())
                result["calculator"] = fertilizer_calculator(crop_key, float(area), request.form.get("unit", "acre"))
            session["fert_inputs"] = inputs
            session["fert_result"] = result
            log_activity("Fertilizer Recommendation", inputs, {"fertilizer": result["recommended_fertilizer"],
                                                                "confidence": result["confidence"]})
        except (ValueError, KeyError) as exc:
            flash(f"Invalid input: {exc}", "danger")
    from .marketplace import search_products

    products = search_products(query=result["recommended_fertilizer"], category="Fertilizer", limit=3) if result else []
    return render_template("fertilizer_recommendation.html", result=result, inputs=inputs, soil_types=SOIL_TYPES,
                           crop_types=FERT_CROP_TYPES, products=products)


@bp.route("/download_fertilizer_report", methods=["GET", "POST"])
@farmer_required
def download_fertilizer_report():
    inputs, result = session.get("fert_inputs"), session.get("fert_result")
    if not inputs or not result:
        flash("Run a fertilizer recommendation first.", "warning")
        return redirect(url_for("farmer.fertilizer_recommendation"))
    buf = reports.fertilizer_report(inputs, result, current_app.static_folder, current_farmer().name)
    return send_file(buf, as_attachment=True, download_name="krishidisha_fertilizer_report.pdf",
                     mimetype="application/pdf")


# --------------------------------------------------------------- yield
@bp.route("/crop_yield", methods=["GET", "POST"])
@farmer_required
def crop_yield():
    meta = current_app.ml.yield_meta
    result = None
    inputs = session.get("yield_inputs") or {}
    if request.method == "POST":
        try:
            inputs = {
                "crop": request.form["crop"], "crop_year": int(request.form["crop_year"]),
                "season": request.form["season"], "state": request.form["state"],
                "area": form_float("area"), "production": form_float("production"),
                "annual_rainfall": form_float("annual_rainfall"), "fertilizer": form_float("fertilizer"),
                "pesticide": form_float("pesticide"),
            }
            result = current_app.ml.predict_yield(**inputs)
            session["yield_inputs"] = inputs
            session["yield_result"] = result
            log_activity("Crop Yield Prediction", inputs, {"predicted_yield": result["predicted_yield"]})
        except (ValueError, KeyError) as exc:
            flash(f"Invalid input: {exc}", "danger")
    return render_template("crop_yield.html", result=result, inputs=inputs, crops=meta["crops"], states=meta["states"],
                           seasons=meta["seasons"])


@bp.route("/download_yield_report", methods=["GET", "POST"])
@farmer_required
def download_yield_report():
    inputs, result = session.get("yield_inputs"), session.get("yield_result")
    if not inputs or not result:
        flash("Run a yield prediction first.", "warning")
        return redirect(url_for("farmer.crop_yield"))
    buf = reports.yield_report(inputs, result, current_farmer().name)
    return send_file(buf, as_attachment=True, download_name="krishidisha_yield_report.pdf", mimetype="application/pdf")


# ------------------------------------------------------------- disease
@bp.route("/crop_detection")
@farmer_required
def crop_detection():
    return render_template("crop_detection.html", detector=current_app.detector.info())


def save_upload(file_storage) -> tuple[str, str]:
    """Validate and store an uploaded leaf image. Returns (absolute_path, relative_url)."""
    name = secure_filename(file_storage.filename or "")
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext not in ALLOWED_IMAGE:
        raise ValueError("Please upload a PNG, JPG or WEBP image.")
    try:
        img = Image.open(file_storage.stream)
        img.verify()
    except (UnidentifiedImageError, OSError):
        raise ValueError("The file is not a valid image.")
    file_storage.stream.seek(0)
    fname = f"{uuid.uuid4().hex}.{ext}"
    upload_dir = current_app.config["UPLOAD_DIR"]
    path = os.path.join(upload_dir, fname)
    file_storage.save(path)
    return path, f"/static/uploads/{fname}"


@bp.route("/submit", methods=["POST"])
@farmer_required
def submit():
    file = request.files.get("image")
    if not file or not file.filename:
        flash("Choose an image first.", "warning")
        return redirect(url_for("farmer.crop_detection"))
    try:
        path, url = save_upload(file)
    except ValueError as exc:
        flash(str(exc), "danger")
        return redirect(url_for("farmer.crop_detection"))
    result = current_app.detector.predict(path)
    if not result.get("available"):
        flash(result.get("message", "Disease model unavailable."), "danger")
        return redirect(url_for("farmer.crop_detection"))
    top = result["top"]
    is_plant = result.get("is_plant", True)
    info = current_app.kb.disease_by_label(top["label"]) if is_plant else None
    from .marketplace import search_products

    # never recommend products for a photo the model does not recognise as a leaf
    products = search_products(disease=top["label"], crop=top["crop"], limit=4) if is_plant and not top["is_healthy"] else []
    session["disease_result"] = {"result": result, "info": info, "image_path": path, "image_url": url,
                                 "product_ids": [p.id for p in products]}
    log_activity("Crop Disease Detection", {"image": url},
                 {"disease": top["name"], "confidence": top["confidence"], "uncertain": result.get("uncertain"),
                  "is_plant": is_plant, "model": result.get("model")})
    return render_template("submit.html", result=result, top=top, info=info, image_url=url, products=products)


@bp.route("/download_disease_report")
@farmer_required
def download_disease_report():
    data = session.get("disease_result")
    if not data:
        flash("Analyse a leaf image first.", "warning")
        return redirect(url_for("farmer.crop_detection"))
    from ..models import Product

    products = [p.to_dict() for p in Product.query.filter(Product.id.in_(data["product_ids"])).all()] if data["product_ids"] else []
    buf = reports.disease_report(data["result"], data["info"], data["image_path"], products, current_farmer().name)
    return send_file(buf, as_attachment=True, download_name="krishidisha_disease_report.pdf", mimetype="application/pdf")


@bp.route("/supplements")
def supplements():
    return redirect(url_for("marketplace.index"))


# ----------------------------------------------------------- dashboard
@bp.route("/farmer_dashboard", methods=["GET", "POST"])
@farmer_required
def dashboard():
    farmer = current_farmer()
    if request.method == "POST":
        act = db.session.get(FarmerActivity, request.form.get("activity_id", type=int))
        if act and act.farmer_id == farmer.id:
            act.deleted = True
            db.session.commit()
            flash("Activity removed.", "success")
        return redirect(url_for("farmer.dashboard"))

    activities = FarmerActivity.query.filter_by(farmer_id=farmer.id, deleted=False) \
        .order_by(FarmerActivity.timestamp.desc()).limit(100).all()
    activity_counts = Counter(a.activity_type for a in activities)
    orders = Order.query.filter_by(farmer_id=farmer.id).order_by(Order.created_at.desc()).limit(5).all()

    stats = _dataset_stats(current_app.config["DATA_DIR"])
    return render_template("farmer_dashboard.html", farmer=farmer, activities=activities,
                           activity_counts=dict(activity_counts), orders=orders, **stats)


def _dataset_stats(data_dir) -> dict:
    """Aggregate charts data from the bundled datasets (cached in module scope)."""
    if getattr(_dataset_stats, "_cache", None):
        return _dataset_stats._cache
    out = {"top_crops_by_yield": {}, "fertilizer_usage": {}, "state_production": {}, "season_counts": {},
           "recommended_crop_counts": {}}
    try:
        cy = pd.read_csv(os.path.join(data_dir, "crop_yield.csv"))
        cy["Season"] = cy["Season"].str.strip()
        out["top_crops_by_yield"] = cy.groupby("Crop")["Yield"].mean().sort_values(ascending=False).head(10).round(2).to_dict()
        out["season_counts"] = cy["Season"].value_counts().to_dict()
        out["state_production"] = cy.groupby("State")["Production"].sum().sort_values(ascending=False).head(10).round(0).to_dict()
        fert = pd.read_csv(os.path.join(data_dir, "Fertilizer Prediction.csv"))
        out["fertilizer_usage"] = fert["Fertilizer Name"].value_counts().to_dict()
        cr = pd.read_csv(os.path.join(data_dir, "Crop_recommendation.csv"))
        out["recommended_crop_counts"] = cr["label"].value_counts().to_dict()
    except Exception as exc:  # noqa: BLE001
        current_app.logger.warning("dashboard stats failed: %s", exc)
    _dataset_stats._cache = out
    return out
