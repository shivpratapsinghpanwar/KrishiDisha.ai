"""Farmer registration / login / profile and admin login."""
from __future__ import annotations

import re

from flask import Blueprint, current_app, flash, redirect, render_template, request, session, url_for
from sqlalchemy.exc import IntegrityError

from ..extensions import db
from ..models import Admin, Farmer
from ..utils import current_farmer, farmer_required

bp = Blueprint("auth", __name__)

INDIAN_STATES = ["Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa", "Gujarat", "Haryana",
                 "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur",
                 "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana",
                 "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal", "Delhi", "Jammu and Kashmir", "Ladakh",
                 "Puducherry", "Chandigarh", "Andaman and Nicobar Islands"]

LANGUAGES = [("en", "English"), ("hi", "हिन्दी (Hindi)"), ("hinglish", "Hinglish"), ("mr", "मराठी (Marathi)"),
             ("pa", "ਪੰਜਾਬੀ (Punjabi)"), ("gu", "ગુજરાતી (Gujarati)"), ("ta", "தமிழ் (Tamil)"), ("te", "తెలుగు (Telugu)"),
             ("kn", "ಕನ್ನಡ (Kannada)"), ("bn", "বাংলা (Bengali)")]


@bp.route("/farmer_registration", methods=["GET", "POST"])
def farmer_registration():
    if request.method == "POST":
        f = request.form
        errors = []
        if not re.fullmatch(r"[6-9]\d{9}", f.get("phone", "").strip()):
            errors.append("Enter a valid 10-digit Indian mobile number.")
        if len(f.get("password", "")) < 6:
            errors.append("Password must be at least 6 characters.")
        if not re.fullmatch(r"[A-Za-z0-9_.]{3,45}", f.get("username", "")):
            errors.append("Username: 3-45 letters, digits, dot or underscore.")
        if errors:
            for e in errors:
                flash(e, "danger")
            return render_template("farmer_registration.html", states=INDIAN_STATES, languages=LANGUAGES, form=f)
        farmer = Farmer(name=f["name"].strip(), email=f["email"].strip().lower(), phone=f["phone"].strip(),
                        username=f["username"].strip(), state=f.get("state") or None, district=f.get("district") or None,
                        land_area_acres=float(f["land_area_acres"]) if f.get("land_area_acres") else None,
                        preferred_language=f.get("preferred_language", "en"),
                        verified=current_app.config["AUTO_VERIFY_FARMERS"])
        farmer.set_password(f["password"])
        try:
            db.session.add(farmer)
            db.session.commit()
        except IntegrityError:
            db.session.rollback()
            flash("Username, email or phone already registered.", "danger")
            return render_template("farmer_registration.html", states=INDIAN_STATES, languages=LANGUAGES, form=f)
        if farmer.verified:
            session["farmer_id"] = farmer.id
            flash(f"Welcome to KrishiDisha, {farmer.name}!", "success")
            return redirect(url_for("farmer.home"))
        flash("Registration successful. An admin will verify your account shortly.", "success")
        return redirect(url_for("auth.farmer_login"))
    return render_template("farmer_registration.html", states=INDIAN_STATES, languages=LANGUAGES, form={})


@bp.route("/farmer_login", methods=["GET", "POST"])
def farmer_login():
    if request.method == "POST":
        ident = request.form.get("username", "").strip()
        farmer = Farmer.query.filter((Farmer.username == ident) | (Farmer.email == ident.lower()) |
                                     (Farmer.phone == ident)).first()
        if farmer and farmer.check_password(request.form.get("password", "")):
            if not farmer.verified:
                flash("Your account is awaiting admin verification.", "warning")
            else:
                session["farmer_id"] = farmer.id
                session.permanent = True
                flash(f"Welcome back, {farmer.name}!", "success")
                nxt = request.args.get("next") or request.form.get("next")
                return redirect(nxt if nxt and nxt.startswith("/") else url_for("farmer.home"))
        else:
            flash("Invalid credentials. Please try again.", "danger")
    return render_template("farmer_login.html")


@bp.route("/farmer_logout")
def farmer_logout():
    session.pop("farmer_id", None)
    flash("Logged out.", "info")
    return redirect(url_for("main.index"))


@bp.route("/profile", methods=["GET", "POST"])
@farmer_required
def profile():
    farmer = current_farmer()
    if request.method == "POST":
        f = request.form
        farmer.name = f.get("name", farmer.name).strip()
        farmer.state = f.get("state") or None
        farmer.district = f.get("district") or None
        farmer.land_area_acres = float(f["land_area_acres"]) if f.get("land_area_acres") else None
        farmer.preferred_language = f.get("preferred_language", "en")
        if f.get("new_password"):
            if len(f["new_password"]) < 6:
                flash("New password must be at least 6 characters.", "danger")
                return redirect(url_for("auth.profile"))
            farmer.set_password(f["new_password"])
        db.session.commit()
        flash("Profile updated.", "success")
        return redirect(url_for("auth.profile"))
    return render_template("profile.html", farmer=farmer, states=INDIAN_STATES, languages=LANGUAGES)


# ------------------------------------------------------------------ admin
@bp.route("/admin_login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        admin = Admin.query.filter_by(username=request.form.get("username", "").strip()).first()
        if admin and admin.check_password(request.form.get("password", "")):
            session["admin_logged_in"] = True
            session["admin_id"] = admin.id
            flash("Admin login successful.", "success")
            return redirect(url_for("admin.dashboard"))
        flash("Invalid admin credentials.", "danger")
    return render_template("admin_login.html")


@bp.route("/admin_logout")
def admin_logout():
    session.pop("admin_logged_in", None)
    session.pop("admin_id", None)
    flash("Admin logged out.", "info")
    return redirect(url_for("auth.admin_login"))
