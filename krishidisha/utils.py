"""Small helpers shared by blueprints: auth guards, activity logging, form parsing."""
from __future__ import annotations

import json
from functools import wraps
from typing import Any

from flask import current_app, flash, g, redirect, request, session, url_for

from .extensions import db
from .models import DataConsent, Farmer, FarmerActivity


def current_farmer() -> Farmer | None:
    if "farmer" in g:
        return g.farmer
    fid = session.get("farmer_id")
    g.farmer = db.session.get(Farmer, fid) if fid else None
    return g.farmer


def farmer_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if current_farmer() is None:
            if request.path.startswith("/api/") or request.is_json:
                return {"error": "login required"}, 401
            flash("Please log in to continue.", "warning")
            return redirect(url_for("auth.farmer_login", next=request.path))
        return view(*args, **kwargs)
    return wrapped


def admin_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("admin_logged_in"):
            flash("Admin login required.", "warning")
            return redirect(url_for("auth.admin_login"))
        return view(*args, **kwargs)
    return wrapped


def log_activity(activity_type: str, input_data: Any, output_data: Any, farmer: Farmer | None = None) -> None:
    farmer = farmer or current_farmer()
    if farmer is None:
        return
    try:
        db.session.add(FarmerActivity(
            farmer_id=farmer.id, activity_type=activity_type,
            input_data=json.dumps(input_data, default=str), output_data=json.dumps(output_data, default=str)))
        db.session.commit()
    except Exception as exc:  # noqa: BLE001
        db.session.rollback()
        current_app.logger.warning("activity log failed: %s", exc)


# --------------------------------------------------------------------------- #
# Data-use consent
# --------------------------------------------------------------------------- #
# Feedback kinds map onto the two consent switches a farmer actually sees.
CONSENT_FIELD_FOR = {
    "photos": "photos", "diagnosis": "photos", "image": "photos",
    "chats": "chats", "chat": "chats", "yield": "chats", "fertilizer": "chats",
}


def consent_row(farmer: Farmer | None) -> DataConsent | None:
    if farmer is None:
        return None
    return DataConsent.query.filter_by(farmer_id=farmer.id).first()


def has_consent(farmer: Farmer | None, kind: str) -> bool:
    """True when *farmer* opted in to reusing this kind of data ("photos"/"chats" or a feedback kind)."""
    row = consent_row(farmer)
    if row is None:
        return False
    return bool(getattr(row, CONSENT_FIELD_FOR.get(kind, "chats"), False))


def set_consent(farmer: Farmer | None, photos: bool, chats: bool, commit: bool = True) -> DataConsent | None:
    """Create or update the farmer's consent row."""
    if farmer is None:
        return None
    row = consent_row(farmer)
    if row is None:
        row = DataConsent(farmer_id=farmer.id)
        db.session.add(row)
    row.photos, row.chats = bool(photos), bool(chats)
    if commit:
        db.session.commit()
    return row


def form_float(name: str, default: float | None = None) -> float:
    raw = request.form.get(name, "") if request.form else ""
    if raw == "" and request.is_json:
        raw = (request.get_json(silent=True) or {}).get(name, "")
    if raw in ("", None):
        if default is None:
            raise ValueError(f"{name} is required")
        return default
    return float(raw)


def wants_json() -> bool:
    return request.is_json or request.args.get("format") == "json" or \
        request.headers.get("Accept", "").startswith("application/json")
