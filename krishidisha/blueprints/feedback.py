"""Feedback loop: farmers rate answers, and consented photos become human-labelling tasks.

Two public endpoints:

``POST /feedback``         store one :class:`~krishidisha.models.Feedback` row (and, for a
                           diagnosis with photo consent, queue the image for labelling).
``GET  /feedback/labels``  the canonical class list used by every correction dropdown.

The image helpers here are also used by :mod:`krishidisha.blueprints.chat`,
:mod:`krishidisha.blueprints.admin` and ``ml/datasets/export_feedback.py``.
"""
from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

from flask import Blueprint, current_app, jsonify, request, session

from ..extensions import db
from ..models import FEEDBACK_KINDS, UNSURE, Feedback, LabelTask
from ..utils import current_farmer, has_consent

bp = Blueprint("feedback", __name__)

ALLOWED_EXT = {"jpg", "jpeg", "png", "webp"}
MIME_EXT = {"image/jpeg": "jpg", "image/jpg": "jpg", "image/png": "png", "image/webp": "webp"}

# A small, sensible default when no trained detector is loaded (stub backend, fresh checkout).
_EXTRA_LABELS = [
    "Rice___healthy", "Rice___Bacterial_leaf_blight", "Rice___Blast", "Rice___Brown_spot", "Rice___Tungro",
    "Wheat___healthy", "Wheat___Leaf_rust", "Wheat___Stripe_rust", "Wheat___Septoria",
    "Maize___healthy", "Maize___Common_rust", "Maize___Northern_leaf_blight", "Maize___Gray_leaf_spot",
    "Cotton___healthy", "Cotton___Bacterial_blight", "Cotton___Leaf_curl_virus", "Cotton___Whitefly",
    "Sugarcane___healthy", "Sugarcane___Red_rot", "Sugarcane___Rust", "Sugarcane___Yellow_leaf",
    "Soybean___healthy", "Soybean___Rust", "Groundnut___healthy", "Groundnut___Early_leaf_spot",
    "Chilli___healthy", "Chilli___Leaf_curl", "Tomato___healthy", "Tomato___Early_blight", "Tomato___Late_blight",
    "Potato___healthy", "Potato___Early_blight", "Potato___Late_blight",
    "Onion___healthy", "Onion___Purple_blotch", "Mango___healthy", "Mango___Anthracnose",
    "Banana___healthy", "Banana___Sigatoka", "Mustard___healthy", "Chickpea___healthy", "Chickpea___Wilt",
]


# --------------------------------------------------------------------------- #
# Canonical labels
# --------------------------------------------------------------------------- #
def not_a_leaf_label() -> str:
    try:
        from ml.datasets.taxonomy import NOT_A_LEAF

        return NOT_A_LEAF
    except Exception:  # noqa: BLE001 - the ml package is optional at runtime
        return "Other___not_a_leaf"


def canonical_labels() -> list[str]:
    """The detector's classes when a real model is loaded, else the taxonomy defaults.

    Always contains ``Other___not_a_leaf`` and the ``unsure`` sentinel (last).
    """
    labels: list[str] = []
    detector = getattr(current_app, "detector", None)
    if detector is not None:
        try:
            detector.info()  # loads the model lazily
            labels = [str(c) for c in (getattr(detector, "classes", None) or [])]
        except Exception:  # noqa: BLE001 - never break the dropdown because a model failed
            labels = []
    if not labels:
        try:
            from ml.datasets.taxonomy import SOURCE_MAPS

            labels = [v for table in SOURCE_MAPS.values() for v in table.values()]
        except Exception:  # noqa: BLE001
            labels = []
        labels += _EXTRA_LABELS
    ordered = sorted({lbl for lbl in labels if lbl and lbl != UNSURE})
    nal = not_a_leaf_label()
    if nal not in ordered:
        ordered.append(nal)
    ordered.append(UNSURE)
    return ordered


def grouped_labels(labels: list[str] | None = None) -> dict[str, list[str]]:
    """``{crop: [label, ...]}`` for <optgroup>-style dropdowns."""
    groups: dict[str, list[str]] = {}
    for label in labels if labels is not None else canonical_labels():
        crop = label.split("___", 1)[0] if "___" in label else "Other"
        groups.setdefault(crop.replace("_", " "), []).append(label)
    return groups


# --------------------------------------------------------------------------- #
# Image storage helpers (shared with chat + admin)
# --------------------------------------------------------------------------- #
def upload_root() -> Path:
    return Path(current_app.config["UPLOAD_DIR"])


def resolve_image(image_path: str | None) -> Path | None:
    """Absolute path for a :class:`LabelTask.image_path` (relative to UPLOAD_DIR, or absolute)."""
    if not image_path:
        return None
    p = Path(image_path)
    return p if p.is_absolute() else upload_root() / p


def image_url(image_path: str | None) -> str | None:
    """Browser URL for a stored image, or ``None`` when it lives outside the upload dir."""
    if not image_path:
        return None
    p = Path(image_path)
    if p.is_absolute():
        try:
            p = p.relative_to(upload_root())
        except ValueError:
            return None
    return "/static/uploads/" + p.as_posix()


def _safe_ext(name: str | None, mime: str | None = None, default: str = "jpg") -> str:
    ext = (Path(name or "").suffix or "").lstrip(".").lower()
    if ext in ALLOWED_EXT:
        return ext
    return MIME_EXT.get((mime or "").lower(), default)


def store_bytes(data: bytes, subdir: str, ext: str = "jpg") -> str:
    """Write raw image bytes under ``UPLOAD_DIR/<subdir>/`` and return the relative path."""
    folder = upload_root() / subdir
    folder.mkdir(parents=True, exist_ok=True)
    rel = f"{subdir}/{uuid.uuid4().hex}.{ext}"
    (upload_root() / rel).write_bytes(data)
    return rel


def store_copy(src: str | os.PathLike, subdir: str) -> str | None:
    """Copy an existing upload into ``UPLOAD_DIR/<subdir>/`` and return the relative path."""
    src_path = Path(src)
    if not src_path.is_file():
        return None
    folder = upload_root() / subdir
    folder.mkdir(parents=True, exist_ok=True)
    rel = f"{subdir}/{uuid.uuid4().hex}.{_safe_ext(src_path.name)}"
    shutil.copyfile(src_path, upload_root() / rel)
    return rel


def create_label_task(image_path: str, source: str, farmer=None, model_label: str | None = None,
                      model_confidence: float | None = None, consent: bool = True,
                      notes: str | None = None, commit: bool = True) -> LabelTask:
    task = LabelTask(image_path=image_path, source=source, farmer_id=farmer.id if farmer else None,
                     model_label=model_label, model_confidence=model_confidence, consent=bool(consent),
                     status="queued", notes=notes)
    db.session.add(task)
    if commit:
        db.session.commit()
    return task


def capture_chat_photo(chat, image_bytes: bytes, image_mime: str, result: dict | None = None) -> LabelTask | None:
    """Hook used by ``chat.send``: keep a consented leaf photo for labelling."""
    farmer = current_farmer()
    if not image_bytes or not has_consent(farmer, "photos"):
        return None
    try:
        detection = (result or {}).get("detection") or {}
        top = detection.get("top") or {}
        rel = store_bytes(image_bytes, "chat", _safe_ext(None, image_mime))
        return create_label_task(rel, "chat", farmer=farmer, model_label=top.get("label"),
                                 model_confidence=top.get("confidence"), consent=True,
                                 notes=f"chat session {getattr(chat, 'session_key', '')}")
    except Exception as exc:  # noqa: BLE001 - collection must never break a chat reply
        db.session.rollback()
        current_app.logger.warning("chat photo capture failed: %s", exc)
        return None


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@bp.route("/feedback/labels")
def labels():
    labs = canonical_labels()
    return jsonify(labels=labs, grouped=grouped_labels(labs), not_a_leaf=not_a_leaf_label(),
                   unsure=UNSURE, count=len(labs))


def _as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@bp.route("/feedback", methods=["POST"])
def submit():
    data = request.get_json(silent=True) or request.form or {}
    kind = (data.get("kind") or "").strip().lower()
    if kind not in FEEDBACK_KINDS:
        return jsonify(error=f"kind must be one of {', '.join(FEEDBACK_KINDS)}"), 400
    ref_id = (str(data.get("ref_id") or "")).strip()
    if not ref_id:
        return jsonify(error="ref_id is required"), 400
    try:
        rating = int(data.get("rating", 0))
    except (TypeError, ValueError):
        return jsonify(error="rating must be -1, 0 or 1"), 400
    if rating not in (-1, 0, 1):
        return jsonify(error="rating must be -1, 0 or 1"), 400

    farmer = current_farmer()
    consent = has_consent(farmer, kind)
    corrected = (data.get("corrected_label") or "").strip() or None
    comment = (data.get("comment") or "").strip() or None
    model_label = (data.get("model_label") or "").strip() or None
    model_confidence = _as_float(data.get("model_confidence"))
    language = (data.get("language") or "").strip()[:10] or (farmer.preferred_language if farmer else None)

    image_path = None
    stored = session.get("disease_result") if kind == "diagnosis" else None
    if stored:
        image_path = stored.get("image_path")
        top = (stored.get("result") or {}).get("top") or {}
        model_label = model_label or top.get("label")
        if model_confidence is None:
            model_confidence = _as_float(top.get("confidence"))

    row = Feedback(farmer_id=farmer.id if farmer else None, kind=kind, ref_id=ref_id[:200], rating=rating,
                   corrected_label=corrected, comment=comment, image_path=image_path, model_label=model_label,
                   model_confidence=model_confidence, language=language, consent=bool(consent), status="pending")
    db.session.add(row)
    db.session.commit()

    task_id = None
    if kind == "diagnosis" and image_path and has_consent(farmer, "photos"):
        rel = store_copy(image_path, "labels")
        if rel:
            task = create_label_task(rel, "app_upload", farmer=farmer, model_label=model_label,
                                     model_confidence=model_confidence, consent=True,
                                     notes=f"feedback #{row.id}")
            row.image_path = rel
            db.session.commit()
            task_id = task.id

    return jsonify(ok=True, id=row.id, label_task_id=task_id, consent=bool(consent))
