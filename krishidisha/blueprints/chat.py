"""Assistant chat: web UI, JSON send endpoint (text + optional leaf photo), session history."""
from __future__ import annotations

import json
import secrets

from flask import Blueprint, current_app, jsonify, redirect, render_template, request, session, url_for

from ..extensions import db
from ..models import ChatMessage, ChatSession
from ..services.llm import LANGUAGE_NAMES
from ..utils import current_farmer, log_activity

bp = Blueprint("chat", __name__)

MAX_IMAGE_BYTES = 6 * 1024 * 1024
SUGGESTIONS = [
    "Which crop should I grow? N=90 P=42 K=43 temp=21 humidity=82 ph=6.5 rain=200",
    "How much urea and DAP for 2 acres of wheat?",
    "Weather forecast for Indore this week",
    "Mandi price of soybean in Madhya Pradesh",
    "How do I control late blight in tomato?",
    "Tell me about PM-KISAN and how to apply",
    "गेहूं में पीला रतुआ का इलाज क्या है?",
]


def _anon_keys() -> list[str]:
    return session.get("chat_keys", [])


def _visible_sessions():
    farmer = current_farmer()
    if farmer:
        return ChatSession.query.filter_by(farmer_id=farmer.id).order_by(ChatSession.updated_at.desc()).limit(30).all()
    keys = _anon_keys()
    if not keys:
        return []
    return ChatSession.query.filter(ChatSession.session_key.in_(keys)).order_by(ChatSession.updated_at.desc()).all()


def _get_session(key: str | None, create: bool = True) -> ChatSession | None:
    farmer = current_farmer()
    chat = ChatSession.query.filter_by(session_key=key).first() if key else None
    if chat:
        owner_ok = (farmer and chat.farmer_id == farmer.id) or (chat.farmer_id is None and key in _anon_keys())
        if not owner_ok:
            chat = None
    if chat is None and create:
        lang = farmer.preferred_language if farmer and farmer.preferred_language else "en"
        chat = ChatSession(farmer_id=farmer.id if farmer else None, session_key=secrets.token_urlsafe(24), language=lang)
        db.session.add(chat)
        db.session.commit()
        if not farmer:
            keys = _anon_keys()
            keys.append(chat.session_key)
            session["chat_keys"] = keys[-10:]
    return chat


def _serialize(chat: ChatSession) -> dict:
    return {
        "key": chat.session_key, "title": chat.title, "language": chat.language,
        "updated_at": chat.updated_at.isoformat(),
        "messages": [{"role": m.role, "content": m.content, "tools_used": json.loads(m.tools_used or "[]"),
                      "provider": m.provider, "created_at": m.created_at.isoformat()} for m in chat.messages],
    }


# --------------------------------------------------------------------- pages
@bp.route("/chat", methods=["GET"])
def page():
    key = request.args.get("s")
    chat = _get_session(key, create=False) if key else None
    farmer = current_farmer()
    language = (chat.language if chat else None) or (farmer.preferred_language if farmer else None) or "en"
    return render_template("chat/index.html", chat=_serialize(chat) if chat else None, sessions=_visible_sessions(),
                           languages=LANGUAGE_NAMES, language=language, suggestions=SUGGESTIONS,
                           assistant=current_app.assistant.describe())


@bp.route("/chat/new")
def new():
    chat = _get_session(None, create=True)
    return redirect(url_for("chat.page", s=chat.session_key))


# ----------------------------------------------------------------------- api
@bp.route("/chat", methods=["POST"])
@bp.route("/chat/send", methods=["POST"])
def send():
    """Accepts JSON {message, session_key?, language?} or multipart form with an `image` file."""
    if request.is_json:
        data = request.get_json(silent=True) or {}
        image_bytes, image_mime = None, "image/jpeg"
    else:
        data = request.form
        file = request.files.get("image")
        image_bytes, image_mime = None, "image/jpeg"
        if file and file.filename:
            image_bytes = file.read(MAX_IMAGE_BYTES + 1)
            if len(image_bytes) > MAX_IMAGE_BYTES:
                return jsonify(error="Image larger than 6 MB"), 413
            image_mime = file.mimetype if file.mimetype in ("image/png", "image/jpeg", "image/webp") else "image/jpeg"

    message = (data.get("message") or "").strip()
    if not message and not image_bytes:
        return jsonify(error="message is required"), 400
    if not message:
        message = "Please analyse this leaf photo and tell me what to do."

    chat = _get_session(data.get("session_key"), create=True)
    language = (data.get("language") or chat.language or "en")[:10]
    chat.language = language
    farmer = current_farmer()

    history = [{"role": m.role, "content": m.content} for m in chat.messages]
    try:
        result = current_app.assistant.chat(
            message, history=history, farmer_context=farmer.profile_summary() if farmer else None,
            language=language, image_bytes=image_bytes, image_mime=image_mime)
    except Exception as exc:  # noqa: BLE001 - never leave the widget hanging
        current_app.logger.exception("assistant failed")
        result = {"reply": "Sorry, the assistant is unavailable right now. Please try again in a moment.",
                  "tools_used": [], "provider": "error", "sources": [], "error": str(exc)[:200]}

    stored_user = message + (" [photo attached]" if image_bytes else "")
    db.session.add(ChatMessage(session_id=chat.id, role="user", content=stored_user))
    db.session.add(ChatMessage(session_id=chat.id, role="assistant", content=result["reply"],
                               tools_used=json.dumps(result.get("tools_used", [])), provider=result.get("provider")))
    if chat.title == "New conversation" or not chat.title:
        chat.title = (message[:60] + ("…" if len(message) > 60 else "")) or "Leaf photo"
    db.session.commit()
    if farmer:
        log_activity("Assistant Chat", {"message": message[:200]}, {"provider": result.get("provider"),
                                                                    "tools": result.get("tools_used", [])})

    detection = result.get("detection")
    return jsonify(
        reply=result["reply"], session_key=chat.session_key, title=chat.title, provider=result.get("provider"),
        model=result.get("model"), tools_used=result.get("tools_used", []), sources=result.get("sources", []),
        detection=detection if detection and detection.get("available") else None,
        fallback_reason=result.get("fallback_reason"),
    )


@bp.route("/chat/sessions")
def sessions():
    return jsonify(sessions=[{"key": s.session_key, "title": s.title, "updated_at": s.updated_at.isoformat(),
                              "messages": len(s.messages)} for s in _visible_sessions()])


@bp.route("/chat/sessions/<key>")
def session_detail(key):
    chat = _get_session(key, create=False)
    if not chat:
        return jsonify(error="not found"), 404
    return jsonify(_serialize(chat))


@bp.route("/chat/sessions/<key>/delete", methods=["POST"])
def session_delete(key):
    chat = _get_session(key, create=False)
    if chat:
        db.session.delete(chat)
        db.session.commit()
        if not current_farmer():
            session["chat_keys"] = [k for k in _anon_keys() if k != key]
    if request.is_json or request.headers.get("X-Requested-With") == "fetch":
        return jsonify(ok=True)
    return redirect(url_for("chat.page"))
