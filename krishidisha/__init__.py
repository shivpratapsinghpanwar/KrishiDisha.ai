"""KrishiDisha application factory."""
from __future__ import annotations

import logging
import os
from pathlib import Path

import click
from flask import Flask, jsonify, render_template, request

from .config import Config
from .extensions import db


def create_app(config_object=Config) -> Flask:
    base = Path(__file__).resolve().parent.parent
    app = Flask(__name__, template_folder=str(base / "templates"), static_folder=str(base / "static"),
                instance_path=str(base / "instance"))
    app.config.from_object(config_object)
    os.makedirs(app.instance_path, exist_ok=True)
    os.makedirs(app.config["UPLOAD_DIR"], exist_ok=True)
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    db.init_app(app)
    _register_services(app)
    _register_blueprints(app)
    _register_errors(app)
    _register_cli(app)
    _register_context(app)

    with app.app_context():
        from . import models  # noqa: F401 - ensure tables are registered
        db.create_all()
        from .seed import seed_admin, seed_products

        seed_admin(app.config["DEFAULT_ADMIN_USERNAME"], app.config["DEFAULT_ADMIN_PASSWORD"])
        seed_products(app.config["DATA_DIR"])
    return app


def _register_services(app: Flask) -> None:
    """Instantiate services once per process and attach them to the app."""
    from .services.disease import DiseaseDetector
    from .services.knowledge import KnowledgeBase
    from .services.llm import AgriAssistant
    from .services.ml import MLService
    from .services.tools import build_tools

    cfg = app.config
    app.ml = MLService(cfg["DATA_DIR"], cfg["MODELS_DIR"])
    app.kb = KnowledgeBase(cfg["DATA_DIR"])
    app.detector = DiseaseDetector(cfg["MODELS_DIR"], cfg["DISEASE_MODEL_BACKEND"], cfg["DISEASE_HF_MODEL"])

    def search_products(query=None, category=None, disease=None, crop=None, limit=6):
        from .blueprints.marketplace import search_products as _search

        with app.app_context():
            return [dict(p.to_dict(), url=f"/marketplace/product/{p.slug}")
                    for p in _search(query, category, disease, crop, limit)]

    class _Cfg:  # attribute-style access for the assistant
        def __getattr__(self, item):
            return cfg.get(item)

    app.tools = build_tools(app.ml, app.kb, cfg["DATA_DIR"], cfg["DATA_GOV_API_KEY"], search_products)
    app.assistant = AgriAssistant(_Cfg(), app.tools, app.kb, app.detector)


def _register_blueprints(app: Flask) -> None:
    from .blueprints.admin import bp as admin_bp
    from .blueprints.api import bp as api_bp
    from .blueprints.auth import bp as auth_bp
    from .blueprints.chat import bp as chat_bp
    from .blueprints.farmer import bp as farmer_bp
    from .blueprints.feedback import bp as feedback_bp
    from .blueprints.main import bp as main_bp
    from .blueprints.marketplace import bp as market_bp

    for bp in (main_bp, auth_bp, farmer_bp, market_bp, admin_bp, chat_bp, api_bp, feedback_bp):
        app.register_blueprint(bp)


def _register_errors(app: Flask) -> None:
    @app.errorhandler(404)
    def not_found(_):
        if request.path.startswith("/api/"):
            return jsonify(error="not found"), 404
        return render_template("errors/404.html"), 404

    @app.errorhandler(413)
    def too_large(_):
        return render_template("errors/error.html", code=413, message="Upload too large (max 10 MB)."), 413

    @app.errorhandler(500)
    def server_error(exc):
        app.logger.exception("Unhandled error: %s", exc)
        if request.path.startswith("/api/"):
            return jsonify(error="internal error"), 500
        return render_template("errors/error.html", code=500, message="Something went wrong on our side."), 500


def _register_context(app: Flask) -> None:
    from .utils import current_farmer

    @app.context_processor
    def inject():
        from .models import CartItem

        farmer = current_farmer()
        cart_count = 0
        if farmer:
            cart_count = db.session.query(db.func.coalesce(db.func.sum(CartItem.quantity), 0)) \
                .filter(CartItem.farmer_id == farmer.id).scalar() or 0
        return {
            "current_farmer": farmer,
            "cart_count": int(cart_count),
            "assistant_info": app.assistant.describe(),
            "app_name": "KrishiDisha",
        }

    @app.template_filter("inr")
    def inr(value):
        try:
            return f"₹{float(value):,.0f}"
        except (TypeError, ValueError):
            return value

    @app.template_filter("pct")
    def pct(value):
        try:
            return f"{float(value) * 100:.1f}%"
        except (TypeError, ValueError):
            return value


def _register_cli(app: Flask) -> None:
    @app.cli.command("seed")
    @click.option("--force", is_flag=True, help="Re-import products even if the table is populated.")
    def seed_cmd(force):
        """Seed the default admin and the marketplace catalogue."""
        from .seed import seed_admin, seed_products

        created = seed_admin(app.config["DEFAULT_ADMIN_USERNAME"], app.config["DEFAULT_ADMIN_PASSWORD"])
        n = seed_products(app.config["DATA_DIR"], force=force)
        click.echo(f"admin created: {created}; products imported: {n}")

    @app.cli.command("create-admin")
    @click.argument("username")
    @click.argument("password")
    def create_admin_cmd(username, password):
        from .models import Admin

        admin = Admin.query.filter_by(username=username).first() or Admin(username=username)
        admin.set_password(password)
        db.session.add(admin)
        db.session.commit()
        click.echo(f"admin '{username}' ready")

    @app.cli.command("warmup")
    def warmup_cmd():
        """Load / train all ML models so the first request is fast."""
        app.ml.crop_model, app.ml.fertilizer_model, app.ml.yield_model  # noqa: B018
        click.echo(f"tabular models: {app.ml.status()}")
        click.echo(f"disease model: {app.detector.info()}")
        click.echo(f"assistant: {app.assistant.describe()}")
