"""Idempotent database seeding: default admin + marketplace catalogue."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from .extensions import db
from .models import Admin, Product

log = logging.getLogger(__name__)


def seed_admin(username: str, password: str) -> bool:
    if Admin.query.count():
        return False
    admin = Admin(username=username)
    admin.set_password(password)
    db.session.add(admin)
    db.session.commit()
    log.info("Created default admin '%s'", username)
    return True


def seed_products(data_dir: Path, force: bool = False) -> int:
    path = Path(data_dir) / "knowledge" / "products.json"
    if not path.exists():
        log.warning("No product catalogue at %s", path)
        return 0
    if Product.query.count() and not force:
        return 0
    items = json.loads(path.read_text(encoding="utf-8"))
    existing = {p.slug: p for p in Product.query.all()}
    n = 0
    for it in items:
        p = existing.get(it["slug"]) or Product(slug=it["slug"])
        for key in ("name", "category", "brand", "description", "price", "mrp", "unit", "stock", "image_url",
                    "external_url", "npk", "disease_tags", "crop_tags", "is_organic", "rating"):
            if key in it:
                setattr(p, key, it[key])
        db.session.add(p)
        n += 1
    db.session.commit()
    log.info("Seeded %d products", n)
    return n
