"""Input marketplace: catalogue, product pages, cart, checkout and order history."""
from __future__ import annotations

import random
import re
from datetime import datetime

from flask import Blueprint, abort, current_app, flash, jsonify, redirect, render_template, request, url_for
from sqlalchemy import or_

from ..extensions import db
from ..models import ORDER_STATUSES, Address, CartItem, Order, OrderItem, Product
from ..utils import current_farmer, farmer_required, log_activity, wants_json

bp = Blueprint("marketplace", __name__, url_prefix="/marketplace")

CATEGORIES = ["Fertilizer", "Fungicide", "Insecticide", "Seeds", "Organic", "Tools", "Bio-stimulant", "Herbicide"]
PAGE_SIZE = 12

# Words in a PlantVillage condition that hint at the product category that treats it.
_CONDITION_HINTS = {
    "Fungicide": ("blight", "rot", "rust", "mildew", "scab", "spot", "mold", "mould", "leaf scorch", "measles", "esca",
                  "blast", "smut", "wilt", "anthracnose", "greening"),
    "Insecticide": ("mite", "spider", "borer", "aphid", "thrips", "whitefly", "hopper", "jassid", "psyllid", "worm"),
    "Organic": ("healthy",),
}


def _tokens(text: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", (text or "").lower()) if len(t) > 2]


def search_products(query: str | None = None, category: str | None = None, disease: str | None = None,
                    crop: str | None = None, limit: int = 6) -> list[Product]:
    """Rank active products for a free-text query, a PlantVillage disease label and/or a crop.

    Scoring (highest first): exact disease tag match, crop tag match, query tokens found in
    name / brand / category / description. When a disease label matches no tagged product the
    condition words (blight, rust, mite ...) are used to pick a sensible category instead.
    """
    q = Product.query.filter_by(is_active=True)
    if category:
        q = q.filter(db.func.lower(Product.category) == category.strip().lower())
    products = q.all()
    if not products:
        return []

    query_tokens = _tokens(query or "")
    crop_key = (crop or "").strip().lower().replace("(", " ").split()[0] if crop else ""
    disease_label = (disease or "").strip()
    condition = disease_label.split("___", 1)[1].replace("_", " ").lower() if "___" in disease_label else disease_label.lower()
    hint_category = next((cat for cat, words in _CONDITION_HINTS.items() if any(w in condition for w in words)), None)

    scored: list[tuple[float, Product]] = []
    for p in products:
        score = 0.0
        tags = [t.strip() for t in (p.disease_tags or "").split(",") if t.strip()]
        crops = [t.strip().lower() for t in (p.crop_tags or "").split(",") if t.strip()]
        if disease_label and disease_label in tags:
            score += 100
        elif disease_label and any(_overlap(disease_label, t) >= 0.6 for t in tags):
            score += 60
        if crop_key and crop_key in crops:
            score += 20
        if hint_category and p.category == hint_category:
            score += 10
        if query_tokens:
            hay = f"{p.name} {p.brand or ''} {p.category} {p.npk or ''} {p.description or ''}".lower()
            name_hay = f"{p.name} {p.brand or ''} {p.npk or ''}".lower()
            hits = sum(1 for t in query_tokens if t in hay)
            name_hits = sum(1 for t in query_tokens if t in name_hay)
            score += hits * 4 + name_hits * 6
            if query and query.lower().strip() in name_hay:
                score += 25
        score += (p.rating or 0) * 0.5
        if p.stock <= 0:
            score -= 50
        scored.append((score, p))

    scored.sort(key=lambda x: (-x[0], x[1].price))
    if query_tokens or disease_label or crop_key:
        # keep only products that matched something beyond the baseline rating score (max 5 * 0.5)
        matched = [p for s, p in scored if s > 2.6]
        if matched:
            return matched[:limit]
        if disease_label or crop_key:
            return [p for _, p in scored][:limit]
        return []
    return [p for _, p in scored][:limit]


def _overlap(a: str, b: str) -> float:
    ta, tb = set(_tokens(a.replace("___", " "))), set(_tokens(b.replace("___", " ")))
    return len(ta & tb) / len(ta | tb) if ta and tb else 0.0


# ------------------------------------------------------------------ catalogue
@bp.route("/")
def index():
    q = request.args.get("q", "").strip()
    category = request.args.get("category", "").strip()
    sort = request.args.get("sort", "popular")
    organic = request.args.get("organic") == "1"
    page = max(request.args.get("page", 1, type=int), 1)

    query = Product.query.filter_by(is_active=True)
    if category:
        query = query.filter(db.func.lower(Product.category) == category.lower())
    if organic:
        query = query.filter_by(is_organic=True)
    if q:
        like = f"%{q}%"
        query = query.filter(or_(Product.name.ilike(like), Product.brand.ilike(like), Product.description.ilike(like),
                                 Product.npk.ilike(like), Product.crop_tags.ilike(like)))
    order = {"price_asc": Product.price.asc(), "price_desc": Product.price.desc(), "newest": Product.created_at.desc(),
             "name": Product.name.asc()}.get(sort, Product.rating.desc())
    query = query.order_by(order)
    total = query.count()
    products = query.offset((page - 1) * PAGE_SIZE).limit(PAGE_SIZE).all()
    pages = max((total + PAGE_SIZE - 1) // PAGE_SIZE, 1)
    counts = dict(db.session.query(Product.category, db.func.count(Product.id))
                  .filter_by(is_active=True).group_by(Product.category).all())
    if wants_json():
        return jsonify(total=total, page=page, pages=pages, products=[p.to_dict() for p in products])
    return render_template("marketplace/index.html", products=products, q=q, category=category, sort=sort,
                           organic=organic, page=page, pages=pages, total=total, categories=CATEGORIES,
                           category_counts=counts)


@bp.route("/product/<slug>")
def product(slug):
    p = Product.query.filter_by(slug=slug).first_or_404()
    related = [r for r in search_products(crop=(p.crop_tags or "").split(",")[0], category=p.category, limit=5)
               if r.id != p.id][:4]
    diseases = []
    for tag in [t for t in (p.disease_tags or "").split(",") if t]:
        info = current_app.kb.disease_by_label(tag)
        diseases.append({"label": tag, "name": info["name"] if info else tag.replace("___", " : ").replace("_", " ")})
    if wants_json():
        return jsonify(product=p.to_dict(), related=[r.to_dict() for r in related])
    return render_template("marketplace/product.html", product=p, related=related, diseases=diseases)


# ----------------------------------------------------------------------- cart
def _cart_items(farmer):
    return CartItem.query.filter_by(farmer_id=farmer.id).order_by(CartItem.added_at.asc()).all()


def _cart_totals(items):
    subtotal = round(sum(i.line_total for i in items), 2)
    cfg = current_app.config
    delivery = 0.0 if (subtotal >= cfg["FREE_DELIVERY_ABOVE"] or subtotal == 0) else cfg["DELIVERY_CHARGE"]
    return {"subtotal": subtotal, "delivery": delivery, "total": round(subtotal + delivery, 2),
            "free_delivery_above": cfg["FREE_DELIVERY_ABOVE"]}


@bp.route("/cart")
@farmer_required
def cart():
    items = _cart_items(current_farmer())
    return render_template("marketplace/cart.html", items=items, totals=_cart_totals(items))


@bp.route("/cart/add", methods=["POST"])
@farmer_required
def cart_add():
    data = request.get_json(silent=True) or request.form
    product_id = int(data.get("product_id", 0) or 0)
    qty = max(int(data.get("quantity", 1) or 1), 1)
    p = db.session.get(Product, product_id)
    if not p or not p.is_active:
        if wants_json():
            return jsonify(error="product not found"), 404
        abort(404)
    farmer = current_farmer()
    item = CartItem.query.filter_by(farmer_id=farmer.id, product_id=p.id).first()
    if item:
        item.quantity = min(item.quantity + qty, max(p.stock, 1))
    else:
        item = CartItem(farmer_id=farmer.id, product_id=p.id, quantity=min(qty, max(p.stock, 1)))
        db.session.add(item)
    db.session.commit()
    count = db.session.query(db.func.coalesce(db.func.sum(CartItem.quantity), 0)).filter_by(farmer_id=farmer.id).scalar()
    if wants_json():
        return jsonify(ok=True, cart_count=int(count), message=f"{p.name} added to cart")
    flash(f"Added {p.name} to your cart.", "success")
    return redirect(request.referrer or url_for("marketplace.cart"))


@bp.route("/cart/update", methods=["POST"])
@farmer_required
def cart_update():
    data = request.get_json(silent=True) or request.form
    item = db.session.get(CartItem, int(data.get("item_id", 0) or 0))
    if not item or item.farmer_id != current_farmer().id:
        abort(404)
    qty = int(data.get("quantity", 1) or 0)
    if qty <= 0:
        db.session.delete(item)
    else:
        item.quantity = min(qty, max(item.product.stock, 1))
    db.session.commit()
    if wants_json():
        items = _cart_items(current_farmer())
        return jsonify(ok=True, totals=_cart_totals(items), count=sum(i.quantity for i in items))
    return redirect(url_for("marketplace.cart"))


@bp.route("/cart/remove/<int:item_id>", methods=["POST"])
@farmer_required
def cart_remove(item_id):
    item = db.session.get(CartItem, item_id)
    if item and item.farmer_id == current_farmer().id:
        db.session.delete(item)
        db.session.commit()
        flash("Item removed.", "info")
    return redirect(url_for("marketplace.cart"))


# ------------------------------------------------------------------- checkout
def _order_number() -> str:
    return f"KD{datetime.utcnow():%y%m%d}{random.randint(1000, 9999)}"


@bp.route("/checkout", methods=["GET", "POST"])
@farmer_required
def checkout():
    farmer = current_farmer()
    items = _cart_items(farmer)
    if not items:
        flash("Your cart is empty.", "warning")
        return redirect(url_for("marketplace.index"))
    totals = _cart_totals(items)
    addresses = Address.query.filter_by(farmer_id=farmer.id).order_by(Address.id.desc()).all()

    if request.method == "POST":
        f = request.form
        payment = f.get("payment_method", "COD")
        if payment not in ("COD", "UPI"):
            payment = "COD"
        address = None
        if f.get("address_id"):
            address = db.session.get(Address, int(f["address_id"]))
            if address and address.farmer_id != farmer.id:
                address = None
        if address is None:
            required = ("full_name", "phone", "line1", "district", "state", "pincode")
            if any(not f.get(k, "").strip() for k in required):
                flash("Please fill in all address fields.", "danger")
                return render_template("marketplace/checkout.html", items=items, totals=totals, addresses=addresses,
                                       form=f)
            if not re.fullmatch(r"\d{6}", f.get("pincode", "").strip()):
                flash("Enter a valid 6-digit PIN code.", "danger")
                return render_template("marketplace/checkout.html", items=items, totals=totals, addresses=addresses,
                                       form=f)
            address = Address(farmer_id=farmer.id, full_name=f["full_name"].strip(), phone=f["phone"].strip(),
                              line1=f["line1"].strip(), village=f.get("village", "").strip() or None,
                              district=f["district"].strip(), state=f["state"].strip(), pincode=f["pincode"].strip())
            db.session.add(address)
            db.session.flush()

        # stock check
        for it in items:
            if it.product.stock < it.quantity:
                flash(f"Only {it.product.stock} units of {it.product.name} are in stock.", "danger")
                return redirect(url_for("marketplace.cart"))

        order = Order(order_number=_order_number(), farmer_id=farmer.id, address_text=address.formatted(),
                      subtotal=totals["subtotal"], delivery_charge=totals["delivery"], total=totals["total"],
                      payment_method=payment, payment_status="Pending" if payment == "COD" else "Awaiting UPI",
                      status="Placed")
        for it in items:
            order.items.append(OrderItem(product_id=it.product_id, product_name=it.product.name,
                                         unit_price=it.product.price, quantity=it.quantity))
            it.product.stock = max(it.product.stock - it.quantity, 0)
            db.session.delete(it)
        db.session.add(order)
        db.session.commit()
        log_activity("Marketplace Order", {"order": order.order_number, "items": len(order.items)},
                     {"total": order.total, "payment": payment})
        flash(f"Order {order.order_number} placed successfully!", "success")
        return redirect(url_for("marketplace.order_detail", order_number=order.order_number))

    form = {"full_name": farmer.name, "phone": farmer.phone, "district": farmer.district or "", "state": farmer.state or ""}
    return render_template("marketplace/checkout.html", items=items, totals=totals, addresses=addresses, form=form)


# --------------------------------------------------------------------- orders
@bp.route("/orders")
@farmer_required
def orders():
    rows = Order.query.filter_by(farmer_id=current_farmer().id).order_by(Order.created_at.desc()).all()
    return render_template("marketplace/orders.html", orders=rows, statuses=ORDER_STATUSES)


@bp.route("/orders/<order_number>")
@farmer_required
def order_detail(order_number):
    order = Order.query.filter_by(order_number=order_number, farmer_id=current_farmer().id).first_or_404()
    upi_id = current_app.config.get("UPI_ID") or "krishidisha@upi"
    return render_template("marketplace/order_detail.html", order=order, statuses=ORDER_STATUSES, upi_id=upi_id)


@bp.route("/orders/<order_number>/cancel", methods=["POST"])
@farmer_required
def order_cancel(order_number):
    order = Order.query.filter_by(order_number=order_number, farmer_id=current_farmer().id).first_or_404()
    if order.status in ("Placed", "Confirmed"):
        order.status = "Cancelled"
        for it in order.items:
            if it.product:
                it.product.stock += it.quantity
        db.session.commit()
        flash("Order cancelled.", "info")
    else:
        flash("This order can no longer be cancelled.", "warning")
    return redirect(url_for("marketplace.order_detail", order_number=order_number))


@bp.route("/orders/<order_number>/paid", methods=["POST"])
@farmer_required
def order_mark_paid(order_number):
    """Farmer confirms a UPI payment and enters the transaction reference."""
    order = Order.query.filter_by(order_number=order_number, farmer_id=current_farmer().id).first_or_404()
    ref = request.form.get("reference", "").strip()
    if order.payment_method == "UPI" and ref:
        order.payment_reference = ref[:120]
        order.payment_status = "Paid (pending verification)"
        db.session.commit()
        flash("Payment reference saved. We will verify it shortly.", "success")
    return redirect(url_for("marketplace.order_detail", order_number=order_number))
