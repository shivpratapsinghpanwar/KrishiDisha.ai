"""Admin console: farmer verification and management, orders, catalogue, activity audit."""
from __future__ import annotations

import re
from collections import Counter
from datetime import datetime, timedelta

from flask import Blueprint, abort, flash, jsonify, redirect, render_template, request, session, url_for
from sqlalchemy import or_
from sqlalchemy.exc import IntegrityError

from ..extensions import db
from ..models import (FEEDBACK_KINDS, ORDER_STATUSES, UNSURE, Admin, ChatMessage, ChatSession, Farmer,
                      FarmerActivity, Feedback, LabelTask, Order, Product)
from ..utils import admin_required
from .auth import INDIAN_STATES, LANGUAGES
from .marketplace import CATEGORIES

bp = Blueprint("admin", __name__)


# ------------------------------------------------------------------ dashboard
@bp.route("/admin_dashboard")
@admin_required
def dashboard():
    since = datetime.utcnow() - timedelta(days=30)
    farmers_total = Farmer.query.count()
    pending = Farmer.query.filter_by(verified=False).order_by(Farmer.created_at.desc()).all()
    orders_total = Order.query.count()
    revenue = db.session.query(db.func.coalesce(db.func.sum(Order.total), 0)) \
        .filter(Order.status != "Cancelled").scalar() or 0
    open_orders = Order.query.filter(Order.status.in_(["Placed", "Confirmed", "Packed"])).count()
    activities = FarmerActivity.query.filter(FarmerActivity.timestamp >= since, FarmerActivity.deleted.is_(False)).all()
    activity_counts = Counter(a.activity_type for a in activities)
    recent_activities = FarmerActivity.query.filter_by(deleted=False) \
        .order_by(FarmerActivity.timestamp.desc()).limit(10).all()
    recent_orders = Order.query.order_by(Order.created_at.desc()).limit(6).all()
    chats = ChatMessage.query.filter(ChatMessage.created_at >= since, ChatMessage.role == "user").count()
    label_queued = LabelTask.query.filter(LabelTask.status.in_(["queued", "needs_second"])).count()
    label_disagreements = LabelTask.query.filter_by(status="disagreement").count()
    feedback_pending = Feedback.query.filter_by(status="pending").count()
    low_stock = Product.query.filter(Product.is_active.is_(True), Product.stock <= 10).order_by(Product.stock).limit(8).all()

    # registrations per day for the last 14 days (for a small chart)
    days = [(datetime.utcnow() - timedelta(days=i)).date() for i in range(13, -1, -1)]
    reg_counts = Counter(f.created_at.date() for f in Farmer.query.filter(Farmer.created_at >= days[0]).all())
    signups = [{"date": d.strftime("%d %b"), "count": reg_counts.get(d, 0)} for d in days]

    return render_template("admin/dashboard.html", farmers_total=farmers_total, pending=pending,
                           orders_total=orders_total, revenue=revenue, open_orders=open_orders,
                           activity_counts=dict(activity_counts), recent_activities=recent_activities,
                           recent_orders=recent_orders, chats=chats, low_stock=low_stock, signups=signups,
                           label_queued=label_queued, label_disagreements=label_disagreements,
                           feedback_pending=feedback_pending)


# -------------------------------------------------------------------- farmers
@bp.route("/admin/farmers")
@admin_required
def farmers():
    q = request.args.get("q", "").strip()
    status = request.args.get("status", "")
    query = Farmer.query
    if q:
        like = f"%{q}%"
        query = query.filter(or_(Farmer.name.ilike(like), Farmer.email.ilike(like), Farmer.phone.ilike(like),
                                 Farmer.username.ilike(like), Farmer.district.ilike(like), Farmer.state.ilike(like)))
    if status == "pending":
        query = query.filter_by(verified=False)
    elif status == "verified":
        query = query.filter_by(verified=True)
    rows = query.order_by(Farmer.created_at.desc()).all()
    return render_template("admin/farmers.html", farmers=rows, q=q, status=status)


@bp.route("/verify_farmer/<int:farmer_id>", methods=["GET", "POST"])
@admin_required
def verify_farmer(farmer_id):
    farmer = db.session.get(Farmer, farmer_id) or abort(404)
    farmer.verified = True
    db.session.commit()
    flash(f"{farmer.name} verified. They can log in now.", "success")
    return redirect(request.referrer or url_for("admin.dashboard"))


@bp.route("/unverify_farmer/<int:farmer_id>", methods=["POST"])
@admin_required
def unverify_farmer(farmer_id):
    farmer = db.session.get(Farmer, farmer_id) or abort(404)
    farmer.verified = False
    db.session.commit()
    flash(f"{farmer.name} access suspended.", "warning")
    return redirect(request.referrer or url_for("admin.farmers"))


def _farmer_from_form(farmer: Farmer, f) -> list[str]:
    errors = []
    if not f.get("name", "").strip():
        errors.append("Name is required.")
    if not re.fullmatch(r"[6-9]\d{9}", f.get("phone", "").strip()):
        errors.append("Enter a valid 10-digit Indian mobile number.")
    if not re.fullmatch(r"[A-Za-z0-9_.]{3,45}", f.get("username", "")):
        errors.append("Username: 3-45 letters, digits, dot or underscore.")
    if "@" not in f.get("email", ""):
        errors.append("Enter a valid email.")
    if errors:
        return errors
    farmer.name = f["name"].strip()
    farmer.email = f["email"].strip().lower()
    farmer.phone = f["phone"].strip()
    farmer.username = f["username"].strip()
    farmer.state = f.get("state") or None
    farmer.district = f.get("district", "").strip() or None
    farmer.land_area_acres = float(f["land_area_acres"]) if f.get("land_area_acres") else None
    farmer.preferred_language = f.get("preferred_language", "en")
    farmer.verified = f.get("verified") == "on"
    if f.get("password"):
        if len(f["password"]) < 6:
            return ["Password must be at least 6 characters."]
        farmer.set_password(f["password"])
    return errors


@bp.route("/create_farmer", methods=["GET", "POST"])
@admin_required
def create_farmer():
    if request.method == "POST":
        farmer = Farmer()
        errors = _farmer_from_form(farmer, request.form)
        if not request.form.get("password"):
            errors.append("Password is required for a new farmer.")
        if errors:
            for e in errors:
                flash(e, "danger")
        else:
            try:
                db.session.add(farmer)
                db.session.commit()
                flash(f"Farmer {farmer.name} created.", "success")
                return redirect(url_for("admin.farmers"))
            except IntegrityError:
                db.session.rollback()
                flash("Username, email or phone already exists.", "danger")
        return render_template("admin/farmer_form.html", farmer=None, form=request.form, states=INDIAN_STATES,
                               languages=LANGUAGES, mode="create")
    return render_template("admin/farmer_form.html", farmer=None, form={}, states=INDIAN_STATES, languages=LANGUAGES,
                           mode="create")


@bp.route("/update_farmer/<int:farmer_id>", methods=["GET", "POST"])
@admin_required
def update_farmer(farmer_id):
    farmer = db.session.get(Farmer, farmer_id) or abort(404)
    if request.method == "POST":
        errors = _farmer_from_form(farmer, request.form)
        if errors:
            db.session.rollback()
            for e in errors:
                flash(e, "danger")
        else:
            try:
                db.session.commit()
                flash("Farmer updated.", "success")
                return redirect(url_for("admin.farmer_detail", farmer_id=farmer.id))
            except IntegrityError:
                db.session.rollback()
                flash("Username, email or phone already exists.", "danger")
    return render_template("admin/farmer_form.html", farmer=farmer, form={}, states=INDIAN_STATES, languages=LANGUAGES,
                           mode="edit")


@bp.route("/admin/farmers/<int:farmer_id>")
@admin_required
def farmer_detail(farmer_id):
    farmer = db.session.get(Farmer, farmer_id) or abort(404)
    activities = FarmerActivity.query.filter_by(farmer_id=farmer.id, deleted=False) \
        .order_by(FarmerActivity.timestamp.desc()).limit(50).all()
    orders = Order.query.filter_by(farmer_id=farmer.id).order_by(Order.created_at.desc()).all()
    sessions = ChatSession.query.filter_by(farmer_id=farmer.id).order_by(ChatSession.updated_at.desc()).limit(10).all()
    return render_template("admin/farmer_detail.html", farmer=farmer, activities=activities, orders=orders,
                           sessions=sessions)


@bp.route("/delete_farmer/<int:farmer_id>", methods=["POST"])
@admin_required
def delete_farmer(farmer_id):
    farmer = db.session.get(Farmer, farmer_id) or abort(404)
    db.session.delete(farmer)
    db.session.commit()
    flash("Farmer and all their data deleted.", "info")
    return redirect(url_for("admin.farmers"))


# --------------------------------------------------------------------- orders
@bp.route("/admin/orders")
@admin_required
def orders():
    status = request.args.get("status", "")
    query = Order.query
    if status:
        query = query.filter_by(status=status)
    rows = query.order_by(Order.created_at.desc()).limit(300).all()
    counts = dict(db.session.query(Order.status, db.func.count(Order.id)).group_by(Order.status).all())
    return render_template("admin/orders.html", orders=rows, statuses=ORDER_STATUSES, status=status, counts=counts)


@bp.route("/admin/orders/<int:order_id>", methods=["GET", "POST"])
@admin_required
def order_detail(order_id):
    order = db.session.get(Order, order_id) or abort(404)
    if request.method == "POST":
        new_status = request.form.get("status")
        if new_status in ORDER_STATUSES:
            if new_status == "Cancelled" and order.status != "Cancelled":
                for it in order.items:
                    if it.product:
                        it.product.stock += it.quantity
            order.status = new_status
        pay = request.form.get("payment_status")
        if pay:
            order.payment_status = pay[:30]
        db.session.commit()
        flash("Order updated.", "success")
        return redirect(url_for("admin.order_detail", order_id=order.id))
    return render_template("admin/order_detail.html", order=order, statuses=ORDER_STATUSES,
                           payment_statuses=["Pending", "Awaiting UPI", "Paid (pending verification)", "Paid", "Refunded"])


# ------------------------------------------------------------------- products
@bp.route("/admin/products")
@admin_required
def products():
    q = request.args.get("q", "").strip()
    category = request.args.get("category", "")
    query = Product.query
    if q:
        like = f"%{q}%"
        query = query.filter(or_(Product.name.ilike(like), Product.brand.ilike(like), Product.slug.ilike(like)))
    if category:
        query = query.filter_by(category=category)
    rows = query.order_by(Product.is_active.desc(), Product.category, Product.name).all()
    return render_template("admin/products.html", products=rows, q=q, category=category, categories=CATEGORIES)


def _slugify(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:110] or "product"


def _product_from_form(p: Product, f) -> list[str]:
    errors = []
    name = f.get("name", "").strip()
    if not name:
        errors.append("Name is required.")
    try:
        price = float(f.get("price", ""))
        if price <= 0:
            errors.append("Price must be positive.")
    except ValueError:
        errors.append("Price must be a number.")
    if errors:
        return errors
    p.name = name
    p.category = f.get("category") or "Fertilizer"
    p.brand = f.get("brand", "").strip() or None
    p.description = f.get("description", "").strip() or None
    p.price = price
    p.mrp = float(f["mrp"]) if f.get("mrp") else None
    p.unit = f.get("unit", "").strip() or "pack"
    p.stock = int(f.get("stock") or 0)
    p.image_url = f.get("image_url", "").strip() or None
    p.external_url = f.get("external_url", "").strip() or None
    p.npk = f.get("npk", "").strip() or None
    p.disease_tags = ",".join(t.strip() for t in f.get("disease_tags", "").split(",") if t.strip())
    p.crop_tags = ",".join(t.strip().lower() for t in f.get("crop_tags", "").split(",") if t.strip())
    p.is_organic = f.get("is_organic") == "on"
    p.is_active = f.get("is_active", "on") == "on"
    p.rating = float(f["rating"]) if f.get("rating") else p.rating
    if not p.slug:
        base = _slugify(name)
        slug, i = base, 2
        while Product.query.filter_by(slug=slug).first():
            slug = f"{base}-{i}"
            i += 1
        p.slug = slug
    return errors


@bp.route("/admin/products/new", methods=["GET", "POST"])
@admin_required
def product_new():
    if request.method == "POST":
        p = Product()
        errors = _product_from_form(p, request.form)
        if errors:
            for e in errors:
                flash(e, "danger")
            return render_template("admin/product_form.html", product=None, form=request.form, categories=CATEGORIES)
        db.session.add(p)
        db.session.commit()
        flash(f"Product '{p.name}' added.", "success")
        return redirect(url_for("admin.products"))
    return render_template("admin/product_form.html", product=None, form={}, categories=CATEGORIES)


@bp.route("/admin/products/<int:product_id>/edit", methods=["GET", "POST"])
@admin_required
def product_edit(product_id):
    p = db.session.get(Product, product_id) or abort(404)
    if request.method == "POST":
        errors = _product_from_form(p, request.form)
        if errors:
            db.session.rollback()
            for e in errors:
                flash(e, "danger")
        else:
            db.session.commit()
            flash("Product updated.", "success")
            return redirect(url_for("admin.products"))
    return render_template("admin/product_form.html", product=p, form={}, categories=CATEGORIES)


@bp.route("/admin/products/<int:product_id>/toggle", methods=["POST"])
@admin_required
def product_toggle(product_id):
    p = db.session.get(Product, product_id) or abort(404)
    p.is_active = not p.is_active
    db.session.commit()
    flash(f"{p.name} is now {'visible' if p.is_active else 'hidden'}.", "info")
    return redirect(request.referrer or url_for("admin.products"))


# ------------------------------------------------------------------- activity
@bp.route("/admin/activities")
@admin_required
def activities():
    kind = request.args.get("type", "")
    query = FarmerActivity.query.filter_by(deleted=False)
    if kind:
        query = query.filter_by(activity_type=kind)
    rows = query.order_by(FarmerActivity.timestamp.desc()).limit(300).all()
    kinds = [k for (k,) in db.session.query(FarmerActivity.activity_type).distinct().all()]
    return render_template("admin/activities.html", activities=rows, kinds=sorted(kinds), kind=kind)


@bp.route("/admin/chats")
@admin_required
def chats():
    sessions = ChatSession.query.order_by(ChatSession.updated_at.desc()).limit(100).all()
    return render_template("admin/chats.html", sessions=sessions)


@bp.route("/admin/chats/<int:session_id>")
@admin_required
def chat_detail(session_id):
    s = db.session.get(ChatSession, session_id) or abort(404)
    return render_template("admin/chat_detail.html", chat=s)


# -------------------------------------------------------------- admin account
@bp.route("/create_admin", methods=["GET", "POST"])
@admin_required
def create_admin():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if not re.fullmatch(r"[A-Za-z0-9_.]{3,45}", username) or len(password) < 6:
            flash("Username 3-45 chars and password at least 6 characters.", "danger")
        elif Admin.query.filter_by(username=username).first():
            flash("That admin username already exists.", "danger")
        else:
            admin = Admin(username=username)
            admin.set_password(password)
            db.session.add(admin)
            db.session.commit()
            flash(f"Admin '{username}' created.", "success")
            return redirect(url_for("admin.dashboard"))
    admins = Admin.query.order_by(Admin.created_at).all()
    return render_template("admin/create_admin.html", admins=admins)


# --------------------------------------------------------------- labelling
def _admin_name() -> str:
    """Identify the labeller. Falls back to the admin id so two admins never share a name."""
    admin = db.session.get(Admin, session.get("admin_id")) if session.get("admin_id") else None
    return admin.username if admin else f"admin:{session.get('admin_id', 'unknown')}"


def _label_counts() -> dict:
    rows = db.session.query(LabelTask.status, db.func.count(LabelTask.id)).group_by(LabelTask.status).all()
    counts = {status: n for status, n in rows}
    counts["open"] = counts.get("queued", 0) + counts.get("needs_second", 0)
    return counts


def _skipped() -> list[int]:
    return session.get("label_skipped", [])


@bp.route("/admin/label", methods=["GET", "POST"])
@admin_required
def label_queue():
    from .feedback import canonical_labels, grouped_labels, image_url, not_a_leaf_label

    me = _admin_name()
    if request.method == "POST":
        action = (request.form.get("action") or "label").strip()
        task = db.session.get(LabelTask, request.form.get("task_id", type=int))
        if task is None:
            flash("That labelling task no longer exists.", "warning")
        elif action == "skip":
            session["label_skipped"] = (_skipped() + [task.id])[-100:]
        elif action == "discard":
            task.status = "discarded"
            task.notes = ((task.notes or "") + f" discarded by {me};").strip()
            db.session.commit()
            flash("Photo discarded.", "info")
        else:
            label = {"not_leaf": not_a_leaf_label(), "unsure": UNSURE}.get(action) or \
                (request.form.get("label") or "").strip()
            if not label:
                flash("Choose a class first.", "warning")
            elif task.labeller_1 == me or task.labeller_2 == me:
                flash("You already labelled that photo - it needs a second labeller.", "warning")
            else:
                status = task.record_label(label, me)
                db.session.commit()
                flash({"needs_second": "Saved - waiting for a second labeller.",
                       "labelled": "Agreed! Photo is ready for export.",
                       "disagreement": "Labels disagree - sent to the disagreement queue.",
                       "discarded": "Both labellers were unsure - discarded."}.get(status, "Saved."), "success")
        return redirect(url_for("admin.label_queue", n=request.form.get("n", 1)))

    n = min(max(request.args.get("n", 1, type=int) or 1, 1), 12)
    query = LabelTask.query.filter(LabelTask.status.in_(["queued", "needs_second"])) \
        .filter(or_(LabelTask.labeller_1.is_(None), LabelTask.labeller_1 != me)) \
        .filter(or_(LabelTask.labeller_2.is_(None), LabelTask.labeller_2 != me))
    skipped = _skipped()
    if skipped:
        query = query.filter(~LabelTask.id.in_(skipped))
    tasks = query.order_by(LabelTask.status.asc(), LabelTask.created_at.asc()).limit(n).all()
    labels = canonical_labels()
    return render_template("admin/label.html", tasks=tasks, labels=labels, grouped=grouped_labels(labels),
                           counts=_label_counts(), n=n, me=me, image_url=image_url,
                           not_a_leaf=not_a_leaf_label(), skipped=len(skipped))


@bp.route("/admin/label/reset_skips", methods=["POST"])
@admin_required
def label_reset_skips():
    session.pop("label_skipped", None)
    flash("Skipped photos are back in the queue.", "info")
    return redirect(url_for("admin.label_queue"))


@bp.route("/admin/label/disagreements", methods=["GET", "POST"])
@admin_required
def label_disagreements():
    from .feedback import canonical_labels, grouped_labels, image_url

    if request.method == "POST":
        task = db.session.get(LabelTask, request.form.get("task_id", type=int))
        if task is None:
            flash("That labelling task no longer exists.", "warning")
        else:
            task.resolve(request.form.get("final_label", ""), _admin_name())
            db.session.commit()
            flash(f"Task #{task.id} resolved as {task.final_label or 'discarded'}.", "success")
        return redirect(url_for("admin.label_disagreements"))

    tasks = LabelTask.query.filter_by(status="disagreement").order_by(LabelTask.updated_at.asc()).limit(100).all()
    labels = canonical_labels()
    return render_template("admin/label_disagreements.html", tasks=tasks, labels=labels,
                           grouped=grouped_labels(labels), counts=_label_counts(), image_url=image_url)


@bp.route("/admin/label/upload", methods=["GET", "POST"])
@admin_required
def label_upload():
    from .feedback import ALLOWED_EXT, canonical_labels, create_label_task, grouped_labels, store_bytes

    if request.method == "POST":
        me = _admin_name()
        label = (request.form.get("label") or "").strip()
        trusted = request.form.get("trust_label") == "on" and label and label != UNSURE
        saved, rejected = 0, 0
        for file in request.files.getlist("images"):
            if not file or not file.filename:
                continue
            ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
            data = file.read()
            if ext not in ALLOWED_EXT or not data:
                rejected += 1
                continue
            rel = store_bytes(data, "drive", ext)
            task = create_label_task(rel, "drive", model_label=label or None, consent=True, commit=False,
                                     notes=f"collection drive upload by {me}")
            if trusted:
                task.label_1 = task.label_2 = task.final_label = label
                task.labeller_1 = task.labeller_2 = me
                task.status = "labelled"
            saved += 1
        db.session.commit()
        flash(f"{saved} photo(s) added{' as labelled ' + label if trusted else ' to the labelling queue'}."
              + (f" {rejected} file(s) rejected." if rejected else ""), "success" if saved else "warning")
        return redirect(url_for("admin.label_upload"))

    labels = canonical_labels()
    return render_template("admin/label_upload.html", labels=labels, grouped=grouped_labels(labels),
                           counts=_label_counts())


@bp.route("/admin/label/export", methods=["POST"])
@admin_required
def label_export():
    from ml.datasets.export_feedback import run_export

    try:
        summary = run_export()
    except Exception as exc:  # noqa: BLE001 - surface the reason instead of a 500
        db.session.rollback()
        flash(f"Export failed: {exc}", "danger")
        return redirect(request.referrer or url_for("admin.label_queue"))
    if request.is_json or request.args.get("format") == "json":
        return jsonify(summary)
    flash(f"Exported {summary['exported']} photo(s) to {summary['out_dir']} "
          f"({summary['chat_turns']} rated chat turns written).", "success")
    return redirect(request.referrer or url_for("admin.label_queue"))


# ---------------------------------------------------------------- feedback
@bp.route("/admin/feedback", methods=["GET", "POST"])
@admin_required
def feedback_review():
    from .feedback import create_label_task, image_url

    if request.method == "POST":
        row = db.session.get(Feedback, request.form.get("feedback_id", type=int))
        action = request.form.get("action", "")
        if row is None or action not in ("accept", "reject"):
            flash("Nothing to do.", "warning")
            return redirect(request.referrer or url_for("admin.feedback_review"))
        me = _admin_name()
        row.status = "accepted" if action == "accept" else "rejected"
        row.reviewer, row.reviewed_at = me, datetime.utcnow()
        if action == "accept" and row.kind == "diagnosis" and row.corrected_label and \
                row.corrected_label != UNSURE and row.image_path:
            task = LabelTask.query.filter_by(image_path=row.image_path).first()
            if task is None:
                task = create_label_task(row.image_path, "app_upload", model_label=row.model_label,
                                         model_confidence=row.model_confidence, consent=bool(row.consent),
                                         commit=False, notes=f"from feedback #{row.id}")
            task.final_label = row.corrected_label
            task.status = "labelled"
            task.label_1 = task.label_1 or row.corrected_label
            task.labeller_1 = task.labeller_1 or "farmer"
            task.label_2, task.labeller_2 = row.corrected_label, me
        db.session.commit()
        flash(f"Feedback #{row.id} {row.status}.", "success")
        return redirect(request.referrer or url_for("admin.feedback_review"))

    kind = request.args.get("kind", "")
    status = request.args.get("status", "")
    query = Feedback.query
    if kind:
        query = query.filter_by(kind=kind)
    if status:
        query = query.filter_by(status=status)
    rows = query.order_by(Feedback.created_at.desc()).limit(300).all()
    counts = dict(db.session.query(Feedback.status, db.func.count(Feedback.id)).group_by(Feedback.status).all())
    return render_template("admin/feedback.html", rows=rows, kind=kind, status=status, kinds=FEEDBACK_KINDS,
                           counts=counts, image_url=image_url)
