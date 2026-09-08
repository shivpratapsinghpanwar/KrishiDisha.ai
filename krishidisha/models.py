"""SQLAlchemy models.

Farmer / Admin / FarmerActivity mirror the schema in the KrishiDisha report.
Product / CartItem / Order / OrderItem / Address implement the marketplace,
and ChatSession / ChatMessage persist the assistant conversations.
"""
import json
from datetime import datetime

from werkzeug.security import check_password_hash, generate_password_hash

from .extensions import db


class Farmer(db.Model):
    __tablename__ = "farmer"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    phone = db.Column(db.String(15), unique=True, nullable=False)
    username = db.Column(db.String(60), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    verified = db.Column(db.Boolean, default=False, nullable=False)
    # Profile fields used to personalise recommendations and the assistant.
    state = db.Column(db.String(60))
    district = db.Column(db.String(80))
    land_area_acres = db.Column(db.Float)
    preferred_language = db.Column(db.String(10), default="en")
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    activities = db.relationship("FarmerActivity", backref="farmer", lazy=True, cascade="all, delete-orphan")
    cart_items = db.relationship("CartItem", backref="farmer", lazy=True, cascade="all, delete-orphan")
    orders = db.relationship("Order", backref="farmer", lazy=True, cascade="all, delete-orphan")
    addresses = db.relationship("Address", backref="farmer", lazy=True, cascade="all, delete-orphan")
    chat_sessions = db.relationship("ChatSession", backref="farmer", lazy=True, cascade="all, delete-orphan")

    def set_password(self, password: str) -> None:
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)

    def profile_summary(self) -> str:
        parts = [f"name: {self.name}"]
        if self.state:
            parts.append(f"state: {self.state}")
        if self.district:
            parts.append(f"district: {self.district}")
        if self.land_area_acres:
            parts.append(f"land: {self.land_area_acres} acres")
        if self.preferred_language:
            parts.append(f"preferred language: {self.preferred_language}")
        return ", ".join(parts)


class Admin(db.Model):
    __tablename__ = "admin"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(60), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    def set_password(self, password: str) -> None:
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)


class FarmerActivity(db.Model):
    __tablename__ = "farmer_activity"

    id = db.Column(db.Integer, primary_key=True)
    farmer_id = db.Column(db.Integer, db.ForeignKey("farmer.id"), nullable=False, index=True)
    activity_type = db.Column(db.String(50), nullable=False)
    input_data = db.Column(db.Text, nullable=False)
    output_data = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    deleted = db.Column(db.Boolean, default=False, nullable=False)

    def input_dict(self) -> dict:
        try:
            return json.loads(self.input_data)
        except (TypeError, ValueError):
            return {"raw": self.input_data}

    def output_dict(self) -> dict:
        try:
            return json.loads(self.output_data)
        except (TypeError, ValueError):
            return {"raw": self.output_data}


# --------------------------------------------------------------------------- #
# Marketplace
# --------------------------------------------------------------------------- #
class Product(db.Model):
    __tablename__ = "product"

    id = db.Column(db.Integer, primary_key=True)
    slug = db.Column(db.String(120), unique=True, nullable=False, index=True)
    name = db.Column(db.String(200), nullable=False)
    category = db.Column(db.String(60), nullable=False, index=True)  # Fertilizer, Fungicide, Insecticide, Seeds, Organic, Tools
    brand = db.Column(db.String(120))
    description = db.Column(db.Text)
    price = db.Column(db.Float, nullable=False)
    mrp = db.Column(db.Float)
    unit = db.Column(db.String(40), default="pack")  # e.g. "50 kg bag", "250 ml"
    stock = db.Column(db.Integer, default=100, nullable=False)
    image_url = db.Column(db.String(500))
    external_url = db.Column(db.String(500))  # original vendor link from supplement_info.csv
    npk = db.Column(db.String(20))  # e.g. "46-0-0"
    # Comma-separated tags used by disease detection and the assistant to find products.
    disease_tags = db.Column(db.Text, default="")
    crop_tags = db.Column(db.Text, default="")
    is_organic = db.Column(db.Boolean, default=False)
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    rating = db.Column(db.Float, default=4.3)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    def discount_percent(self) -> int:
        if self.mrp and self.mrp > self.price:
            return int(round((self.mrp - self.price) / self.mrp * 100))
        return 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "slug": self.slug,
            "name": self.name,
            "category": self.category,
            "brand": self.brand,
            "price": self.price,
            "mrp": self.mrp,
            "unit": self.unit,
            "stock": self.stock,
            "image_url": self.image_url,
            "npk": self.npk,
            "is_organic": self.is_organic,
            "rating": self.rating,
            "disease_tags": [t for t in (self.disease_tags or "").split(",") if t],
            "crop_tags": [t for t in (self.crop_tags or "").split(",") if t],
        }


class CartItem(db.Model):
    __tablename__ = "cart_item"

    id = db.Column(db.Integer, primary_key=True)
    farmer_id = db.Column(db.Integer, db.ForeignKey("farmer.id"), nullable=False, index=True)
    product_id = db.Column(db.Integer, db.ForeignKey("product.id"), nullable=False)
    quantity = db.Column(db.Integer, default=1, nullable=False)
    added_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    product = db.relationship("Product")

    __table_args__ = (db.UniqueConstraint("farmer_id", "product_id", name="uq_cart_farmer_product"),)

    @property
    def line_total(self) -> float:
        return round(self.product.price * self.quantity, 2)


class Address(db.Model):
    __tablename__ = "address"

    id = db.Column(db.Integer, primary_key=True)
    farmer_id = db.Column(db.Integer, db.ForeignKey("farmer.id"), nullable=False, index=True)
    full_name = db.Column(db.String(120), nullable=False)
    phone = db.Column(db.String(15), nullable=False)
    line1 = db.Column(db.String(200), nullable=False)
    village = db.Column(db.String(120))
    district = db.Column(db.String(120), nullable=False)
    state = db.Column(db.String(60), nullable=False)
    pincode = db.Column(db.String(10), nullable=False)

    def formatted(self) -> str:
        bits = [self.full_name, self.line1, self.village, self.district, f"{self.state} - {self.pincode}", f"Phone: {self.phone}"]
        return ", ".join(b for b in bits if b)


ORDER_STATUSES = ["Placed", "Confirmed", "Packed", "Shipped", "Delivered", "Cancelled"]


class Order(db.Model):
    __tablename__ = "order"

    id = db.Column(db.Integer, primary_key=True)
    order_number = db.Column(db.String(30), unique=True, nullable=False, index=True)
    farmer_id = db.Column(db.Integer, db.ForeignKey("farmer.id"), nullable=False, index=True)
    address_text = db.Column(db.Text, nullable=False)
    subtotal = db.Column(db.Float, nullable=False)
    delivery_charge = db.Column(db.Float, default=0.0, nullable=False)
    total = db.Column(db.Float, nullable=False)
    payment_method = db.Column(db.String(30), default="COD", nullable=False)  # COD | UPI | Razorpay
    payment_status = db.Column(db.String(30), default="Pending", nullable=False)
    payment_reference = db.Column(db.String(120))
    status = db.Column(db.String(30), default="Placed", nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    items = db.relationship("OrderItem", backref="order", lazy=True, cascade="all, delete-orphan")

    def to_dict(self) -> dict:
        return {
            "order_number": self.order_number,
            "status": self.status,
            "total": self.total,
            "payment_method": self.payment_method,
            "payment_status": self.payment_status,
            "created_at": self.created_at.isoformat(),
            "items": [{"name": i.product_name, "qty": i.quantity, "price": i.unit_price} for i in self.items],
        }


class OrderItem(db.Model):
    __tablename__ = "order_item"

    id = db.Column(db.Integer, primary_key=True)
    order_id = db.Column(db.Integer, db.ForeignKey("order.id"), nullable=False, index=True)
    product_id = db.Column(db.Integer, db.ForeignKey("product.id"))
    product_name = db.Column(db.String(200), nullable=False)
    unit_price = db.Column(db.Float, nullable=False)
    quantity = db.Column(db.Integer, nullable=False)

    product = db.relationship("Product")

    @property
    def line_total(self) -> float:
        return round(self.unit_price * self.quantity, 2)


# --------------------------------------------------------------------------- #
# Assistant conversations
# --------------------------------------------------------------------------- #
class ChatSession(db.Model):
    __tablename__ = "chat_session"

    id = db.Column(db.Integer, primary_key=True)
    farmer_id = db.Column(db.Integer, db.ForeignKey("farmer.id"), nullable=True, index=True)
    session_key = db.Column(db.String(64), unique=True, nullable=False, index=True)
    title = db.Column(db.String(200), default="New conversation")
    language = db.Column(db.String(10), default="en")
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    messages = db.relationship("ChatMessage", backref="session", lazy=True, cascade="all, delete-orphan",
                               order_by="ChatMessage.id")


class ChatMessage(db.Model):
    __tablename__ = "chat_message"

    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey("chat_session.id"), nullable=False, index=True)
    role = db.Column(db.String(12), nullable=False)  # user | assistant
    content = db.Column(db.Text, nullable=False)
    tools_used = db.Column(db.Text)  # JSON list of tool names invoked for this reply
    provider = db.Column(db.String(30))
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
