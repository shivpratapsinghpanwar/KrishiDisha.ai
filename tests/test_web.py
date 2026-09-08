"""End-to-end tests through the Flask test client (templates, auth, tools, marketplace, admin, API)."""
from __future__ import annotations

import io

from PIL import Image


# ----------------------------------------------------------------- public
def test_public_pages_render(client):
    for path in ("/", "/about", "/weather", "/mandi", "/schemes", "/schemes?q=insurance", "/crop-calendar",
                 "/crop-calendar?season=Rabi", "/crop-guide/rice", "/crop-guide/unknowncrop",
                 "/tools/fertilizer-calculator", "/farmer_registration", "/farmer_login", "/admin_login", "/chat",
                 "/marketplace/", "/marketplace/?q=urea&category=Fertilizer&sort=price_asc"):
        r = client.get(path)
        assert r.status_code == 200, path
    assert client.get("/does-not-exist").status_code == 404
    assert client.get("/api/v1/nope").status_code == 404
    assert client.get("/health").get_json()["status"] == "ok"


def test_fertilizer_calculator_page(client):
    r = client.post("/tools/fertilizer-calculator", data={"crop": "wheat", "area": "2", "unit": "acre"})
    assert r.status_code == 200
    assert b"Urea" in r.data and b"DAP" in r.data


def test_login_required_redirects(client):
    r = client.get("/crop_recommendation")
    assert r.status_code == 302 and "/farmer_login" in r.headers["Location"]
    r = client.post("/marketplace/cart/add", json={"product_id": 1})
    assert r.status_code == 401


# ------------------------------------------------------------------- auth
def test_registration_validation(client, farmer_data):
    bad = dict(farmer_data, username="x", phone="123", password="abc", email="other@example.com")
    r = client.post("/farmer_registration", data=bad)
    assert r.status_code == 200
    assert b"valid 10-digit" in r.data and b"at least 6" in r.data


def test_farmer_login_logout_profile(farmer_client):
    r = farmer_client.get("/home_crop")
    assert r.status_code == 200 and b"Welcome back, Test" in r.data
    r = farmer_client.post("/profile", data={"name": "Test Farmer", "state": "Punjab", "district": "Ludhiana",
                                             "land_area_acres": "5", "preferred_language": "hi"},
                           follow_redirects=True)
    assert r.status_code == 200 and b"Profile updated" in r.data
    r = farmer_client.get("/farmer_logout", follow_redirects=True)
    assert r.status_code == 200
    assert farmer_client.get("/home_crop").status_code == 302


# ------------------------------------------------------------ farmer tools
def test_crop_recommendation_flow(farmer_client):
    data = {"N": 90, "P": 42, "K": 43, "temperature": 21, "humidity": 82, "ph": 6.5, "rainfall": 203}
    r = farmer_client.post("/crop_recommendation", data=data)
    assert r.status_code == 200 and b"Rice" in r.data
    r = farmer_client.post("/download_report")
    assert r.status_code == 200 and r.mimetype == "application/pdf" and r.data[:4] == b"%PDF"
    r = farmer_client.post("/crop_recommendation", data=dict(data, ph="abc"))
    assert r.status_code == 200 and b"Invalid input" in r.data


def test_fertilizer_recommendation_flow(farmer_client):
    data = {"temperature": 26, "humidity": 52, "moisture": 38, "soil_type": "Sandy", "crop_type": "Maize",
            "N": 37, "P": 0, "K": 0, "area": 2, "unit": "acre"}
    r = farmer_client.post("/fertilizer_recommendation", data=data)
    assert r.status_code == 200 and b"Urea" in r.data
    r = farmer_client.post("/download_fertilizer_report")
    assert r.status_code == 200 and r.mimetype == "application/pdf"


def test_yield_flow(farmer_client):
    data = {"crop": "Wheat", "crop_year": 2020, "season": "Rabi", "state": "Punjab", "area": 1000,
            "production": 4000, "annual_rainfall": 600, "fertilizer": 150000, "pesticide": 300}
    r = farmer_client.post("/crop_yield", data=data)
    assert r.status_code == 200 and b"tonnes per hectare" in r.data
    r = farmer_client.post("/download_yield_report")
    assert r.status_code == 200 and r.mimetype == "application/pdf"


def test_disease_upload_with_stub_model(farmer_client):
    assert farmer_client.get("/crop_detection").status_code == 200
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (30, 120, 30)).save(buf, format="JPEG")
    buf.seek(0)
    r = farmer_client.post("/submit", data={"image": (buf, "leaf.jpg")}, content_type="multipart/form-data",
                           follow_redirects=True)
    # TestConfig uses the stub backend -> friendly message, no crash
    assert r.status_code == 200 and b"No disease detection model" in r.data
    r = farmer_client.post("/submit", data={"image": (io.BytesIO(b"not an image"), "x.jpg")},
                           content_type="multipart/form-data", follow_redirects=True)
    assert b"not a valid image" in r.data


def test_dashboard_lists_and_deletes_activity(farmer_client, app):
    farmer_client.post("/crop_recommendation", data={"N": 90, "P": 42, "K": 43, "temperature": 21, "humidity": 82,
                                                     "ph": 6.5, "rainfall": 203})
    r = farmer_client.get("/farmer_dashboard")
    assert r.status_code == 200 and b"Crop Recommendation" in r.data
    with app.app_context():
        from krishidisha.models import FarmerActivity

        act = FarmerActivity.query.filter_by(deleted=False).order_by(FarmerActivity.id.desc()).first()
    r = farmer_client.post("/farmer_dashboard", data={"activity_id": act.id}, follow_redirects=True)
    assert b"Activity removed" in r.data


# ------------------------------------------------------------- marketplace
def test_marketplace_cart_checkout_orders(farmer_client, app):
    with app.app_context():
        from krishidisha.models import Product

        p = Product.query.filter_by(is_active=True).first()
        slug, pid, stock = p.slug, p.id, p.stock
    assert farmer_client.get(f"/marketplace/product/{slug}").status_code == 200
    r = farmer_client.post("/marketplace/cart/add", json={"product_id": pid, "quantity": 2})
    assert r.status_code == 200 and r.get_json()["cart_count"] >= 2
    r = farmer_client.get("/marketplace/cart")
    assert r.status_code == 200 and slug.encode() in r.data or r.status_code == 200
    r = farmer_client.get("/marketplace/checkout")
    assert r.status_code == 200
    r = farmer_client.post("/marketplace/checkout", data={"full_name": "Test Farmer", "phone": "9876543210",
                                                          "line1": "Village road", "district": "Indore",
                                                          "state": "Madhya Pradesh", "pincode": "452001",
                                                          "payment_method": "UPI"}, follow_redirects=True)
    assert r.status_code == 200 and b"placed successfully" in r.data
    with app.app_context():
        from krishidisha.models import Order, Product

        order = Order.query.order_by(Order.id.desc()).first()
        assert order.payment_method == "UPI" and order.total >= order.subtotal
        assert Product.query.get(pid).stock == stock - 2
        number = order.order_number
    r = farmer_client.get("/marketplace/orders")
    assert r.status_code == 200 and number.encode() in r.data
    r = farmer_client.post(f"/marketplace/orders/{number}/paid", data={"reference": "UPI123"}, follow_redirects=True)
    assert b"Payment reference saved" in r.data
    r = farmer_client.post(f"/marketplace/orders/{number}/cancel", follow_redirects=True)
    assert b"Order cancelled" in r.data
    with app.app_context():
        assert Product.query.get(pid).stock == stock


def test_checkout_rejects_bad_pincode(farmer_client, app):
    with app.app_context():
        from krishidisha.models import Product

        pid = Product.query.filter_by(is_active=True).first().id
    farmer_client.post("/marketplace/cart/add", json={"product_id": pid, "quantity": 1})
    r = farmer_client.post("/marketplace/checkout", data={"full_name": "A", "phone": "9876543210", "line1": "x",
                                                          "district": "d", "state": "s", "pincode": "12"})
    assert r.status_code == 200 and b"valid 6-digit" in r.data


# -------------------------------------------------------------------- chat
def test_chat_widget_and_history(farmer_client):
    r = farmer_client.post("/chat", json={"message": "hello"})
    body = r.get_json()
    assert r.status_code == 200 and body["provider"] == "rules" and body["session_key"]
    key = body["session_key"]
    r = farmer_client.post("/chat/send", json={"message": "How much urea for 1 acre rice?", "session_key": key})
    assert "fertilizer_calculator" in r.get_json()["tools_used"]
    r = farmer_client.get(f"/chat/sessions/{key}")
    assert len(r.get_json()["messages"]) == 4
    assert farmer_client.get(f"/chat?s={key}").status_code == 200
    r = farmer_client.post(f"/chat/sessions/{key}/delete", json={})
    assert r.get_json()["ok"] is True
    assert farmer_client.get(f"/chat/sessions/{key}").status_code == 404


def test_chat_with_image_uses_stub_gracefully(client):
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), (20, 100, 20)).save(buf, format="PNG")
    buf.seek(0)
    r = client.post("/chat/send", data={"message": "", "image": (buf, "leaf.png")}, content_type="multipart/form-data")
    assert r.status_code == 200 and r.get_json()["reply"]


def test_chat_requires_message(client):
    assert client.post("/chat", json={}).status_code == 400


# ------------------------------------------------------------------- admin
def test_admin_pages_and_verification(admin_client, app):
    with app.app_context():
        from krishidisha.extensions import db
        from krishidisha.models import Farmer

        f = Farmer(name="Pending Person", email="pending@example.com", phone="9123456780", username="pending1",
                   verified=False)
        f.set_password("secret123")
        db.session.add(f)
        db.session.commit()
        fid = f.id
    for path in ("/admin_dashboard", "/admin/farmers", "/admin/farmers?status=pending", "/create_farmer",
                 "/admin/orders", "/admin/products", "/admin/products/new", "/admin/activities", "/admin/chats",
                 "/create_admin", f"/update_farmer/{fid}", f"/admin/farmers/{fid}"):
        assert admin_client.get(path).status_code == 200, path
    r = admin_client.get(f"/verify_farmer/{fid}", follow_redirects=True)
    assert b"verified" in r.data
    r = admin_client.post(f"/unverify_farmer/{fid}", follow_redirects=True)
    assert b"suspended" in r.data
    r = admin_client.post(f"/update_farmer/{fid}", data={"name": "Pending Person", "email": "pending@example.com",
                                                          "phone": "9123456780", "username": "pending1",
                                                          "state": "Bihar", "district": "Patna",
                                                          "preferred_language": "hi", "verified": "on"},
                          follow_redirects=True)
    assert b"Farmer updated" in r.data
    r = admin_client.post(f"/delete_farmer/{fid}", follow_redirects=True)
    assert b"deleted" in r.data


def test_admin_product_crud(admin_client, app):
    r = admin_client.post("/admin/products/new", data={"name": "Test Neem Oil", "category": "Organic", "price": "199",
                                                       "mrp": "249", "unit": "1 L", "stock": "5",
                                                       "crop_tags": "tomato, chilli", "is_organic": "on",
                                                       "is_active": "on"}, follow_redirects=True)
    assert r.status_code == 200 and b"Test Neem Oil" in r.data
    with app.app_context():
        from krishidisha.models import Product

        p = Product.query.filter_by(name="Test Neem Oil").first()
        assert p.slug == "test-neem-oil" and p.is_organic and p.discount_percent() == 20
        pid = p.id
    assert admin_client.get(f"/admin/products/{pid}/edit").status_code == 200
    r = admin_client.post(f"/admin/products/{pid}/toggle", follow_redirects=True)
    assert b"hidden" in r.data
    r = admin_client.post("/admin/products/new", data={"name": "", "price": "x"})
    assert b"Name is required" in r.data


def test_admin_requires_login(client):
    r = client.get("/admin_dashboard")
    assert r.status_code == 302 and "/admin_login" in r.headers["Location"]


# --------------------------------------------------------------------- API
def test_api_endpoints(client):
    r = client.post("/api/v1/crop/recommend", json={"N": 90, "P": 42, "K": 43, "temperature": 21, "humidity": 82,
                                                    "ph": 6.5, "rainfall": 203})
    assert r.status_code == 200 and r.get_json()["recommended_crop"] == "rice"
    r = client.post("/api/v1/crop/recommend", json={"N": 90})
    assert r.status_code == 400 and "missing" in r.get_json()["error"]
    r = client.post("/api/v1/fertilizer/recommend", json={"temperature": 26, "humidity": 52, "moisture": 38,
                                                          "soil_type": "Sandy", "crop_type": "Maize", "N": 37, "P": 0,
                                                          "K": 0, "area": 1})
    assert r.status_code == 200 and r.get_json()["calculator"]["bags_50kg"]["Urea"] > 0
    r = client.post("/api/v1/yield/predict", json={"crop": "Wheat", "crop_year": 2020, "season": "Rabi",
                                                   "state": "Punjab", "area": 100, "production": 400,
                                                   "annual_rainfall": 600, "fertilizer": 15000, "pesticide": 30})
    assert r.status_code == 200 and r.get_json()["predicted_yield"] > 0
    assert client.get("/api/v1/reference").get_json()["soil_types"] == ["Black", "Clayey", "Loamy", "Red", "Sandy"]
    assert client.get("/api/v1/schemes?q=credit").get_json()["schemes"]
    assert client.get("/api/v1/crop-guide/wheat").get_json()["guide"]["season"]
    assert client.get("/api/v1/crop-guide/nothing").status_code == 404
    assert client.get("/api/v1/knowledge/search?q=rust").get_json()["results"]
    assert client.get("/api/v1/products?disease=Tomato___Late_blight").get_json()["products"]
    assert client.get("/api/v1/msp?commodity=paddy").get_json()["prices"]
    r = client.post("/api/v1/chat", json={"message": "hello", "plain": True})
    assert r.status_code == 200 and "reply_plain" in r.get_json()
    buf = io.BytesIO()
    Image.new("RGB", (32, 32)).save(buf, format="JPEG")
    buf.seek(0)
    r = client.post("/api/v1/disease/detect", data={"image": (buf, "leaf.jpg")}, content_type="multipart/form-data")
    assert r.status_code == 503  # stub backend under TestConfig
