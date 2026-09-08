"""Feedback loop and own-data collection: consent, ratings, labelling queue, export."""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import pytest

from krishidisha.extensions import db
from krishidisha.models import (Admin, ChatMessage, ChatSession, DataConsent, Farmer, Feedback,
                                LabelTask)

# A 1x1 PNG - small enough to keep the tests fast and fully offline.
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")


# --------------------------------------------------------------------------- helpers
@pytest.fixture()
def uploads(app, tmp_path):
    """Point UPLOAD_DIR at a temp folder so tests never touch static/uploads."""
    original = app.config["UPLOAD_DIR"]
    folder = tmp_path / "uploads"
    folder.mkdir(parents=True, exist_ok=True)
    app.config["UPLOAD_DIR"] = folder
    yield folder
    app.config["UPLOAD_DIR"] = original


def set_consent(app, username, photos, chats):
    with app.app_context():
        farmer = Farmer.query.filter_by(username=username).first()
        assert farmer is not None
        row = DataConsent.query.filter_by(farmer_id=farmer.id).first() or DataConsent(farmer_id=farmer.id)
        row.photos, row.chats = photos, chats
        db.session.add(row)
        db.session.commit()


def second_admin_client(app, username="labeller2", password="second123"):
    with app.app_context():
        admin = Admin.query.filter_by(username=username).first()
        if admin is None:
            admin = Admin(username=username)
            admin.set_password(password)
            db.session.add(admin)
            db.session.commit()
    client = app.test_client()
    r = client.post("/admin_login", data={"username": username, "password": password}, follow_redirects=True)
    assert r.status_code == 200
    return client


def task_marker(task_id) -> bytes:
    """The hidden field that identifies one task in a labelling form."""
    return f'name="task_id" value="{task_id}"'.encode()


def make_task(app, uploads_dir, name, **kwargs):
    """A LabelTask backed by a real image file inside the temp upload dir."""
    (uploads_dir / name).write_bytes(PNG)
    with app.app_context():
        task = LabelTask(image_path=name, source=kwargs.pop("source", "drive"),
                         consent=kwargs.pop("consent", True), status=kwargs.pop("status", "queued"), **kwargs)
        db.session.add(task)
        db.session.commit()
        return task.id


# --------------------------------------------------------------------------- consent
def test_registration_records_consent(app, client):
    data = dict(name="Consent Farmer", email="consent@example.com", phone="9812345678",
                username="consentfarmer", password="secret123", state="Madhya Pradesh", district="Indore",
                land_area_acres="2", preferred_language="en", data_consent="on")
    r = client.post("/farmer_registration", data=data, follow_redirects=True)
    assert r.status_code == 200
    with app.app_context():
        farmer = Farmer.query.filter_by(username="consentfarmer").first()
        assert farmer is not None
        row = DataConsent.query.filter_by(farmer_id=farmer.id).first()
        assert row is not None and row.photos and row.chats


def test_registration_without_consent_records_nothing(app, client):
    data = dict(name="Quiet Farmer", email="quiet@example.com", phone="9812345679", username="quietfarmer",
                password="secret123", state="Madhya Pradesh", district="Indore", preferred_language="en")
    r = client.post("/farmer_registration", data=data, follow_redirects=True)
    assert r.status_code == 200
    with app.app_context():
        farmer = Farmer.query.filter_by(username="quietfarmer").first()
        assert DataConsent.query.filter_by(farmer_id=farmer.id).first() is None


def test_registration_page_shows_consent_checkbox(client):
    r = client.get("/farmer_registration")
    assert r.status_code == 200
    assert b'name="data_consent"' in r.data


def test_profile_consent_toggle(app, farmer_client):
    base = {"name": "Test Farmer", "state": "Madhya Pradesh", "district": "Indore",
            "land_area_acres": "3", "preferred_language": "en"}

    r = farmer_client.post("/profile", data=dict(base, consent_photos="on"), follow_redirects=True)
    assert r.status_code == 200
    with app.app_context():
        farmer = Farmer.query.filter_by(username="tester").first()
        row = DataConsent.query.filter_by(farmer_id=farmer.id).first()
        assert row.photos is True and row.chats is False

    r = farmer_client.post("/profile", data=dict(base, consent_chats="on"), follow_redirects=True)
    assert r.status_code == 200
    with app.app_context():
        farmer = Farmer.query.filter_by(username="tester").first()
        row = DataConsent.query.filter_by(farmer_id=farmer.id).first()
        assert row.photos is False and row.chats is True


def test_crop_detection_page_has_consent_notice(farmer_client):
    r = farmer_client.get("/crop_detection")
    assert r.status_code == 200
    assert b"Manage in Profile" in r.data


# --------------------------------------------------------------------------- /feedback
def test_feedback_chat_rating_is_stored(app, client):
    r = client.post("/feedback", json={"kind": "chat", "ref_id": "session-key#3", "rating": -1,
                                       "comment": "wrong crop", "language": "hi"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    with app.app_context():
        row = db.session.get(Feedback, body["id"])
        assert row.kind == "chat" and row.rating == -1
        assert row.ref_id == "session-key#3" and row.comment == "wrong crop"
        assert row.language == "hi" and row.status == "pending"
        assert row.consent is False  # anonymous visitor: no consent on record


def test_feedback_records_consent_for_logged_in_farmer(app, farmer_client):
    set_consent(app, "tester", photos=False, chats=True)
    r = farmer_client.post("/feedback", json={"kind": "chat", "ref_id": "key#1", "rating": 1})
    assert r.status_code == 200
    assert r.get_json()["consent"] is True


def test_feedback_rejects_bad_input(client):
    assert client.post("/feedback", json={"kind": "nonsense", "ref_id": "x", "rating": 1}).status_code == 400
    assert client.post("/feedback", json={"kind": "chat", "ref_id": "", "rating": 1}).status_code == 400
    assert client.post("/feedback", json={"kind": "chat", "ref_id": "x", "rating": 7}).status_code == 400


def test_feedback_accepts_yield_and_fertilizer(app, client):
    for kind in ("yield", "fertilizer"):
        r = client.post("/feedback", json={"kind": kind, "ref_id": "42", "rating": 1})
        assert r.status_code == 200, kind


def test_labels_endpoint(client):
    r = client.get("/feedback/labels")
    assert r.status_code == 200
    data = r.get_json()
    assert data["count"] == len(data["labels"]) > 5
    assert "unsure" in data["labels"]
    assert data["not_a_leaf"] in data["labels"]
    assert "Other___not_a_leaf" == data["not_a_leaf"]
    assert data["grouped"], "labels must be grouped by crop for the dropdowns"


def _stage_diagnosis(client, image_path):
    with client.session_transaction() as sess:
        sess["disease_result"] = {
            "result": {"top": {"label": "Tomato___Late_blight", "confidence": 0.81}},
            "info": None, "image_path": str(image_path),
            "image_url": "/static/uploads/" + Path(image_path).name, "product_ids": []}


def test_diagnosis_feedback_creates_label_task_with_consent(app, farmer_client, uploads):
    set_consent(app, "tester", photos=True, chats=True)
    image = uploads / "diagnosis_yes.png"
    image.write_bytes(PNG)
    _stage_diagnosis(farmer_client, image)

    r = farmer_client.post("/feedback", json={"kind": "diagnosis", "ref_id": image.name, "rating": -1,
                                              "corrected_label": "Tomato___Early_blight"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["label_task_id"] is not None

    with app.app_context():
        task = db.session.get(LabelTask, body["label_task_id"])
        assert task.source == "app_upload" and task.consent is True and task.status == "queued"
        assert task.model_label == "Tomato___Late_blight"
        assert round(task.model_confidence, 2) == 0.81
        row = db.session.get(Feedback, body["id"])
        assert row.corrected_label == "Tomato___Early_blight"
        assert row.image_path == task.image_path
    # the original upload is copied, never moved
    assert image.exists()
    assert (uploads / task_path(app, body["label_task_id"])).exists()


def task_path(app, task_id):
    with app.app_context():
        return db.session.get(LabelTask, task_id).image_path


def test_diagnosis_feedback_without_consent_creates_no_label_task(app, farmer_client, uploads):
    set_consent(app, "tester", photos=False, chats=False)
    image = uploads / "diagnosis_no.png"
    image.write_bytes(PNG)
    _stage_diagnosis(farmer_client, image)

    with app.app_context():
        before = LabelTask.query.count()
    r = farmer_client.post("/feedback", json={"kind": "diagnosis", "ref_id": image.name, "rating": 1})
    assert r.status_code == 200
    assert r.get_json()["label_task_id"] is None
    with app.app_context():
        assert LabelTask.query.count() == before


# --------------------------------------------------------------------------- chat photo capture
def test_chat_photo_creates_label_task_with_consent(app, farmer_client, uploads):
    set_consent(app, "tester", photos=True, chats=True)
    with app.app_context():
        before = LabelTask.query.filter_by(source="chat").count()

    r = farmer_client.post("/chat/send", data={"message": "is patte me kya bimari hai?",
                                               "image": (io.BytesIO(PNG), "leaf.png")},
                           content_type="multipart/form-data")
    assert r.status_code == 200
    assert "message_index" in r.get_json()  # the chat UI needs it for the feedback ref_id

    with app.app_context():
        tasks = LabelTask.query.filter_by(source="chat").order_by(LabelTask.id).all()
        assert len(tasks) == before + 1
        task = tasks[-1]
        assert task.consent is True and task.status == "queued"
        assert task.image_path.startswith("chat/")
        assert (uploads / task.image_path).is_file()


def test_chat_photo_without_consent_is_not_kept(app, farmer_client, uploads):
    set_consent(app, "tester", photos=False, chats=False)
    with app.app_context():
        before = LabelTask.query.filter_by(source="chat").count()
    r = farmer_client.post("/chat/send", data={"message": "aur ye?", "image": (io.BytesIO(PNG), "leaf.png")},
                           content_type="multipart/form-data")
    assert r.status_code == 200
    with app.app_context():
        assert LabelTask.query.filter_by(source="chat").count() == before


# --------------------------------------------------------------------------- admin labelling
def test_admin_dashboard_shows_collection_tiles(admin_client):
    r = admin_client.get("/admin_dashboard")
    assert r.status_code == 200
    assert b"Photos waiting for a label" in r.data
    assert b"Feedback awaiting review" in r.data


def test_label_queue_requires_admin(client):
    r = client.get("/admin/label")
    assert r.status_code in (302, 401)


def test_label_queue_agreement_flow(app, admin_client, uploads):
    task_id = make_task(app, uploads, "agree.png", model_label="Rice___Blast", model_confidence=0.6)

    page = admin_client.get("/admin/label?n=12")
    assert page.status_code == 200
    assert b"Labelling queue" in page.data
    assert task_marker(task_id) in page.data

    r = admin_client.post("/admin/label", data={"task_id": task_id, "action": "label",
                                                "label": "Rice___Brown_spot"}, follow_redirects=True)
    assert r.status_code == 200
    with app.app_context():
        task = db.session.get(LabelTask, task_id)
        assert task.status == "needs_second"
        assert task.label_1 == "Rice___Brown_spot" and task.labeller_1 == "admin"
        assert task.final_label is None

    # the first labeller must not see their own task again
    assert task_marker(task_id) not in admin_client.get("/admin/label?n=12").data

    other = second_admin_client(app)
    assert task_marker(task_id) in other.get("/admin/label?n=12").data
    r = other.post("/admin/label", data={"task_id": task_id, "action": "label",
                                         "label": "Rice___Brown_spot"}, follow_redirects=True)
    assert r.status_code == 200
    with app.app_context():
        task = db.session.get(LabelTask, task_id)
        assert task.status == "labelled"
        assert task.final_label == "Rice___Brown_spot"
        assert task.labeller_2 == "labeller2"


def test_label_queue_not_a_leaf_and_discard(app, admin_client, uploads):
    leaf_id = make_task(app, uploads, "notleaf.png")
    admin_client.post("/admin/label", data={"task_id": leaf_id, "action": "not_leaf"}, follow_redirects=True)
    other = second_admin_client(app)
    other.post("/admin/label", data={"task_id": leaf_id, "action": "not_leaf"}, follow_redirects=True)
    with app.app_context():
        assert db.session.get(LabelTask, leaf_id).final_label == "Other___not_a_leaf"

    junk_id = make_task(app, uploads, "junk.png")
    admin_client.post("/admin/label", data={"task_id": junk_id, "action": "discard"}, follow_redirects=True)
    with app.app_context():
        assert db.session.get(LabelTask, junk_id).status == "discarded"


def test_label_disagreement_and_resolution(app, admin_client, uploads):
    task_id = make_task(app, uploads, "disagree.png")
    admin_client.post("/admin/label", data={"task_id": task_id, "action": "label",
                                            "label": "Wheat___Leaf_rust"}, follow_redirects=True)
    other = second_admin_client(app)
    other.post("/admin/label", data={"task_id": task_id, "action": "label",
                                     "label": "Wheat___Stripe_rust"}, follow_redirects=True)
    with app.app_context():
        assert db.session.get(LabelTask, task_id).status == "disagreement"

    page = admin_client.get("/admin/label/disagreements")
    assert page.status_code == 200
    assert task_marker(task_id) in page.data

    r = admin_client.post("/admin/label/disagreements",
                          data={"task_id": task_id, "final_label": "Wheat___Stripe_rust"}, follow_redirects=True)
    assert r.status_code == 200
    with app.app_context():
        task = db.session.get(LabelTask, task_id)
        assert task.status == "labelled" and task.final_label == "Wheat___Stripe_rust"
        assert "resolved by admin" in task.notes


def test_bulk_upload_creates_tasks(app, admin_client, uploads):
    r = admin_client.post("/admin/label/upload", data={
        "images": [(io.BytesIO(PNG), "drive1.png"), (io.BytesIO(PNG), "drive2.png")],
    }, content_type="multipart/form-data", follow_redirects=True)
    assert r.status_code == 200
    with app.app_context():
        queued = LabelTask.query.filter_by(source="drive", status="queued").all()
        assert len(queued) >= 2
        assert all(t.consent for t in queued)
        assert (uploads / queued[-1].image_path).is_file()


def test_bulk_upload_trusted_label_is_labelled_immediately(app, admin_client, uploads):
    r = admin_client.post("/admin/label/upload", data={
        "images": [(io.BytesIO(PNG), "trusted.png")],
        "label": "Cotton___Bacterial_blight", "trust_label": "on",
    }, content_type="multipart/form-data", follow_redirects=True)
    assert r.status_code == 200
    with app.app_context():
        task = LabelTask.query.filter_by(final_label="Cotton___Bacterial_blight").first()
        assert task is not None and task.status == "labelled"
        assert task.labeller_1 == "admin" and task.labeller_2 == "admin"


# --------------------------------------------------------------------------- admin feedback review
def test_admin_feedback_accept_creates_label_task(app, admin_client, farmer_client, uploads):
    set_consent(app, "tester", photos=True, chats=True)
    image = uploads / "review.png"
    image.write_bytes(PNG)
    _stage_diagnosis(farmer_client, image)
    body = farmer_client.post("/feedback", json={"kind": "diagnosis", "ref_id": image.name, "rating": -1,
                                                 "corrected_label": "Tomato___Leaf_mold"}).get_json()

    page = admin_client.get("/admin/feedback?status=pending")
    assert page.status_code == 200
    assert b"Farmer feedback" in page.data

    r = admin_client.post("/admin/feedback", data={"feedback_id": body["id"], "action": "accept"},
                          follow_redirects=True)
    assert r.status_code == 200
    with app.app_context():
        row = db.session.get(Feedback, body["id"])
        assert row.status == "accepted" and row.reviewer == "admin" and row.reviewed_at is not None
        task = LabelTask.query.filter_by(image_path=row.image_path).first()
        assert task.final_label == "Tomato___Leaf_mold" and task.status == "labelled"


def test_admin_feedback_reject(app, admin_client, client):
    body = client.post("/feedback", json={"kind": "chat", "ref_id": "k#0", "rating": -1}).get_json()
    admin_client.post("/admin/feedback", data={"feedback_id": body["id"], "action": "reject"},
                      follow_redirects=True)
    with app.app_context():
        assert db.session.get(Feedback, body["id"]).status == "rejected"


# --------------------------------------------------------------------------- export
def test_export_writes_photos_and_chat_jsonl(app, admin_client, uploads, tmp_path, monkeypatch):
    data_root = tmp_path / "datasets"
    monkeypatch.setenv("KRISHIDISHA_DATA_ROOT", str(data_root))
    original_data_dir = app.config["DATA_DIR"]
    app.config["DATA_DIR"] = tmp_path / "appdata"
    try:
        with app.app_context():
            (uploads / "export.png").write_bytes(PNG)
            task = LabelTask(image_path="export.png", source="drive", consent=True, status="labelled",
                             final_label="Sugarcane___Red_rot", label_1="Sugarcane___Red_rot",
                             label_2="Sugarcane___Red_rot", labeller_1="a", labeller_2="b")
            # consent False -> must never be exported
            (uploads / "private.png").write_bytes(PNG)
            private = LabelTask(image_path="private.png", source="drive", consent=False, status="labelled",
                                final_label="Rice___Blast")
            chat = ChatSession(session_key="export-key", language="hi")
            db.session.add_all([task, private, chat])
            db.session.commit()
            db.session.add_all([ChatMessage(session_id=chat.id, role="user", content="mere dhan me dhabbe hain"),
                                ChatMessage(session_id=chat.id, role="assistant", content="Brown spot lagta hai")])
            db.session.commit()
            db.session.add(Feedback(kind="chat", ref_id="export-key#1", rating=1, comment="madad hui",
                                    consent=True, language="hi"))
            db.session.commit()
            task_id, private_id = task.id, private.id

        r = admin_client.post("/admin/label/export?format=json")
        assert r.status_code == 200
        summary = r.get_json()
        assert summary["exported"] >= 1

        exported = data_root / "own_photos" / "Sugarcane___Red_rot"
        assert exported.is_dir()
        assert (exported / f"{task_id}.png").is_file()
        assert not (data_root / "own_photos" / "Rice___Blast").exists()
        assert (uploads / "export.png").is_file(), "export must copy, never move"

        with app.app_context():
            assert db.session.get(LabelTask, task_id).status == "exported"
            assert db.session.get(LabelTask, private_id).status == "labelled"

        jsonl = tmp_path / "appdata" / "llm" / "feedback.jsonl"
        assert jsonl.is_file()
        lines = [json.loads(x) for x in jsonl.read_text(encoding="utf-8").splitlines() if x.strip()]
        turn = [line for line in lines if line["session_key"] == "export-key"]
        assert len(turn) == 1
        assert turn[0] == {"session_key": "export-key", "user": "mere dhan me dhabbe hain",
                           "assistant": "Brown spot lagta hai", "rating": 1, "comment": "madad hui",
                           "language": "hi"}

        # a second export must not duplicate the already-exported photo
        again = admin_client.post("/admin/label/export?format=json").get_json()
        assert task_id not in [t for t in again["per_class"]]
        assert again["per_class"].get("Sugarcane___Red_rot") is None
    finally:
        app.config["DATA_DIR"] = original_data_dir
