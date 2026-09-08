"""Shared pytest fixtures: an app on an in-memory SQLite DB with the offline assistant."""
from __future__ import annotations

import logging

import pytest

from krishidisha import create_app
from krishidisha.config import TestConfig

logging.disable(logging.WARNING)

FARMER = dict(name="Test Farmer", email="farmer@example.com", phone="9876543210", username="tester",
              password="secret123", state="Madhya Pradesh", district="Indore", land_area_acres="3",
              preferred_language="en")


@pytest.fixture(scope="session")
def app():
    app = create_app(TestConfig)
    yield app


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def farmer_client(app):
    """A fresh client logged in as a verified farmer (registered once per session)."""
    c = app.test_client()
    r = c.post("/farmer_registration", data=FARMER, follow_redirects=True)
    assert r.status_code == 200
    if b"already registered" in r.data:
        r = c.post("/farmer_login", data={"username": FARMER["username"], "password": FARMER["password"]},
                   follow_redirects=True)
        assert r.status_code == 200
    return c


@pytest.fixture()
def admin_client(app):
    c = app.test_client()
    r = c.post("/admin_login", data={"username": "admin", "password": "admin123"}, follow_redirects=True)
    assert r.status_code == 200
    return c
