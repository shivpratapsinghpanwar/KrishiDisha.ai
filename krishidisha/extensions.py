"""Shared Flask extensions (instantiated once, bound in create_app)."""
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()
