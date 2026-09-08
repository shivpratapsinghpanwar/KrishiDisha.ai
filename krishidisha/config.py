"""Application configuration, driven entirely by environment variables.

Copy `.env.example` to `.env` and adjust. Nothing here is hard-coded to a
developer's machine any more.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


class Config:
    # --- Core Flask ---------------------------------------------------------
    SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-production")
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024

    # --- Paths --------------------------------------------------------------
    BASE_DIR = BASE_DIR
    DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
    MODELS_DIR = Path(os.getenv("MODELS_DIR", BASE_DIR / "models"))
    UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", BASE_DIR / "static" / "uploads"))

    # --- Database -----------------------------------------------------------
    # Default is a local SQLite file so the project runs with zero setup.
    # For MySQL: mysql+mysqlconnector://user:pass@localhost/farmer_details
    SQLALCHEMY_DATABASE_URI = os.getenv(
        "DATABASE_URL", f"sqlite:///{(BASE_DIR / 'instance' / 'krishidisha.db').as_posix()}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {"pool_pre_ping": True}

    # --- Auth / onboarding --------------------------------------------------
    # The paper's workflow requires an admin to verify each farmer. Set this
    # to true for demos so new farmers can log in immediately.
    AUTO_VERIFY_FARMERS = _bool("AUTO_VERIFY_FARMERS", False)
    DEFAULT_ADMIN_USERNAME = os.getenv("DEFAULT_ADMIN_USERNAME", "admin")
    DEFAULT_ADMIN_PASSWORD = os.getenv("DEFAULT_ADMIN_PASSWORD", "admin123")

    # --- LLM / chatbot ------------------------------------------------------
    # Provider: "anthropic" (default when ANTHROPIC_API_KEY is set),
    # "openai" (any OpenAI-compatible endpoint: OpenAI, Groq, OpenRouter, Ollama),
    # or "rules" (offline keyword bot, always available as a fallback).
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-5")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")  # e.g. http://localhost:11434/v1 for Ollama
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    LLM_MAX_TOOL_ROUNDS = int(os.getenv("LLM_MAX_TOOL_ROUNDS", "6"))
    CHAT_HISTORY_TURNS = int(os.getenv("CHAT_HISTORY_TURNS", "12"))

    # --- External data ------------------------------------------------------
    # data.gov.in Agmarknet daily mandi prices. The public sample key works
    # with a small daily quota; register for your own key for production.
    DATA_GOV_API_KEY = os.getenv(
        "DATA_GOV_API_KEY", "579b464db66ec23bdd000001cdd3946e44ce4aad7209ff7b23ac571b"
    )
    WEATHER_CACHE_SECONDS = int(os.getenv("WEATHER_CACHE_SECONDS", "900"))

    # --- Marketplace --------------------------------------------------------
    CURRENCY = os.getenv("CURRENCY", "INR")
    FREE_DELIVERY_ABOVE = float(os.getenv("FREE_DELIVERY_ABOVE", "999"))
    DELIVERY_CHARGE = float(os.getenv("DELIVERY_CHARGE", "49"))
    UPI_ID = os.getenv("UPI_ID", "krishidisha@upi")  # shown on the order page for UPI payments
    RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID")
    RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET")

    # --- Disease model ------------------------------------------------------
    # "auto" tries the locally trained checkpoint first, then the Hugging Face
    # fallback model, and finally a stub that explains no model is loaded.
    DISEASE_MODEL_BACKEND = os.getenv("DISEASE_MODEL_BACKEND", "auto")
    DISEASE_HF_MODEL = os.getenv(
        "DISEASE_HF_MODEL", "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
    )


class TestConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
    WTF_CSRF_ENABLED = False
    AUTO_VERIFY_FARMERS = True
    LLM_PROVIDER = "rules"
    DISEASE_MODEL_BACKEND = "stub"
    SECRET_KEY = "test"
