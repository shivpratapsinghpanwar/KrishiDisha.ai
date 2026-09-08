"""Pydantic schemas for the standalone FastAPI service (mirrors the Flask /api/v1 payloads)."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, examples=["How much urea for 2 acres of wheat?"])
    language: str = Field("en", max_length=10, description="en, hi, hinglish, mr, pa, gu, ta, te, kn, bn")
    history: Optional[list[dict[str, str]]] = Field(None, description="Prior turns: [{role, content}]")
    farmer_context: Optional[str] = Field(None, description="Free text profile, e.g. 'state: Punjab, land: 3 acres'")
    plain: bool = Field(False, description="Also return a markdown-stripped reply for SMS / TTS")


class ChatResponse(BaseModel):
    reply: str
    reply_plain: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    tools_used: list[str] = []
    sources: list[str] = []


class CropRecommendationInput(BaseModel):
    N: float = Field(..., ge=0, le=200, description="Soil nitrogen kg/ha")
    P: float = Field(..., ge=0, le=200, description="Soil phosphorus kg/ha")
    K: float = Field(..., ge=0, le=250, description="Soil potassium kg/ha")
    temperature: float = Field(..., ge=-5, le=60, description="Celsius")
    humidity: float = Field(..., ge=0, le=100, description="Relative humidity %")
    ph: float = Field(..., ge=0, le=14)
    rainfall: float = Field(..., ge=0, le=5000, description="mm")


class Alternative(BaseModel):
    crop: Optional[str] = None
    fertilizer: Optional[str] = None
    probability: float


class CropRecommendationOutput(BaseModel):
    recommended_crop: str
    confidence: float
    alternatives: list[Alternative]
    revenue_per_acre: Optional[float] = None
    cost_per_acre: Optional[float] = None
    profit_per_acre: Optional[float] = None
    image: Optional[str] = None
    guide: Optional[dict[str, Any]] = None


class FertilizerRecommendationInput(BaseModel):
    temperature: float = Field(..., ge=-5, le=60)
    humidity: float = Field(..., ge=0, le=100)
    moisture: float = Field(..., ge=0, le=100)
    soil_type: str = Field(..., examples=["Sandy"], description="Black, Clayey, Loamy, Red, Sandy")
    crop_type: str = Field(..., examples=["Maize"])
    N: float = Field(..., ge=0, le=200)
    P: float = Field(..., ge=0, le=200)
    K: float = Field(..., ge=0, le=250)
    area: Optional[float] = Field(None, gt=0, description="Optional field area to get a dose calculator")
    unit: str = Field("acre", pattern="^(acre|hectare)$")


class FertilizerRecommendationOutput(BaseModel):
    recommended_fertilizer: str
    confidence: float
    alternatives: list[Alternative]
    npk: Optional[str] = None
    usage_note: Optional[str] = None
    image: Optional[str] = None
    calculator: Optional[dict[str, Any]] = None


class FertilizerCalculatorInput(BaseModel):
    crop: str
    area: float = Field(..., gt=0)
    unit: str = Field("acre", pattern="^(acre|hectare)$")
    soil_n: Optional[float] = None
    soil_p: Optional[float] = None
    soil_k: Optional[float] = None


class Prediction(BaseModel):
    label: str
    name: str
    crop: str
    condition: str
    is_healthy: bool
    confidence: float


class DiseaseDetectionOutput(BaseModel):
    available: bool
    backend: str
    model: Optional[str] = None
    message: Optional[str] = None
    predictions: list[Prediction] = []
    top: Optional[Prediction] = None
    info: Optional[dict[str, Any]] = None
    products: list[dict[str, Any]] = []


class CropYieldInput(BaseModel):
    crop: str = Field(..., examples=["Wheat"])
    crop_year: int = Field(..., ge=1990, le=2040)
    season: str = Field(..., examples=["Rabi"])
    state: str = Field(..., examples=["Punjab"])
    area: float = Field(..., gt=0, description="hectares")
    production: float = Field(..., ge=0, description="tonnes")
    annual_rainfall: float = Field(..., ge=0, description="mm")
    fertilizer: float = Field(..., ge=0, description="kg")
    pesticide: float = Field(..., ge=0, description="kg")


class CropYieldOutput(BaseModel):
    predicted_yield: float
    unit: str
    estimated_production: float
    tips: list[str] = []


class HealthCheck(BaseModel):
    status: str
    assistant: dict[str, Any]
    models: dict[str, bool]
    disease_model: dict[str, Any]


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
