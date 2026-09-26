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
    #: Inputs outside the 1st-99th percentile of the training data.
    warnings: list[str] = []


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
    #: Rule ranking similarity of the top product (0-1), not a model probability.
    confidence: float
    alternatives: list[Alternative]
    #: Top 3 products with their NPK-ratio similarity to the nutrient deficit.
    ranked: list[dict[str, Any]] = []
    #: One sentence explaining the pick.
    why: Optional[str] = None
    #: Soil-test-adjusted requirement in kg/ha of N, P2O5 and K2O.
    deficit_kg_per_ha: dict[str, float] = {}
    soil_status: dict[str, str] = {}
    crop_key: Optional[str] = None
    method: Optional[str] = None
    #: The trained classifier's opinion, reported but not obeyed.
    model_hint: Optional[dict[str, Any]] = None
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
    """Yield inputs.

    ``production`` is deliberately absent: in the training data
    ``Yield == Production / Area``, so accepting it would leak the target.
    """

    crop: str = Field(..., examples=["Wheat"])
    crop_year: int = Field(..., ge=1990, le=2040)
    season: str = Field(..., examples=["Rabi"])
    state: str = Field(..., examples=["Punjab"])
    area: float = Field(..., gt=0, description="hectares")
    annual_rainfall: float = Field(..., ge=0, description="mm")
    fertilizer: Optional[float] = Field(
        None, ge=0, description="kg, total for the area; omitted -> State x Crop median")
    pesticide: Optional[float] = Field(
        None, ge=0, description="kg, total for the area; omitted -> State x Crop median")


class CropYieldOutput(BaseModel):
    predicted_yield: float
    unit: str
    estimated_production: float
    #: Calibrated 80 % prediction interval in t/ha, when a band is available.
    expected_range: Optional[list[float]] = None
    #: How that band was built and the coverage it actually achieved in back-testing.
    expected_range_note: Optional[str] = None
    #: Median yield for this crop and state over the previous five years.
    baseline_yield: Optional[float] = None
    #: Which estimator won the 2017-2020 back-test and produced this number.
    model: Optional[str] = None
    #: The fertilizer / pesticide figures actually used, and where they came from.
    inputs_used: dict[str, Any] = {}
    tips: list[str] = []


class HealthCheck(BaseModel):
    status: str
    assistant: dict[str, Any]
    models: dict[str, bool]
    disease_model: dict[str, Any]


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
