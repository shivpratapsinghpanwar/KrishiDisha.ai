"""
Pydantic models for KrishiDisha Agriculture API
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class ChatMessage(BaseModel):
    """Model for chat messages"""
    message: str = Field(..., description="User's message to the chatbot")
    farmer_id: Optional[int] = Field(None, description="Farmer ID for personalized responses")


class ChatResponse(BaseModel):
    """Model for chatbot responses"""
    response: str = Field(..., description="Chatbot's response")
    timestamp: datetime = Field(default_factory=datetime.now)


class CropRecommendationInput(BaseModel):
    """Model for crop recommendation input"""
    N: float = Field(..., description="Nitrogen content in soil")
    P: float = Field(..., description="Phosphorus content in soil")
    K: float = Field(..., description="Potassium content in soil")
    temperature: float = Field(..., description="Temperature in Celsius")
    humidity: float = Field(..., description="Humidity percentage")
    ph: float = Field(..., description="Soil pH level")
    rainfall: float = Field(..., description="Rainfall in mm")


class CropRecommendationOutput(BaseModel):
    """Model for crop recommendation output"""
    recommended_crop: str = Field(..., description="Recommended crop name")
    confidence: float = Field(..., description="Prediction confidence score")
    additional_info: Optional[str] = Field(None, description="Additional farming advice")


class FertilizerRecommendationInput(BaseModel):
    """Model for fertilizer recommendation input"""
    N: float = Field(..., description="Nitrogen content in soil")
    P: float = Field(..., description="Phosphorus content in soil")
    K: float = Field(..., description="Potassium content in soil")
    soil_type: str = Field(..., description="Type of soil (e.g., loamy, clay, sandy)")
    crop_type: str = Field(..., description="Crop type")


class FertilizerRecommendationOutput(BaseModel):
    """Model for fertilizer recommendation output"""
    recommended_fertilizer: str = Field(..., description="Recommended fertilizer name")
    npk_ratio: str = Field(..., description="NPK ratio for the fertilizer")
    application_rate: Optional[str] = Field(None, description="Application rate per hectare")


class DiseaseDetectionInput(BaseModel):
    """Model for disease detection input"""
    image_url: Optional[str] = Field(None, description="URL of the plant image")
    image_file: Optional[str] = Field(None, description="Base64 encoded image file")


class DiseaseDetectionOutput(BaseModel):
    """Model for disease detection output"""
    disease_name: str = Field(..., description="Name of the detected disease")
    confidence: float = Field(..., description="Prediction confidence score")
    description: Optional[str] = Field(None, description="Disease description")
    preventive_measures: Optional[str] = Field(None, description="Preventive measures")
    supplement_recommendation: Optional[str] = Field(None, description="Recommended supplement")


class CropYieldInput(BaseModel):
    """Model for crop yield prediction input"""
    crop: str = Field(..., description="Crop name")
    crop_year: int = Field(..., description="Crop year")
    season: str = Field(..., description="Season (Kharif, Rabi, Zaid)")
    state: str = Field(..., description="State name")
    area: float = Field(..., description="Area in hectares")
    production: float = Field(..., description="Production in tonnes")
    annual_rainfall: float = Field(..., description="Annual rainfall in mm")
    fertilizer: float = Field(..., description="Fertilizer usage in kg/ha")
    pesticide: float = Field(..., description="Pesticide usage in kg/ha")


class CropYieldOutput(BaseModel):
    """Model for crop yield prediction output"""
    predicted_yield: float = Field(..., description="Predicted yield in tonnes/hectare")
    confidence: float = Field(..., description="Prediction confidence score")
    recommendations: Optional[List[str]] = Field(None, description="Yield improvement recommendations")


class FarmerActivityLog(BaseModel):
    """Model for logging farmer activities"""
    farmer_id: int = Field(..., description="Farmer ID")
    activity_type: str = Field(..., description="Type of activity")
    input_data: Dict[str, Any] = Field(..., description="Input data for the activity")
    output_data: Dict[str, Any] = Field(..., description="Output data from the activity")


class ActivityResponse(BaseModel):
    """Model for activity log response"""
    status: str = Field(..., description="Status of the operation")
    message: str = Field(..., description="Response message")
    activity_id: Optional[int] = Field(None, description="ID of the logged activity")


class HealthCheck(BaseModel):
    """Model for health check response"""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(default_factory=datetime.now)


class ErrorResponse(BaseModel):
    """Model for error responses"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    status_code: int = Field(..., description="HTTP status code")
