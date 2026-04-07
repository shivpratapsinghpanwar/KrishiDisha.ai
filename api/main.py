"""
KrishiDisha FastAPI Application
Agriculture LLM API for crop recommendation, disease detection, and farmer assistance
"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional, List
import os
import sys
import numpy as np
import pandas as pd
import torch
import joblib
from PIL import Image
import io
import base64
from datetime import datetime
import re
import random

# Import Pydantic models
from api.models import (
    ChatMessage,
    ChatResponse,
    CropRecommendationInput,
    CropRecommendationOutput,
    FertilizerRecommendationInput,
    FertilizerRecommendationOutput,
    DiseaseDetectionInput,
    DiseaseDetectionOutput,
    CropYieldInput,
    CropYieldOutput,
    FarmerActivityLog,
    ActivityResponse,
    HealthCheck,
    ErrorResponse
)

# Import KrishiDisha Bot
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from krishidisha_bot import KrishiDishaBot

# Import CNN model
import CNN

# Initialize FastAPI app
app = FastAPI(
    title="KrishiDisha Agriculture API",
    description="AI-powered agricultural API for crop recommendation, disease detection, fertilizer suggestions, and yield prediction",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Initialize KrishiDisha Bot
chatbot = KrishiDishaBot()

# Load ML Models
try:
    # Crop Recommendation Model
    crop_model_path = os.path.join(MODELS_DIR, "crop_recommendation_model.pkl")
    crop_label_encoder_path = os.path.join(MODELS_DIR, "label_encoder.pkl")
    
    if os.path.exists(crop_model_path):
        crop_model = joblib.load(crop_model_path)
        label_encoder = joblib.load(crop_label_encoder_path)
    else:
        crop_model = None
        label_encoder = None
        print("Warning: Crop recommendation model not found")
    
    # Fertilizer Recommendation Model
    fertilizer_model_path = os.path.join(MODELS_DIR, "fertilizer_recommendation_model.pkl")
    fertilizer_label_encoder_path = os.path.join(MODELS_DIR, "fertilizer_label_encoder.pkl")
    
    if os.path.exists(fertilizer_model_path):
        fertilizer_model = joblib.load(fertilizer_model_path)
        fertilizer_label_encoder = joblib.load(fertilizer_label_encoder_path)
    else:
        fertilizer_model = None
        fertilizer_label_encoder = None
        print("Warning: Fertilizer recommendation model not found")
    
    # Disease Detection Model
    disease_model_path = os.path.join(MODELS_DIR, "plant_disease_model_1_latest.pt")
    
    if os.path.exists(disease_model_path):
        disease_model = CNN.CNN(39)
        disease_model.load_state_dict(torch.load(disease_model_path, map_location=torch.device('cpu')))
        disease_model.eval()
    else:
        disease_model = None
        print("Warning: Disease detection model not found")
    
    # Crop Yield Prediction Pipeline
    yield_pipeline_path = os.path.join(MODELS_DIR, "pipeline_yield_prediction_pipeline.pkl")
    
    if os.path.exists(yield_pipeline_path):
        yield_pipeline = joblib.load(yield_pipeline_path)
    else:
        yield_pipeline = None
        print("Warning: Crop yield prediction pipeline not found")
    
except Exception as e:
    print(f"Error loading models: {e}")
    crop_model = None
    fertilizer_model = None
    disease_model = None
    yield_pipeline = None

# Load Data Files
try:
    disease_info_path = os.path.join(DATA_DIR, "disease_info.csv")
    supplement_info_path = os.path.join(DATA_DIR, "supplement_info.csv")
    crop_yield_data_path = os.path.join(DATA_DIR, "crop_yield.csv")
    
    if os.path.exists(disease_info_path):
        disease_info = pd.read_csv(disease_info_path, encoding='cp1252')
    else:
        disease_info = pd.DataFrame()
        print("Warning: Disease info CSV not found")
    
    if os.path.exists(supplement_info_path):
        supplement_info = pd.read_csv(supplement_info_path, encoding='cp1252')
    else:
        supplement_info = pd.DataFrame()
        print("Warning: Supplement info CSV not found")
    
    if os.path.exists(crop_yield_data_path):
        crop_yield_data = pd.read_csv(crop_yield_data_path)
    else:
        crop_yield_data = pd.DataFrame()
        print("Warning: Crop yield data CSV not found")
        
except Exception as e:
    print(f"Error loading data files: {e}")
    disease_info = pd.DataFrame()
    supplement_info = pd.DataFrame()
    crop_yield_data = pd.DataFrame()

# Helper Functions
def predict_disease(image_bytes: bytes) -> tuple:
    """Predict plant disease from image"""
    if disease_model is None:
        raise HTTPException(status_code=503, detail="Disease detection model not loaded")
    
    try:
        # Load and preprocess image
        image = Image.open(io.BytesIO(image_bytes))
        image = image.resize((224, 224))
        
        # Convert to tensor
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        input_tensor = transform(image).unsqueeze(0)
        
        # Get prediction
        with torch.no_grad():
            output = disease_model(input_tensor)
            output = output.detach().numpy()
            index = np.argmax(output)
            confidence = float(np.max(output))
        
        # Get disease information
        disease_name = CNN.idx_to_classes.get(index, "Unknown Disease")
        
        # Get additional info from CSV if available
        description = ""
        preventive_measures = ""
        supplement = ""
        
        if len(disease_info) > index:
            description = str(disease_info.iloc[index].get('description', ''))
            preventive_measures = str(disease_info.iloc[index].get('Possible Steps', ''))
            
            if len(supplement_info) > index:
                supplement = str(supplement_info.iloc[index].get('supplement name', 'N/A'))
        
        return disease_name, confidence, description, preventive_measures, supplement
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Disease prediction error: {str(e)}")


def recommend_crop(N: float, P: float, K: float, temperature: float, 
                   humidity: float, ph: float, rainfall: float) -> tuple:
    """Recommend crop based on soil and climate parameters"""
    if crop_model is None or label_encoder is None:
        raise HTTPException(status_code=503, detail="Crop recommendation model not loaded")
    
    try:
        # Prepare input array
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        
        # Get prediction
        prediction = crop_model.predict(input_data)
        probabilities = crop_model.predict_proba(input_data)
        
        # Decode label
        recommended_crop = label_encoder.inverse_transform(prediction)[0]
        confidence = float(np.max(probabilities))
        
        # Get additional info from knowledge base
        additional_info = chatbot.knowledge_base["crop_recommendation"]["info"].get(
            recommended_crop.lower(), 
            f"{recommended_crop} is a great choice for your conditions!"
        )
        
        return recommended_crop, confidence, additional_info
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Crop recommendation error: {str(e)}")


def recommend_fertilizer(N: float, P: float, K: float, soil_type: str, 
                         crop_type: str) -> tuple:
    """Recommend fertilizer based on soil and crop parameters"""
    if fertilizer_model is None or fertilizer_label_encoder is None:
        raise HTTPException(status_code=503, detail="Fertilizer recommendation model not loaded")
    
    try:
        # Prepare input array
        # Note: This assumes the model expects N, P, K as numeric and soil/crop as encoded
        # You may need to adjust based on your actual model's expectations
        input_data = np.array([[N, P, K]])
        
        # Get prediction
        prediction = fertilizer_model.predict(input_data)
        probabilities = fertilizer_model.predict_proba(input_data)
        
        # Decode label
        recommended_fertilizer = fertilizer_label_encoder.inverse_transform(prediction)[0]
        confidence = float(np.max(probabilities))
        
        # Get NPK ratio from knowledge base
        npk_ratio = "Balanced NPK"
        application_rate = chatbot.knowledge_base["fertilizer"]["recommendations"].get(
            crop_type.lower(),
            "Apply based on soil test results and crop requirements"
        )
        
        return recommended_fertilizer, npk_ratio, application_rate
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fertilizer recommendation error: {str(e)}")


def predict_yield(crop: str, crop_year: int, season: str, state: str,
                  area: float, production: float, annual_rainfall: float,
                  fertilizer: float, pesticide: float) -> tuple:
    """Predict crop yield"""
    if yield_pipeline is None:
        raise HTTPException(status_code=503, detail="Yield prediction pipeline not loaded")
    
    try:
        # Prepare input data
        input_data = pd.DataFrame({
            'Crop': [crop],
            'Crop_Year': [crop_year],
            'Season': [season],
            'State': [state],
            'Area': [area],
            'Production': [production],
            'Annual_Rainfall': [annual_rainfall],
            'Fertilizer': [fertilizer],
            'Pesticide': [pesticide]
        })
        
        # Get prediction
        prediction = yield_pipeline.predict(input_data)
        predicted_yield = float(prediction[0])
        
        # Calculate confidence (simplified - in production use proper calibration)
        confidence = 0.85
        
        # Generate recommendations
        recommendations = []
        if predicted_yield < 2.0:
            recommendations.append("Consider improving soil health with organic matter")
            recommendations.append("Optimize irrigation scheduling")
        if fertilizer < 100:
            recommendations.append("Increase fertilizer application based on soil test")
        if pesticide < 0.5:
            recommendations.append("Implement integrated pest management practices")
        
        return predicted_yield, confidence, recommendations
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Yield prediction error: {str(e)}")


# Activity Logging (In-memory for demo, use database in production)
activity_logs = []

def log_activity(farmer_id: int, activity_type: str, input_data: dict, output_data: dict) -> int:
    """Log farmer activity"""
    activity_id = len(activity_logs) + 1
    activity_logs.append({
        "id": activity_id,
        "farmer_id": farmer_id,
        "activity_type": activity_type,
        "input_data": input_data,
        "output_data": output_data,
        "timestamp": datetime.now()
    })
    return activity_id


# API Routes

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to KrishiDisha Agriculture API",
        "version": "1.0.0",
        "documentation": "/docs",
        "endpoints": [
            "/health",
            "/chat",
            "/crop/recommend",
            "/fertilizer/recommend",
            "/disease/detect",
            "/yield/predict"
        ]
    }


@app.get("/health", response_model=HealthCheck, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return HealthCheck(
        status="healthy",
        version="1.0.0"
    )


@app.post("/chat", response_model=ChatResponse, tags=["Chatbot"])
async def chat(chat_message: ChatMessage):
    """
    Chat with KrishiDisha agricultural assistant
    
    - **message**: User's message to the chatbot
    - **farmer_id**: Optional farmer ID for personalized responses
    """
    try:
        response = chatbot.get_response(chat_message.message)
        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


@app.post("/crop/recommend", response_model=CropRecommendationOutput, tags=["Crop Recommendation"])
async def get_crop_recommendation(input_data: CropRecommendationInput):
    """
    Get crop recommendation based on soil and climate parameters
    
    - **N**: Nitrogen content in soil
    - **P**: Phosphorus content in soil
    - **K**: Potassium content in soil
    - **temperature**: Temperature in Celsius
    - **humidity**: Humidity percentage
    - **ph**: Soil pH level
    - **rainfall**: Rainfall in mm
    """
    try:
        recommended_crop, confidence, additional_info = recommend_crop(
            input_data.N, input_data.P, input_data.K,
            input_data.temperature, input_data.humidity,
            input_data.ph, input_data.rainfall
        )
        
        return CropRecommendationOutput(
            recommended_crop=recommended_crop,
            confidence=confidence,
            additional_info=additional_info
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@app.post("/fertilizer/recommend", response_model=FertilizerRecommendationOutput, tags=["Fertilizer Recommendation"])
async def get_fertilizer_recommendation(input_data: FertilizerRecommendationInput):
    """
    Get fertilizer recommendation based on soil and crop parameters
    
    - **N**: Nitrogen content in soil
    - **P**: Phosphorus content in soil
    - **K**: Potassium content in soil
    - **soil_type**: Type of soil (loamy, clay, sandy, etc.)
    - **crop_type**: Crop type
    """
    try:
        recommended_fertilizer, npk_ratio, application_rate = recommend_fertilizer(
            input_data.N, input_data.P, input_data.K,
            input_data.soil_type, input_data.crop_type
        )
        
        return FertilizerRecommendationOutput(
            recommended_fertilizer=recommended_fertilizer,
            npk_ratio=npk_ratio,
            application_rate=application_rate
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@app.post("/disease/detect", response_model=DiseaseDetectionOutput, tags=["Disease Detection"])
async def detect_disease(
    image: UploadFile = File(..., description="Plant image for disease detection")
):
    """
    Detect plant disease from uploaded image
    
    - **image**: Plant image file (PNG, JPG, JPEG)
    """
    try:
        # Validate file type
        allowed_extensions = {".png", ".jpg", ".jpeg"}
        file_ext = os.path.splitext(image.filename)[1].lower()
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Read image bytes
        image_bytes = await image.read()
        
        # Predict disease
        disease_name, confidence, description, preventive_measures, supplement = predict_disease(image_bytes)
        
        return DiseaseDetectionOutput(
            disease_name=disease_name,
            confidence=confidence,
            description=description,
            preventive_measures=preventive_measures,
            supplement_recommendation=supplement
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Disease detection error: {str(e)}")


@app.post("/yield/predict", response_model=CropYieldOutput, tags=["Yield Prediction"])
async def predict_crop_yield(input_data: CropYieldInput):
    """
    Predict crop yield based on various parameters
    
    - **crop**: Crop name
    - **crop_year**: Year of cultivation
    - **season**: Season (Kharif, Rabi, Zaid)
    - **state**: State name
    - **area**: Area in hectares
    - **production**: Production in tonnes
    - **annual_rainfall**: Annual rainfall in mm
    - **fertilizer**: Fertilizer usage in kg/ha
    - **pesticide**: Pesticide usage in kg/ha
    """
    try:
        predicted_yield, confidence, recommendations = predict_yield(
            input_data.crop, input_data.crop_year, input_data.season,
            input_data.state, input_data.area, input_data.production,
            input_data.annual_rainfall, input_data.fertilizer, input_data.pesticide
        )
        
        return CropYieldOutput(
            predicted_yield=predicted_yield,
            confidence=confidence,
            recommendations=recommendations
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Yield prediction error: {str(e)}")


@app.post("/activity/log", response_model=ActivityResponse, tags=["Activity Logging"])
async def log_farmer_activity(activity: FarmerActivityLog):
    """
    Log farmer activity for analytics
    
    - **farmer_id**: Farmer ID
    - **activity_type**: Type of activity (crop_recommendation, disease_detection, etc.)
    - **input_data**: Input data for the activity
    - **output_data**: Output data from the activity
    """
    try:
        activity_id = log_activity(
            activity.farmer_id,
            activity.activity_type,
            activity.input_data,
            activity.output_data
        )
        
        return ActivityResponse(
            status="success",
            message="Activity logged successfully",
            activity_id=activity_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Activity logging error: {str(e)}")


@app.get("/crops", tags=["Reference Data"])
async def get_available_crops():
    """Get list of available crops from the dataset"""
    if crop_yield_data.empty:
        return {"crops": [], "message": "No crop data available"}
    
    crops = sorted(crop_yield_data['Crop'].unique().tolist())
    return {"crops": crops, "count": len(crops)}


@app.get("/states", tags=["Reference Data"])
async def get_available_states():
    """Get list of available states from the dataset"""
    if crop_yield_data.empty:
        return {"states": [], "message": "No state data available"}
    
    states = sorted(crop_yield_data['State'].unique().tolist())
    return {"states": states, "count": len(states)}


@app.get("/seasons", tags=["Reference Data"])
async def get_available_seasons():
    """Get list of available seasons"""
    seasons = ["Kharif", "Rabi", "Zaid"]
    return {"seasons": seasons}


@app.get("/diseases", tags=["Reference Data"])
async def get_disease_info():
    """Get disease information from the database"""
    if disease_info.empty:
        return {"diseases": [], "message": "No disease data available"}
    
    diseases = []
    for idx, row in disease_info.iterrows():
        diseases.append({
            "id": idx,
            "name": row.get('disease_name', 'Unknown'),
            "description": row.get('description', ''),
            "preventive_measures": row.get('Possible Steps', '')
        })
    
    return {"diseases": diseases, "count": len(diseases)}


# Error Handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "status_code": 500
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
