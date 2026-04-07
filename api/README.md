# KrishiDisha FastAPI Module

## Overview
Complete FastAPI implementation for the KrishiDisha Agriculture LLM project, providing RESTful APIs for:
- Agricultural chatbot assistance
- Crop recommendation based on soil parameters
- Fertilizer recommendation
- Plant disease detection from images
- Crop yield prediction
- Activity logging and analytics

## Project Structure
```
/workspace/
├── api/
│   ├── __init__.py          # Package initialization
│   ├── models.py            # Pydantic data models
│   ├── main.py              # FastAPI application with all endpoints
│   └── requirements.txt     # API-specific dependencies
├── krishidisha_bot.py       # Chatbot logic (existing)
├── CNN.py                   # Disease detection model architecture (existing)
├── data/                    # Data files (existing)
│   ├── disease_info.csv
│   ├── supplement_info.csv
│   └── crop_yield.csv
└── models/                  # ML models (to be placed here)
    ├── crop_recommendation_model.pkl
    ├── label_encoder.pkl
    ├── fertilizer_recommendation_model.pkl
    ├── fertilizer_label_encoder.pkl
    ├── plant_disease_model_1_latest.pt
    └── pipeline_yield_prediction_pipeline.pkl
```

## Installation

### 1. Install Dependencies
```bash
pip install -r api/requirements.txt
```

### 2. Ensure Model Files
Place your trained ML models in the `/workspace/models/` directory:
- `crop_recommendation_model.pkl`
- `label_encoder.pkl`
- `fertilizer_recommendation_model.pkl`
- `fertilizer_label_encoder.pkl`
- `plant_disease_model_1_latest.pt`
- `pipeline_yield_prediction_pipeline.pkl`

## Running the API

### Development Mode
```bash
cd /workspace
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode
```bash
cd /workspace
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Endpoints

### Root & Health
- **GET /** - API information and available endpoints
- **GET /health** - Health check endpoint

### Chatbot
- **POST /chat** - Chat with KrishiDisha agricultural assistant
  ```json
  {
    "message": "What crops grow well in clay soil?",
    "farmer_id": 123
  }
  ```

### Crop Recommendation
- **POST /crop/recommend** - Get crop recommendations
  ```json
  {
    "N": 90,
    "P": 42,
    "K": 43,
    "temperature": 20.5,
    "humidity": 82,
    "ph": 6.5,
    "rainfall": 202
  }
  ```

### Fertilizer Recommendation
- **POST /fertilizer/recommend** - Get fertilizer suggestions
  ```json
  {
    "N": 60,
    "P": 30,
    "K": 40,
    "soil_type": "loamy",
    "crop_type": "rice"
  }
  ```

### Disease Detection
- **POST /disease/detect** - Detect plant disease from image
  ```bash
  curl -X POST "http://localhost:8000/disease/detect" \
       -H "Content-Type: multipart/form-data" \
       -F "image=@path/to/plant_image.jpg"
  ```

### Yield Prediction
- **POST /yield/predict** - Predict crop yield
  ```json
  {
    "crop": "Rice",
    "crop_year": 2023,
    "season": "Kharif",
    "state": "Punjab",
    "area": 100,
    "production": 300,
    "annual_rainfall": 1200,
    "fertilizer": 150,
    "pesticide": 2.5
  }
  ```

### Reference Data
- **GET /crops** - List of available crops
- **GET /states** - List of available states
- **GET /seasons** - List of seasons
- **GET /diseases** - Disease information database

### Activity Logging
- **POST /activity/log** - Log farmer activities
  ```json
  {
    "farmer_id": 123,
    "activity_type": "crop_recommendation",
    "input_data": {"N": 90, "P": 42, "K": 43},
    "output_data": {"recommended_crop": "Rice"}
  }
  ```

## API Documentation

Once running, access interactive documentation at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Example Usage

### Python Client Example
```python
import requests

# Crop Recommendation
response = requests.post("http://localhost:8000/crop/recommend", json={
    "N": 90, "P": 42, "K": 43,
    "temperature": 20.5, "humidity": 82,
    "ph": 6.5, "rainfall": 202
})
print(response.json())

# Chat
response = requests.post("http://localhost:8000/chat", json={
    "message": "How to prevent wheat rust?"
})
print(response.json())

# Disease Detection
with open("leaf_image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/disease/detect",
        files={"image": f}
    )
    print(response.json())
```

### cURL Examples
```bash
# Health Check
curl http://localhost:8000/health

# Chat
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"message": "What is the best fertilizer for rice?"}'

# Crop Recommendation
curl -X POST "http://localhost:8000/crop/recommend" \
     -H "Content-Type: application/json" \
     -d '{"N":90,"P":42,"K":43,"temperature":20.5,"humidity":82,"ph":6.5,"rainfall":202}'
```

## Features

✅ **Complete Pydantic Models** - Type-safe request/response validation  
✅ **Error Handling** - Comprehensive error responses with status codes  
✅ **CORS Support** - Cross-origin resource sharing enabled  
✅ **Auto Documentation** - Interactive Swagger UI and ReDoc  
✅ **ML Integration** - Seamless integration with existing ML models  
✅ **Chatbot Integration** - Uses KrishiDisha bot knowledge base  
✅ **Activity Logging** - Track farmer interactions for analytics  
✅ **Reference Data APIs** - Access crops, states, seasons, diseases  

## Configuration

### Environment Variables (Optional)
```bash
export API_HOST=0.0.0.0
export API_PORT=8000
export MODEL_PATH=/workspace/models
export DATA_PATH=/workspace/data
```

### CORS Configuration
Edit `api/main.py` to customize allowed origins:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Production: specify domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Production Deployment

### Using Gunicorn
```bash
gunicorn api.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Docker (Optional)
Create a `Dockerfile`:
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY . .
RUN pip install -r api/requirements.txt
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Troubleshooting

### Model Not Found Errors
Ensure all model files are in `/workspace/models/` directory.

### Import Errors
Run from the workspace directory:
```bash
cd /workspace
python -m uvicorn api.main:app --reload
```

### Port Already in Use
Change the port:
```bash
uvicorn api.main:app --port 8001
```

## License
KrishiDisha Project - Agriculture LLM API
