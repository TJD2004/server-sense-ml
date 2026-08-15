import os
import json
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib

app = FastAPI(
    title="SolarSense ML Generation Prediction Microservice",
    description="Scikit-Learn ML Model for Real-Time Solar Power Prediction & Anomaly Detection",
    version="1.0.0"
)

# Enable CORS for Express Backend & React Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_pipeline.joblib")
METRICS_PATH = os.path.join(os.path.dirname(__file__), "metrics.json")

model_pipeline = None
metrics_data = None

def load_artifacts():
    global model_pipeline, metrics_data
    
    # 1. Ensure model is trained & artifacts exist
    if not os.path.exists(MODEL_PATH) or not os.path.exists(METRICS_PATH):
        print("[SETUP] Model artifacts not found. Training model now...")
        from train_model import train_and_evaluate
        train_and_evaluate()
        
    try:
        model_pipeline = joblib.load(MODEL_PATH)
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)
        print("[OK] ML Model & Metrics loaded successfully!")
    except Exception as e:
        print(f"[ERROR] Error loading ML artifacts: {e}")

load_artifacts()

class PredictionRequest(BaseModel):
    hour: float = Field(12.0, ge=0, le=24, description="Hour of day (0-24)")
    month: int = Field(8, ge=1, le=12, description="Month of year (1-12)")
    temp: float = Field(28.0, ge=-10, le=60, description="Ambient Temperature (°C)")
    irradiance: float = Field(850.0, ge=0, le=1400, description="Solar Irradiance (W/m²)")
    cloudCoverage: float = Field(15.0, ge=0, le=100, description="Cloud Coverage (%)")
    humidity: float = Field(45.0, ge=0, le=100, description="Humidity (%)")
    windSpeed: float = Field(12.0, ge=0, le=100, description="Wind Speed (km/h)")
    capacityKW: float = Field(5.0, ge=0.5, le=50, description="System Capacity (kW)")

class PredictionResponse(BaseModel):
    predicted_solar_kw: float
    confidence_min: float
    confidence_max: float
    model_name: str
    r2_score: float
    mae: float

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "SolarSense Machine Learning Microservice",
        "model_loaded": model_pipeline is not None,
        "metrics_available": metrics_data is not None
    }

@app.get("/metrics")
def get_model_metrics():
    if not metrics_data:
        raise HTTPException(status_code=500, detail="Metrics not available")
    return metrics_data

@app.post("/predict", response_model=PredictionResponse)
def predict_solar_generation(req: PredictionRequest):
    if not model_pipeline:
        raise HTTPException(status_code=500, detail="ML Model not loaded")
        
    try:
        # Create single-sample pandas DataFrame with correct feature names
        input_data = pd.DataFrame([{
            "hour": req.hour,
            "month": req.month,
            "temp": req.temp,
            "irradiance": req.irradiance,
            "cloudCoverage": req.cloudCoverage,
            "humidity": req.humidity,
            "windSpeed": req.windSpeed,
            "capacityKW": req.capacityKW
        }])
        
        # Predict expected generation (kW)
        raw_pred = model_pipeline.predict(input_data)[0]
        predicted_kw = max(0.0, round(float(raw_pred), 2))
        
        # Nighttime safeguard (hour < 6 or hour > 19)
        if req.hour < 6.0 or req.hour > 19.5:
            predicted_kw = 0.0
            
        rf_metrics = metrics_data["models"]["random_forest"] if metrics_data else {"r2_score": 0.98, "mae": 0.08}
        mae_val = rf_metrics.get("mae", 0.08)
        
        return PredictionResponse(
            predicted_solar_kw=predicted_kw,
            confidence_min=max(0.0, round(predicted_kw - mae_val * 1.5, 2)),
            confidence_max=round(predicted_kw + mae_val * 1.5, 2),
            model_name="Random Forest Regressor (Scikit-Learn)",
            r2_score=rf_metrics.get("r2_score", 0.98),
            mae=mae_val
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
