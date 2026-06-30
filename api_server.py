import os
from pathlib import Path
from typing import List
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Add current workspace directory to system path to import local files
import sys
sys.path.append(str(Path(__file__).parent.absolute()))

from nowcast import Nowcaster
from forecast import Forecaster

app = FastAPI(title="Aditya-L1 Solar Flare Nowcasting & Forecasting API Service")

# Paths to models
BASE_DIR = Path(__file__).parent.absolute()
NOWCAST_MODEL = BASE_DIR / "nowcast_model.pt"
NOWCAST_META = BASE_DIR / "nowcast_metadata.json"
FORECAST_MODEL = BASE_DIR / "forecast_model.pt"
FORECAST_META = BASE_DIR / "forecast_metadata.json"

# Global model instances
nowcaster_instance = None
forecaster_instance = None


class DataPoint(BaseModel):
    TIME: float
    COUNTS: float


class PredictionRequest(BaseModel):
    history: List[DataPoint]


def load_models():
    """Load or reload model instances from checkpoints if they exist."""
    global nowcaster_instance, forecaster_instance
    
    if NOWCAST_MODEL.exists() and NOWCAST_META.exists():
        try:
            nowcaster_instance = Nowcaster(NOWCAST_MODEL, NOWCAST_META)
            print("Successfully loaded Nowcaster model.")
        except Exception as e:
            print(f"Error loading Nowcaster model: {e}")
            
    if FORECAST_MODEL.exists() and FORECAST_META.exists():
        try:
            forecaster_instance = Forecaster(FORECAST_MODEL, FORECAST_META)
            print("Successfully loaded Forecaster model.")
        except Exception as e:
            print(f"Error loading Forecaster model: {e}")


@app.on_event("startup")
def startup_event():
    load_models()


@app.get("/status")
def get_status():
    """Check system health, model availability, and loaded parameters."""
    # Attempt to reload models in case they were trained after startup
    load_models()
    
    status = {
        "nowcaster_loaded": nowcaster_instance is not None,
        "forecaster_loaded": forecaster_instance is not None,
        "nowcast_checkpoint_exists": NOWCAST_MODEL.exists(),
        "forecast_checkpoint_exists": FORECAST_MODEL.exists(),
    }
    
    if nowcaster_instance:
        status["nowcast_metadata"] = nowcaster_instance.metadata
    if forecaster_instance:
        status["forecast_metadata"] = forecaster_instance.metadata
        
    return status


@app.post("/nowcast")
def run_nowcast(request: PredictionRequest):
    """
    Run flare nowcasting prediction.
    Expects at least 60 historic sequential records.
    """
    global nowcaster_instance
    if nowcaster_instance is None:
        load_models()
        if nowcaster_instance is None:
            raise HTTPException(status_code=503, detail="Nowcasting model is not loaded or trained yet.")
            
    if len(request.history) < 60:
        raise HTTPException(
            status_code=400, 
            detail=f"History length must be at least 60 steps, got {len(request.history)}"
        )
        
    try:
        # Convert request to DataFrame
        data = [{"TIME": pt.TIME, "COUNTS": pt.COUNTS} for pt in request.history]
        df = pd.DataFrame(data)
        
        prob = nowcaster_instance.predict(df)
        return {
            "flare_probability": prob,
            "is_active_flare": prob >= 0.5
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/forecast")
def run_forecast(request: PredictionRequest):
    """
    Run multi-task forecasting prediction.
    Expects at least 60 historic sequential records.
    """
    global forecaster_instance
    if forecaster_instance is None:
        load_models()
        if forecaster_instance is None:
            raise HTTPException(status_code=503, detail="Forecasting model is not loaded or trained yet.")
            
    if len(request.history) < 60:
        raise HTTPException(
            status_code=400, 
            detail=f"History length must be at least 60 steps, got {len(request.history)}"
        )
        
    try:
        # Convert request to DataFrame
        data = [{"TIME": pt.TIME, "COUNTS": pt.COUNTS} for pt in request.history]
        df = pd.DataFrame(data)
        
        preds = forecaster_instance.predict(df)
        
        # Map class label to scientific flare intensity
        class_mapping = {
            0: "None / Quiet",
            1: "B-class (Low Intensity)",
            2: "C-class (Moderate Intensity)",
            3: "M-class (High Intensity)",
            4: "X-class (Extreme Intensity)"
        }
        
        return {
            "flare_probability": preds["flare_prob"],
            "magnitude_class_id": preds["magnitude_class"],
            "predicted_flare_class": class_mapping.get(preds["magnitude_class"], "Unknown"),
            "model_confidence": preds["confidence"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8001, reload=True)