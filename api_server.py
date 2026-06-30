"""
Aditya-L1 Solar Flare Prediction API Server.

This FastAPI server runs a continuous processing engine that accepts sliding windows
of soft and hard X-ray flux data (SoLEXS and HEL1OS), normalizes the measurements,
and executes parallel Nowcasting (signal processing tripwire) and Forecasting (PyTorch LSTM)
analysis pipelines to return real-time prediction and alert responses.
"""

import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

# Import the recommended Improved LSTM model
from models import ImprovedFlareForecasterLSTM

# Global container to store the preloaded model and runtime device
ml_models = {}

# Physical normalization constants derived from instrument baseline statistics:
# SoLEXS counts stats: Mean: 722.7386, Std: 1324.5471
# HEL1OS counts stats: Mean: 9.304749, Std: 92.98037
SOLEXS_MEAN = 722.7386
SOLEXS_STD = 1324.5471
HEL1OS_MEAN = 9.304749
HEL1OS_STD = 92.98037

# Physics-based static thresholds for flare nowcasting
STATIC_SOLEXS_THRESHOLD = 1000.0
STATIC_HEL1OS_THRESHOLD = 200.0


# ============================================================================
# Pydantic Schemas (DTOs)
# ============================================================================

class HistoryPoint(BaseModel):
    timestamp: str = Field(
        ...,
        description="ISO 8601 formatted datetime string"
    )
    solexs_flux: float = Field(
        ...,
        description="Soft X-ray intensity measurement (SoLEXS counts/s), castable to float32"
    )
    hel1os_flux: float = Field(
        ...,
        description="Hard X-ray intensity measurement (HEL1OS counts/s), castable to float32"
    )


class SlidingWindowPayload(BaseModel):
    sequence_length: int = Field(
        ...,
        description="Number of data points in the history sequence (used for validation)"
    )
    history_sequence: List[HistoryPoint] = Field(
        ...,
        description="Chronological list of recent X-ray flux measurements (sliding window)"
    )


class NowcastEngineResponse(BaseModel):
    flare_detected: bool = Field(
        ...,
        description="True if the signal-processing tripwire algorithm detects a current spike"
    )
    trigger_timestamp: Optional[str] = Field(
        None,
        description="The exact ISO 8601 timestamp the spike was detected. Null if no flare is detected"
    )
    flare_class: Optional[str] = Field(
        None,
        description="Estimated flare severity class (e.g. 'A-Class', 'B-Class', 'C-Class', 'M-Class', 'X-Class')"
    )


class ForecastEngineResponse(BaseModel):
    flare_probability: float = Field(
        ...,
        description="AI-generated flare probability score between 0.0 and 1.0"
    )
    flare_probability_percent: float = Field(
        ...,
        description="AI-generated flare probability percentage between 0.0 and 100.0"
    )
    estimated_lead_time_minutes: int = Field(
        ...,
        description="Estimated lead time in minutes until peak flux is reached"
    )
    alert_triggered: bool = Field(
        ...,
        description="True if the flare probability exceeds the defined safety threshold (> 85%)"
    )


class PredictionResponse(BaseModel):
    status: str = Field(
        ...,
        description="'success' or 'error'"
    )
    processed_timestamp: str = Field(
        ...,
        description="Timestamp of the most recent data point analyzed in the window"
    )
    nowcast_engine: NowcastEngineResponse
    forecast_engine: ForecastEngineResponse


# ============================================================================
# Lifespan Management (Model Pre-loading)
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events. The PyTorch LSTM model weights
    are loaded into memory only once when the server starts up.
    """
    # 1. Determine execution device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"INFO: Initializing API Server. Using device: {device}")
    
    # 2. Instantiate the model architecture
    model = ImprovedFlareForecasterLSTM(
        input_dim=2,
        hidden_dim=128,
        num_layers=3,
        num_heads=4,
        dropout=0.3,
        output_dim=1,
        bidirectional=True
    )
    
    # 3. Load trained parameters from disk
    weights_path = os.getenv("MODEL_WEIGHTS_PATH", "flare_lstm_weights.pth")
    if not os.path.exists(weights_path):
        # Fallback to absolute path or display descriptive warning
        raise FileNotFoundError(
            f"CRITICAL ERROR: Model weights file '{weights_path}' was not found. "
            f"Please verify the workspace root or set the MODEL_WEIGHTS_PATH environment variable."
        )
    
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.to(device)
        # Put the model in evaluation mode once on startup (disables dropout layers, etc.)
        model.eval()
        
        ml_models["model"] = model
        ml_models["device"] = device
        print("INFO: PyTorch LSTM model weights loaded successfully into memory.")
    except Exception as e:
        raise RuntimeError(f"CRITICAL ERROR: Failed to load model weights. Details: {e}")
        
    yield
    
    # Clean up on shutdown
    ml_models.clear()
    print("INFO: Shutting down API Server. Model unloaded from memory.")


# ============================================================================
# FastAPI Application & Endpoints
# ============================================================================

app = FastAPI(
    title="Aditya-L1 Solar Flare Prediction API",
    description="Backend API system for real-time solar flare detection and forecasting",
    version="1.0.0",
    lifespan=lifespan
)


@app.post(
    "/api/v1/analyze_flux",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK
)
async def analyze_flux(payload: SlidingWindowPayload):
    """
    Accepts a chronological sequence of Soft and Hard X-ray measurements, aligns the data,
    computes real-time spike indicators (Nowcasting), and runs PyTorch LSTM inference (Forecasting).
    """
    try:
        # ===== Step 1: Validation =====
        actual_len = len(payload.history_sequence)
        if actual_len != payload.sequence_length:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Invalid request payload: sequence_length parameter ({payload.sequence_length}) "
                    f"does not match actual number of points in history_sequence ({actual_len})."
                )
            )
        
        if actual_len == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid request payload: history_sequence must not be empty."
            )

        # Retrieve the latest chronological point (at the end of the sliding window)
        latest_point = payload.history_sequence[-1]
        processed_timestamp = latest_point.timestamp

        # Extract flux history sequences
        solexs_history = [p.solexs_flux for p in payload.history_sequence]
        hel1os_history = [p.hel1os_flux for p in payload.history_sequence]

        latest_solexs = latest_point.solexs_flux
        latest_hel1os = latest_point.hel1os_flux

        # ===== Step 2: Nowcast Engine (Signal Processing Tripwire) =====
        # Check both physical static thresholds and rolling std-deviation spikes.
        dynamic_solexs_triggered = False
        dynamic_hel1os_triggered = False

        if actual_len > 5:
            # Calculate mean and standard deviation of historical window excluding the latest element
            past_solexs = solexs_history[:-1]
            mean_s = float(np.mean(past_solexs))
            std_s = float(np.std(past_solexs))
            if std_s > 0 and latest_solexs > (mean_s + 3.0 * std_s):
                dynamic_solexs_triggered = True

            past_hel1os = hel1os_history[:-1]
            mean_h = float(np.mean(past_hel1os))
            std_h = float(np.std(past_hel1os))
            if std_h > 0 and latest_hel1os > (mean_h + 3.0 * std_h):
                dynamic_hel1os_triggered = True

        # Nowcast trigger conditions
        flare_detected = (
            latest_solexs > STATIC_SOLEXS_THRESHOLD or
            latest_hel1os > STATIC_HEL1OS_THRESHOLD or
            dynamic_solexs_triggered or
            dynamic_hel1os_triggered
        )

        trigger_timestamp = None
        flare_class = None

        if flare_detected:
            trigger_timestamp = processed_timestamp
            # Classify flare severity based on soft X-ray (SoLEXS) peak intensity
            if latest_solexs >= 1000.0:
                flare_class = "X-Class"
            elif latest_solexs >= 500.0:
                flare_class = "M-Class"
            elif latest_solexs >= 100.0:
                flare_class = "C-Class"
            elif latest_solexs >= 10.0:
                flare_class = "B-Class"
            else:
                flare_class = "A-Class"

        # ===== Step 3: Forecast Engine (LSTM Inference) =====
        # 1. Alignment & Synchronization:
        # Points are processed in chronological order. We align features as [normalized_solexs, normalized_hel1os].
        norm_sequence = []
        for p in payload.history_sequence:
            norm_s = (p.solexs_flux - SOLEXS_MEAN) / SOLEXS_STD
            norm_h = (p.hel1os_flux - HEL1OS_MEAN) / HEL1OS_STD
            norm_sequence.append([norm_s, norm_h])

        # 2. Sliding Window Extraction & Shape Enforcement:
        # Ensure the sequence length is exactly 60 (to match model training input dimension).
        EXPECTED_SEQ_LEN = 60
        if len(norm_sequence) > EXPECTED_SEQ_LEN:
            norm_sequence = norm_sequence[-EXPECTED_SEQ_LEN:]
        elif len(norm_sequence) < EXPECTED_SEQ_LEN:
            # Pad with zeros at the beginning
            pad_len = EXPECTED_SEQ_LEN - len(norm_sequence)
            norm_sequence = [[0.0, 0.0]] * pad_len + norm_sequence

        # Create a 3D NumPy array of shape (Batch_Size=1, Sequence_Length=60, Features=2)
        features_np = np.array([norm_sequence], dtype=np.float32)

        # 3. Retrieve Model from Lifespan cache
        model = ml_models.get("model")
        device = ml_models.get("device")
        if model is None or device is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Machine Learning Model has not been initialized. Please try again later."
            )

        # 4. Convert to PyTorch tensor and execute inference under torch.no_grad()
        x_tensor = torch.from_numpy(features_np).to(device)
        
        with torch.no_grad():
            outputs = model(x_tensor)
            if isinstance(outputs, tuple):
                probs = outputs[0]
            else:
                probs = outputs
            
            # Extract probability float value
            p_val = float(probs.squeeze().cpu().item())

        # Ensure probability is bound between 0.0 and 1.0
        p_val = max(0.0, min(1.0, p_val))
        flare_prob_percent = p_val * 100.0

        # Safety trigger threshold (> 85%)
        alert_triggered = flare_prob_percent > 85.0

        # ===== Step 4: Quantifiable Lead Time Heuristic =====
        # If flare probability is high (>= 50%), estimate lead time based on flux rate of change
        estimated_lead_time_minutes = 0
        if flare_prob_percent >= 50.0:
            # Look at derivative over the last 5 minutes (assuming 1-minute cadence)
            recent_fluxes = solexs_history[-5:] if len(solexs_history) >= 5 else solexs_history
            if len(recent_fluxes) >= 2:
                diffs = [recent_fluxes[i] - recent_fluxes[i-1] for i in range(1, len(recent_fluxes))]
                avg_diff = float(np.mean(diffs))
                
                if avg_diff > 0:
                    # Simple linear extrapolation: minutes to reach the X-class threshold (1000.0)
                    remaining_flux = max(0.0, STATIC_SOLEXS_THRESHOLD - latest_solexs)
                    estimated_lead = int(remaining_flux / avg_diff)
                    # Clip between 5 and 60 minutes
                    estimated_lead_time_minutes = max(5, min(60, estimated_lead))
                else:
                    estimated_lead_time_minutes = 30  # Default fallback when flux flatlines/decays but prob is high
            else:
                estimated_lead_time_minutes = 30  # Fallback for short window

        # ===== Step 5: Format Unified Response =====
        return PredictionResponse(
            status="success",
            processed_timestamp=processed_timestamp,
            nowcast_engine=NowcastEngineResponse(
                flare_detected=flare_detected,
                trigger_timestamp=trigger_timestamp,
                flare_class=flare_class
            ),
            forecast_engine=ForecastEngineResponse(
                flare_probability=round(p_val, 4),
                flare_probability_percent=round(flare_prob_percent, 2),
                estimated_lead_time_minutes=estimated_lead_time_minutes,
                alert_triggered=alert_triggered
            )
        )

    except HTTPException as he:
        # Re-raise standard FastAPI HTTP exceptions
        raise he
    except Exception as e:
        # Catch and return any other exception during tensor conversion, execution or formatting
        # to ensure the API does not silently crash
        print(f"ERROR: Exception occurred during analyze_flux handler. Details: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal API processing error: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    # Enable running the file directly for development purposes
    uvicorn.run("api_server:app", host="127.0.0.1", port=8000, reload=True)