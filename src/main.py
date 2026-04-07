from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import xgboost as xgb
import json
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

app = FastAPI(title="SentinelScore Credit Risk Engine", version="1.0.0")

MODEL = None
THRESHOLD = 0.5
EXPECTED_FEATURES = []

@app.on_event("startup")
def load_artifacts():
    global MODEL, THRESHOLD, EXPECTED_FEATURES
    print("🚀 Booting up SentinelScore Engine...")
    
    try:
        with open("models/threshold.json", "r") as f:
            config = json.load(f)
            THRESHOLD = config.get("optimal_threshold", 0.5)
            EXPECTED_FEATURES = config.get("features", [])
        print(f"✅ Config loaded. Threshold: {THRESHOLD:.4f}")
    except Exception as e:
        print("⚠️ Could not load threshold.json.")

    try:
        # Load the physical model file directly! No MLflow required.
        MODEL = xgb.XGBClassifier()
        MODEL.load_model("models/production_model.json")
        print("✅ Production XGBoost Model loaded successfully.")
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Failed to load model. {e}")

class ScoringRequest(BaseModel):
    features: dict 

@app.get("/")
def health_check():
    return {"status": "online"}

@app.post("/predict")
def predict_risk(request: ScoringRequest):
    if not MODEL:
        raise HTTPException(status_code=500, detail="Model is offline.")
    
    try:
        df = pd.DataFrame([request.features])
        
        missing_cols = [col for col in EXPECTED_FEATURES if col not in df.columns]
        if missing_cols:
             for col in missing_cols:
                 df[col] = 0
                 
        df = df[EXPECTED_FEATURES]
        y_proba = MODEL.predict_proba(df)[:, 1][0]
        
        decision = "REJECT" if y_proba > THRESHOLD else "APPROVE"
        
        return {
            "applicant_status": decision,
            "default_probability": round(float(y_proba), 4),
            "threshold_applied": THRESHOLD,
            "risk_tier": "HIGH" if y_proba > THRESHOLD else ("MEDIUM" if y_proba > THRESHOLD * 0.7 else "LOW")
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")