"""FastAPI app exposing the trained model.

Run with (from the diabetes-predictor/ root):
    uvicorn api.main:app --reload
"""

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.predict import predict

FEATURE_ORDER = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]

app = FastAPI(title="Diabetes Predictor API")


class PatientFeatures(BaseModel):
    """One row of the sklearn diabetes dataset's 10 input features."""
    age: float
    sex: float
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float


@app.get("/")
def root():
    return {"status": "ok", "message": "Diabetes Predictor API is running"}


@app.post("/predict")
def make_prediction(patient: PatientFeatures):
    try:
        row = pd.DataFrame([patient.model_dump()], columns=FEATURE_ORDER)
        prediction = predict(row)
        return {"prediction": float(prediction[0])}
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Model file not found. Run `python -m src.train` first.",
        )
