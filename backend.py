from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

# Load models
with open("lr_model.pkl", "rb") as f:
    lr_model = pickle.load(f)

with open("rf_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

app = FastAPI(title="Hospital Readmission Prediction API")

# Define the input structure
class PatientData(BaseModel):
    # Replace these with the exact feature names from your training
    feature1: float
    feature2: float
    feature3: float
    feature4: float
    feature5: float
    # ... add all 17 features here

@app.post("/predict")
def predict(data: PatientData):
    # Convert input to DataFrame
    input_df = pd.DataFrame([data.dict()])

    # Predict with Logistic Regression
    lr_pred = lr_model.predict(input_df)[0]
    lr_prob = lr_model.predict_proba(input_df)[0, 1]

    # Predict with Random Forest
    rf_pred = rf_model.predict(input_df)[0]
    rf_prob = rf_model.predict_proba(input_df)[0, 1]

    return {
        "logistic_regression": {"prediction": int(lr_pred), "probability": float(lr_prob)},
        "random_forest": {"prediction": int(rf_pred), "probability": float(rf_prob)}
    }
