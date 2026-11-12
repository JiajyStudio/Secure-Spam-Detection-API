from pathlib import Path

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(
    title="Secure Spam Detection API"
    description="Spam classification using TP-IDF and Naive Bayes",
    version="1.0.0"
)

# Load the trained ML pipeline
MODEL_PATH = Path(__file__).parent / "models" / "spam_model.joblib"

model = joblib.load(MODEL_PATH)