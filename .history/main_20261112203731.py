from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Secure Spam Detection API",
    description="Spam classification using TP-IDF and Naive Bayes",
    version="1.0.0"
)

# Load the trained ML pipeline
MODEL_PATH = Path(__file__).parent / "models" / "spam_model.joblib"

model = joblib.load(MODEL_PATH)

class MessageInput(BaseModel):
    text: str = Field(
        min_length=1,
        max_length=1000,
        description="The message to be classified as spam or ham"
    )

@app.post("/predict")
def predict_spam(message: MessageInput):
    text = message.text.strip()
    
    if not text:
        raise HTTPException(
            status_code=400,
            detail="Message text cannot be empty."
        )
        
    prediction = model.predict([text])[0]
    
    probabilities = model.predict_proba([text])[0]
    spam_index = list(model.classes_).index("spam")
    
    return {
        "prediction": str(prediction),
        "spam_probability": round(float(probabilities[spam_index]), 4),
    }