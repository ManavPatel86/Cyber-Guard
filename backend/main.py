from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import pickle,joblib

app = FastAPI()

frontend_origins = [
    origin.strip()
    for origin in os.getenv(
        "FRONTEND_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]

# Allow your React frontend to communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=frontend_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Load the models
print("Loading model and vectorizer...")
try:
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    # Load the new compressed joblib model!
    model = joblib.load("model.joblib") 
except FileNotFoundError:
    print("Warning: Model files not found.")

# 2. Define the data format we expect from React
class EmailData(BaseModel):
    text: str

# 3. Create the prediction endpoint
@app.post("/predict")
def predict_spam(email: EmailData):
    # Vectorize the incoming text
    text_vectorized = vectorizer.transform([email.text]).toarray()
    
    # Get the probability score
    probability = model.predict_proba(text_vectorized)[0][1]
    
    # Apply our strict 80% threshold for high precision
    is_spam = bool(probability > 0.80)
    
    return {
        "is_spam": is_spam,
        "confidence": round(probability * 100, 2)
    }

# Root endpoint just to check if the server is alive
@app.get("/")
def read_root():
    return {"status": "Spam Classifier API is running!"}