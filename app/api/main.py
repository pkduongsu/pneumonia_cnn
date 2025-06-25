from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from api.model import ModelPredictor
from PIL import Image
import os
import io
import uuid
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pneumonia_classifier")

app = FastAPI(title="Pneumonia Classifier")

UPLOAD_DIR = Path("uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
# Load model
try:
    predictor = ModelPredictor(model_path="models/pneumonia_detection_model.pth")
    logger.info("✅ Model loaded successfully")
except Exception as e:
    logger.error(f"❌ Model loading failed: {e}")
    predictor = None

@app.get("/")
def root():
    return {"message": "Pneumonia Detection API running."}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file:
        return {"error": "No file uploaded"}
    
    if predictor is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    logger.info(f"Received: {file.filename}")
    
    try:
        # Read file content
        content = await file.read()
        pil_image = Image.open(io.BytesIO(content))
        
        # Make prediction directly without saving to disk
        result = predictor.predict(pil_image)
        logger.info(f"Prediction result: {result}")
        
        return result
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
        