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

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create temp directory
TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

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
    
    logger.info(f"Received: {file.filename}")
    
    # Read file content
    content = await file.read()
    pil_image = Image.open(io.BytesIO(content))

    width, height = pil_image.size
        
    # Save uploaded image
    file_extension = file.filename.split(".")[-1]
    unique_filename = f"{uuid.uuid4()}.{file_extension}"
    file_path = UPLOAD_DIR / unique_filename
        
    with open(file_path, "wb") as buffer:
        buffer.write(content)
    
    try:
        result = predictor.predict(pil_image)
        return result
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")
    
        