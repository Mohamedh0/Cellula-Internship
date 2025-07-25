import os
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from io import BytesIO
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Define paths and constants
MODEL_PATH = "models/best_model.h5"
IMG_SIZE = (224, 224)
LABEL_MAP = {0: "CaS", 1: "CoS", 2: "Gum", 3: "MC", 4: "OC", 5: "OLP", 6: "OT"}

# Load the Keras model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    raise Exception(f"Failed to load Keras model: {str(e)}")

def preprocess_image(image: Image.Image):
    """Preprocess the input image."""
    image = image.resize(IMG_SIZE)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the frontend HTML page."""
    with open("static/index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Handle image uploads and return predictions."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and preprocess the image
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        processed_image = preprocess_image(image)
        
        # Make prediction
        prediction = model.predict(processed_image)
        predicted_class_idx = np.argmax(prediction, axis=1)[0]
        predicted_label = LABEL_MAP.get(predicted_class_idx)
        confidence = float(prediction[0][predicted_class_idx])
        
        return {
            "predicted_class": predicted_label,
            "confidence": confidence
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "Teeth Classification API is running"}
