from fastapi import FastAPI, File, UploadFile, HTTPException
from ultralytics import YOLO
import numpy as np
import cv2
import uvicorn
import logging
from typing import List, Dict
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the YOLO model
try:
    model = YOLO('best.pt')
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

# Create FastAPI app
app = FastAPI(title="YOLO Object Detection API")

class Prediction(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str

class DetectionResponse(BaseModel):
    predictions: List[Prediction]

@app.post("/detect", response_model=DetectionResponse)
async def detect_objects(file: UploadFile = File(...)):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Perform detection
        results = model(img)
        
        # Process results
        predictions = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Extract detection details
                cls = int(box.cls[0])
                class_name = model.names[cls]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                predictions.append(Prediction(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=conf,
                    class_id=cls,
                    class_name=class_name
                ))
        
        return DetectionResponse(predictions=predictions)
    
    except Exception as e:
        logger.error(f"Detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Optional: Swagger UI documentation and server launch
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)