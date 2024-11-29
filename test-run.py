import cv2
import numpy as np
from ultralytics import YOLO
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load the model
try:
    model = YOLO("best.pt")
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    exit(1)

def run_detection(image_path):
    """
    Perform object detection on an image.

    Args:
        image_path (str): Path to the image file.

    Returns:
        None. Logs detection details to console.
    """
    try:
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Invalid image file or path")
        logger.info(f"Image loaded successfully from {image_path}")

        # Perform detection
        logger.info("Running object detection...")
        results = model(img)

        # Process results
        logger.info("Processing detection results...")
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                class_name = model.names[cls]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                logger.info(f"Detection: Class='{class_name}', Confidence={conf:.2f}, "
                            f"Box=({x1}, {y1}, {x2}, {y2})")

        logger.info("Object detection completed successfully.")
    except Exception as e:
        logger.error(f"Error during detection: {e}")

if __name__ == "__main__":
    image_path = "Untitled.jpg"  
    run_detection(image_path)
