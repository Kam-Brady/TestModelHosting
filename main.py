from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
from datetime import datetime
import numpy as np
import cv2
from ultralytics import YOLO
import io
from PIL import Image
import time
import os

app = FastAPI(title="YOLOv8 API", description="API for object detection using YOLOv8")

# Initialize the YOLO model
# Global variable for model
model = None

# Initialize model at module level
def initialize_model():
    global model
    try:
        print("Attempting to load YOLO model...")
        model = YOLO("yolov8n.pt")
        print("Model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

# Try to initialize immediately
model_initialized = initialize_model()

@app.on_event("startup")
async def startup_event():
    global model, model_initialized
    print("API starting up")
    
    # If model initialization failed earlier, try again
    if not model_initialized:
        print("Attempting to initialize model again during startup...")
        model_initialized = initialize_model()
        
    if model_initialized:
        print("Model is ready to use")
    else:
        print("WARNING: Model failed to initialize, API may not function correctly")

@app.get("/")
async def root():
    return {"message": "YOLOv8 Object Detection API"}

@app.get("/test")
async def test():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {"message": "This works!", "timestamp": timestamp}

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    try:
        # Check if model is loaded
        global model
        if model is None:
            # Try one more time to load the model
            print("Model not loaded. Attempting to load now...")
            if initialize_model():
                print("Model loaded successfully on demand")
            else:
                return JSONResponse(
                    status_code=500,
                    content={"success": False, "error": "Model could not be loaded. Please check server logs."}
                )
            
        # Read the uploaded image
        contents = await file.read()
        
        # Convert to OpenCV format
        try:
            image = np.array(Image.open(io.BytesIO(contents)))
        except Exception as img_error:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": f"Invalid image format: {str(img_error)}"}
            )
        
        # Run detection
        start_time = time.time()
        results = model(image)
        end_time = time.time()
        
        # Process results
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                
                detections.append({
                    "class": class_name,
                    "confidence": confidence,
                    "box": [round(x1), round(y1), round(x2), round(y2)]
                })
        
        # Prepare response
        response = {
            "success": True,
            "processing_time": round(end_time - start_time, 3),
            "detections": detections
        }
        
        return JSONResponse(content=response)
    
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"Error in detect_objects: {error_detail}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

@app.get("/models")
async def get_models():
    return {"models": ["yolov8n.pt"]}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=80, reload=True)