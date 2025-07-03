"""
Simple FastAPI server for YOLO inference
Updated to use standard ultralytics loading and FastAPI lifespan
"""

import os
import io
import base64
import torch
from typing import List, Optional
from contextlib import asynccontextmanager
from PIL import Image
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from ultralytics import YOLO
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model variable
model = None

class DetectionResult(BaseModel):
    """Detection result for a single object"""
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]

class InferenceResponse(BaseModel):
    """Response model for inference"""
    success: bool
    message: str
    detections: List[DetectionResult]
    image_size: List[int]  # [width, height]
    inference_time: float

class ServerStatus(BaseModel):
    """Server status response"""
    status: str
    model_loaded: bool
    model_path: str
    pytorch_version: str
    ultralytics_version: str

def load_model(model_path: str):
    """Load YOLO model using standard ultralytics method"""
    global model
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        logger.info(f"Loading YOLO model from: {model_path}")
        logger.info(f"PyTorch version: {torch.__version__}")
        
        # Load model using standard ultralytics method
        # This handles both pre-trained and custom finetuned models
        model = YOLO(model_path)
        
        # Store model path for reference
        model.model_path = model_path
        
        logger.info("✅ Model loaded successfully")
        logger.info(f"📊 Model classes: {list(model.names.values())}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to load model: {str(e)}")
        return False

def process_image(image_data: bytes) -> np.ndarray:
    """Convert uploaded image to numpy array"""
    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        image_array = np.array(image)
        
        return image_array
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI startup and shutdown"""
    # Startup
    logger.info("🚀 Starting up YOLO Inference Server...")
    
    # Get model path from environment variable or use default
    model_path = os.getenv("YOLO_MODEL_PATH", "runs/detect/metasejong_objects_augmented/weights/best.pt")
    
    # Try alternative paths if default doesn't exist
    if not os.path.exists(model_path):
        alternative_paths = [
            "/home/kwdahun/metasejong-airobotics/sources/yolo11n.pt",
            "yolo11n.pt",
            "yolov8n.pt",  # Common pretrained model
            "runs/detect/metasejong_objects/weights/last.pt"
        ]
        
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                model_path = alt_path
                logger.info(f"📍 Using alternative model path: {alt_path}")
                break
    
    success = load_model(model_path)
    if not success:
        logger.warning("⚠️ Server started but model loading failed. Use /load_model endpoint to load a model.")
    
    yield
    
    # Shutdown
    logger.info("🔄 Shutting down YOLO Inference Server...")
    global model
    if model:
        del model
        model = None
    logger.info("✅ Server shutdown complete")

# FastAPI app with lifespan
app = FastAPI(
    title="YOLO Inference Server",
    description="Simple FastAPI server for YOLO object detection inference",
    version="1.2.0",
    lifespan=lifespan
)

@app.get("/", response_model=ServerStatus)
async def root():
    """Get server status"""
    try:
        import ultralytics
        ultralytics_version = ultralytics.__version__
    except:
        ultralytics_version = "unknown"
    
    return ServerStatus(
        status="running",
        model_loaded=model is not None,
        model_path=getattr(model, 'model_path', 'Not loaded') if model else 'Not loaded',
        pytorch_version=torch.__version__,
        ultralytics_version=ultralytics_version
    )

@app.post("/load_model")
async def load_model_endpoint(model_path: str = Query(..., description="Path to YOLO model file")):
    """Load or reload YOLO model"""
    success = load_model(model_path)
    if success:
        return {
            "success": True, 
            "message": f"Model loaded successfully from: {model_path}",
            "pytorch_version": torch.__version__,
            "model_classes": list(model.names.values()) if model else []
        }
    else:
        raise HTTPException(status_code=500, detail="Failed to load model")

@app.post("/predict", response_model=InferenceResponse)
async def predict(
    file: UploadFile = File(..., description="Image file for inference"),
    conf_threshold: float = Query(0.25, ge=0.0, le=1.0, description="Confidence threshold"),
    iou_threshold: float = Query(0.5, ge=0.0, le=1.0, description="IoU threshold for NMS")
):
    """Perform YOLO inference on uploaded image"""
    
    # Check if model is loaded
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Use /load_model endpoint first.")
    
    try:
        # Read and process image
        image_data = await file.read()
        image_array = process_image(image_data)
        
        # Get image dimensions
        height, width = image_array.shape[:2]
        
        # Perform inference
        import time
        start_time = time.time()
        
        # Use ultralytics standard prediction method
        results = model.predict(
            image_array,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )
        
        inference_time = time.time() - start_time
        
        # Process results
        detections = []
        if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            
            # Get class names
            class_names = model.names
            
            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                x1, y1, x2, y2 = box.tolist()
                
                detection = DetectionResult(
                    class_id=int(cls_id),
                    class_name=class_names.get(cls_id, f'class_{cls_id}'),
                    confidence=float(conf),
                    bbox=[float(x1), float(y1), float(x2), float(y2)]
                )
                detections.append(detection)
        
        return InferenceResponse(
            success=True,
            message=f"Inference completed successfully. Found {len(detections)} objects.",
            detections=detections,
            image_size=[width, height],
            inference_time=inference_time
        )
        
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@app.post("/predict_base64", response_model=InferenceResponse)
async def predict_base64(
    image_base64: str = Query(..., description="Base64 encoded image"),
    conf_threshold: float = Query(0.25, ge=0.0, le=1.0, description="Confidence threshold"),
    iou_threshold: float = Query(0.5, ge=0.0, le=1.0, description="IoU threshold for NMS")
):
    """Perform YOLO inference on base64 encoded image"""
    
    # Check if model is loaded
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Use /load_model endpoint first.")
    
    try:
        # Decode base64 image
        try:
            image_data = base64.b64decode(image_base64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {str(e)}")
        
        image_array = process_image(image_data)
        
        # Get image dimensions
        height, width = image_array.shape[:2]
        
        # Perform inference
        import time
        start_time = time.time()
        
        results = model.predict(
            image_array,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )
        
        inference_time = time.time() - start_time
        
        # Process results
        detections = []
        if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            
            # Get class names
            class_names = model.names
            
            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                x1, y1, x2, y2 = box.tolist()
                
                detection = DetectionResult(
                    class_id=int(cls_id),
                    class_name=class_names.get(cls_id, f'class_{cls_id}'),
                    confidence=float(conf),
                    bbox=[float(x1), float(y1), float(x2), float(y2)]
                )
                detections.append(detection)
        
        return InferenceResponse(
            success=True,
            message=f"Inference completed successfully. Found {len(detections)} objects.",
            detections=detections,
            image_size=[width, height],
            inference_time=inference_time
        )
        
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@app.get("/model_info")
async def get_model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        return {
            "model_loaded": True,
            "model_type": str(type(model)),
            "model_path": getattr(model, 'model_path', 'Unknown'),
            "class_names": model.names,
            "number_of_classes": len(model.names) if model.names else 0,
            "pytorch_version": torch.__version__,
            "device": str(model.device) if hasattr(model, 'device') else 'Unknown'
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": __import__('datetime').datetime.now().isoformat()
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="YOLO Inference Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server")
    parser.add_argument("--model-path", help="Path to YOLO model file")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    # Set model path as environment variable if provided
    if args.model_path:
        os.environ["YOLO_MODEL_PATH"] = args.model_path
    
    print("🚀 Starting YOLO Inference Server...")
    print(f"🔧 PyTorch version: {torch.__version__}")
    print(f"📡 Server will be available at: http://{args.host}:{args.port}")
    print(f"📖 API documentation: http://{args.host}:{args.port}/docs")
    print(f"🏥 Health check: http://{args.host}:{args.port}/health")
    
    uvicorn.run(
        app, 
        host=args.host, 
        port=args.port,
        reload=args.reload
    )