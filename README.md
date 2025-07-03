# YOLO Inference Server

A simple FastAPI server for YOLO object detection inference.

## Features

- Load YOLO models (both pre-trained and custom finetuned)
- Perform inference on uploaded images
- Support for both file upload and base64 image input
- Configurable confidence and IoU thresholds
- RESTful API with automatic documentation

## Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Starting the Server

Basic usage:
```bash
python yolo_inference_server.py
```

With custom options:
```bash
python yolo_inference_server.py --host 0.0.0.0 --port 8000 --model-path path/to/your/model.pt
```

Available arguments:
- `--host`: Host to bind the server (default: 127.0.0.1)
- `--port`: Port to bind the server (default: 8000)
- `--model-path`: Path to YOLO model file
- `--reload`: Enable auto-reload for development

### API Documentation

After starting the server, you can view the interactive API documentation at:
```
http://<server_ip>:<server_port>/docs
```

For example, if running locally on port 8000:
```
http://localhost:8000/docs
```

## API Endpoints

### GET /
Get server status and model information

### POST /load_model
Load or reload a YOLO model
- Query parameter: `model_path` (string) - Path to model file

### POST /predict
Perform inference on uploaded image
- Form data: `file` (image file)
- Query parameters:
  - `conf_threshold` (float, 0.0-1.0) - Confidence threshold (default: 0.25)
  - `iou_threshold` (float, 0.0-1.0) - IoU threshold for NMS (default: 0.5)

### POST /predict_base64
Perform inference on base64 encoded image
- Query parameters:
  - `image_base64` (string) - Base64 encoded image
  - `conf_threshold` (float, 0.0-1.0) - Confidence threshold (default: 0.25)
  - `iou_threshold` (float, 0.0-1.0) - IoU threshold for NMS (default: 0.5)

### GET /model_info
Get detailed information about the loaded model

### GET /health
Health check endpoint

## Model Loading

The server attempts to load models in the following order:
1. Path specified by `YOLO_MODEL_PATH` environment variable
2. `runs/detect/metasejong_objects/weights/best.pt` (default)
3. Alternative paths if default not found

## Response Format

All prediction endpoints return JSON responses with:
- `success`: Boolean indicating success
- `message`: Status message
- `detections`: Array of detected objects with:
  - `class_id`: Object class ID
  - `class_name`: Object class name
  - `confidence`: Detection confidence score
  - `bbox`: Bounding box coordinates [x1, y1, x2, y2]
- `image_size`: Image dimensions [width, height]
- `inference_time`: Time taken for inference in seconds

## Requirements

See `requirements.txt` for complete list of dependencies.

## Environment Variables

- `YOLO_MODEL_PATH`: Path to YOLO model file (optional)