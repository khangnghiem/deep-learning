"""
Modal.com deployment — Serverless YOLO11-seg endpoints for fast-diag.

Provides a FastAPI gateway routing to the hardware tier:
1. /detect/batch  -> L4 GPU-bound, executes PyTorch model via @modal.batched

Usage:
    modal deploy modal_app.py
"""

import base64
import io
import os
import time
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import modal
from pydantic import BaseModel


# --- Modal Infrastructure ---

app = modal.App("fast-diag-polyp-detection-yolo")

volume = modal.Volume.from_name("fast-diag-models", create_if_missing=True)

# Container image for PyTorch GPU
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch==2.1.2",
        "torchvision==0.16.2",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "fastapi[standard]",
        "pillow",
        "numpy<2.0.0",
        "opencv-python-headless",
        "ultralytics==8.4.33",
    )
)

MODEL_DIR = "/models/trained/polyp_detection/run/weights"
PT_WEIGHTS = os.path.join(MODEL_DIR, "best.pt")


# --- API Models ---

class DetectBatchRequest(BaseModel):
    images_base64: list[str]

class Detection(BaseModel):
    bbox: list[float]  # [x, y, w, h] absolute pixels
    bbox_normalized: list[float]  # [x1, y1, x2, y2] 0-1
    confidence: float
    segmentation: list[list[float]]  # polygon points
    area: float

class DetectResponse(BaseModel):
    detections: list[Detection]
    inference_ms: int
    model_source: str


# --- Core Inference Logic ---

def process_results(results_list) -> list[list[Detection]]:
    """Parse Ultralytics Results into API Detection format."""
    batch_detections = []
    for results in results_list:
        detections = []
        if len(results) == 0:
            batch_detections.append(detections)
            continue
            
        orig_img = results.orig_img
        h, w = orig_img.shape[:2]
        
        boxes = results.boxes
        masks = results.masks
        
        for i in range(len(boxes)):
            box = boxes[i]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            
            segmentation = []
            if masks is not None:
                poly = masks.xy[i]
                if len(poly) >= 3:
                    flat = []
                    for px, py in poly:
                        flat.extend([float(px), float(py)])
                    segmentation.append(flat)

            detections.append(Detection(
                bbox=[float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                bbox_normalized=[float(x1/w), float(y1/h), float(x2/w), float(y2/h)],
                confidence=conf,
                segmentation=segmentation,
                area=float((x2 - x1) * (y2 - y1)),
            ))
            
        batch_detections.append(detections)
        
    return batch_detections


def _decode_b64(b64_str: str):
    from PIL import Image
    import numpy as np
    header, encoded = b64_str.split(",", 1) if "," in b64_str else ("", b64_str)
    image_bytes = base64.b64decode(encoded)
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.array(pil_img)


# --- Hardware Tiers ---

@app.function(image=image, gpu="L4", volumes={"/models": volume}, scaledown_window=30)
@modal.batched(max_batch_size=16, wait_ms=250)
def infer_gpu_batched(images_b64: list[str]) -> list[dict]:
    """Run PyTorch dynamically batched on L4 GPU."""
    import time
    from ultralytics import YOLO
    
    start = time.monotonic()
    
    if not hasattr(infer_gpu_batched, "model"):
        infer_gpu_batched.model = YOLO(PT_WEIGHTS if os.path.exists(PT_WEIGHTS) else "yolo11n-seg.pt")
        
    imgs_np = [_decode_b64(b64) for b64 in images_b64]
    
    results_list = infer_gpu_batched.model(imgs_np, verbose=False)
    batch_detections = process_results(results_list)
    
    output = []
    ms_per_image = int(((time.monotonic() - start) * 1000) / max(1, len(images_b64)))
    
    for detections in batch_detections:
        output.append({
            "detections": [d.model_dump() for d in detections],
            "inference_ms": ms_per_image,
            "model_source": "gpu-l4-batched" if os.path.exists(PT_WEIGHTS) else "gpu-fallback-batched",
        })
        
    return output


# --- API Gateway ---

@app.function(image=image)
@modal.asgi_app()
def serve():
    web_app = FastAPI(title="Fast-Diag Polyp YOLO11 Bulk")

    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @web_app.get("/health")
    def health_check():
        return {"status": "ok", "framework": "YOLO11 GPU Bulk"}

    @web_app.post("/detect/batch", response_model=list[DetectResponse])
    def detect_batch(request: DetectBatchRequest):
        try:
            responses = []
            for r in infer_gpu_batched.map(request.images_base64):
                responses.append(DetectResponse(**r))
            return responses
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return web_app
