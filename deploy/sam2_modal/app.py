import modal
import io

# Define the Modal App
app = modal.App("sam2-polyp-annotation")

# Build the custom image: Debian + Python 3.11 + PyTorch + SAM 2 + OpenCV
vol = modal.Volume.from_name("sam2-weights", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0", "wget")
    .pip_install("torch", "torchvision", "numpy", "Pillow", "opencv-python-headless", "fastapi[standard]")
    .run_commands(
        "pip install git+https://github.com/facebookresearch/segment-anything-2.git",
        "mkdir -p /weights",
        "wget -q https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt -O /weights/sam2_hiera_large.pt"
    )
)

@app.function(image=image, gpu="T4", volumes={"/weights": vol})
@modal.web_endpoint(method="POST")
def predict_mask(item: dict):
    """
    Accepts a base64 encoded image and a list of internal (positive) points.
    Returns the JSON representation of the predicted COCO polygon mask.
    """
    import torch
    import numpy as np
    import base64
    from PIL import Image
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    
    # 1. Parse request
    b64_img = item.get("image_b64", "")
    points = item.get("points", []) # e.g., [[x, y], ...]
    if not b64_img or not points:
        return {"error": "Missing image_b64 or points"}
    
    # 2. Decode image
    img_bytes = base64.b64decode(b64_img)
    image_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    image_np = np.array(image_pil)
    
    # 3. Load SAM 2
    checkpoint = "/weights/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
    
    # Suppress warnings for clean output
    import warnings
    warnings.filterwarnings('ignore')
    
    sam2_model = build_sam2(model_cfg, checkpoint, device="cuda")
    predictor = SAM2ImagePredictor(sam2_model)
    
    # 4. Predict
    predictor.set_image(image_np)
    input_point = np.array(points)
    input_label = np.array([1] * len(points)) # 1 = foreground
    
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )
    
    # 5. Convert binary mask to polygon (simplified for edge extraction)
    import cv2
    mask = (masks[0] > 0.0).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygon = []
    if contours:
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        polygon = largest_contour.flatten().tolist()
        
    return {
        "status": "success",
        "score": float(scores[0]),
        "polygon": polygon
    }
