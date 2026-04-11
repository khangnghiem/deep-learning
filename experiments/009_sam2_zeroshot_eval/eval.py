import os, json, yaml, sys, cv2
import torch
import numpy as np
from ultralytics import YOLO
from PIL import Image
from tqdm import tqdm
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config.paths import PRETRAINED

# SAM2 model — lives in models/pretrained/sam2/
# Download source: https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
SAM2_CHECKPOINT = PRETRAINED / "sam2" / "sam2_hiera_large.pt"

print("Loading SAM-2...")
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    if not SAM2_CHECKPOINT.exists():
        print(f"Checkpoint not found at {SAM2_CHECKPOINT}, downloading...")
        os.system(f"wget -q https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt -O {SAM2_CHECKPOINT}")
    sam2_model = build_sam2("sam2_hiera_l.yaml", str(SAM2_CHECKPOINT), device="cuda")
    predictor = SAM2ImagePredictor(sam2_model)
except Exception as e:
    print("SAM-2 load failed:", e)

# Note: The script is launched from the experiment directory
print("Loading config...")
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

yolo_ckpt = cfg['model']['yolo_checkpoint']
coco_json_path = cfg['data']['coco_json']

print(f"Loading YOLOv8 from {yolo_ckpt}")
# Patch ultralytics settings to avoid interactivity
os.environ["YOLO_VERBOSE"] = "False"
yolo = YOLO(yolo_ckpt)

with open(coco_json_path) as f:
    coco = json.load(f)

if 'image_root' in cfg['data']:
    data_dir = cfg['data']['image_root']
else:
    data_dir = '/content/drive/MyDrive/data_lake/01_bronze_medical/polypgen'
    print(f"Warning: image_root not in config, defaulting to {data_dir}")
out_dir = "predictions"
os.makedirs(out_dir, exist_ok=True)
from pathlib import Path

import mlflow
mlflow.set_tracking_uri('file:///content/drive/MyDrive/mlflow/mlruns')

experiment_name = cfg['experiment']['name']
mlflow.set_experiment(experiment_name)

with mlflow.start_run(run_name=f"{experiment_name}_eval"):
    count = 0
    generated = 0
    # Process just 20 images to save time and prove it works end-to-end
    for img_info in tqdm(coco['images'][:20], leave=False, unit="batch"):
        rel_path = img_info['file_name']
        base_name = os.path.basename(rel_path).replace('polypgen_', '')
        
        img_path = None
        for p in Path(data_dir).rglob(f"*{base_name}"):
            img_path = str(p)
            break
            
        if not img_path or not os.path.exists(img_path):
            print(f"File not found recursively for: {base_name}")
            continue
        
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # 1. YOLO prediction
        res = yolo(img_rgb, verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy()
        
        # 2. SAM-2 prediction
        try:
            predictor.set_image(img_rgb)
            for box in boxes:
                masks, scores, logits = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=box[None, :],
                    multimask_output=False,
                )
                mask = masks[0]
                
                # Overlay
                color = np.array([30/255, 144/255, 255/255, 0.6])
                h, w = mask.shape[-2:]
                mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
                mask_image = (mask_image * 255).astype(np.uint8)
                img_rgb = cv2.addWeighted(img_rgb, 1.0, mask_image[:,:,:3], 0.6, 0)
                generated += 1
                
        except Exception as e:
            print(f"SAM-2 error: {e}")
            
        cv2.imwrite(os.path.join(out_dir, os.path.basename(img_path)), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        count += 1
        
    print(f"✅ Evaluation complete. Processed {count} images, generated {generated} masks.")
    mlflow.log_metric('eval_images', count)
    mlflow.log_metric('masks_generated', generated)
