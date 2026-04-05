import os
import json
import base64
import zlib
import cv2
import numpy as np
import shutil
from PIL import Image, ImageEnhance
import random

def base64_2_mask(s):
    z = zlib.decompress(base64.b64decode(s))
    n = np.frombuffer(z, np.uint8)
    mask = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)
    if len(mask.shape) == 3 and mask.shape[2] == 4:
        # Supervisely masks are often RGBA where A is the mask
        mask = mask[:, :, 3]
    return mask

landing_dir = r"g:\My Drive\data_lake\00_landing\polyp\ds"
bronze_dir = r"g:\My Drive\data_lake\01_bronze_medical\cvc_datasetninja"
silver_base = r"g:\My Drive\data_lake\02_silver\cvc_pseudo_video\train"

os.makedirs(os.path.join(bronze_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(bronze_dir, "masks"), exist_ok=True)
os.makedirs(os.path.join(silver_base, "clips"), exist_ok=True)

print("Ingesting CVC-ClinicDB (DatasetNinja) from Landing -> Bronze...")

ann_dir = os.path.join(landing_dir, "ann")
img_dir = os.path.join(landing_dir, "img")

if not os.path.exists(ann_dir):
    print("No annotation directory found. Did extraction finish?")
    exit()

valid_files = [f for f in os.listdir(ann_dir) if f.endswith('.json')]
print(f"Found {len(valid_files)} annotations.")

for ann_file in valid_files:
    item_id = ann_file.replace(".json", "") # e.g. "1.png"
    img_path = os.path.join(img_dir, item_id)
    
    if not os.path.exists(img_path):
        continue
        
    with open(os.path.join(ann_dir, ann_file), 'r') as f:
        ann = json.load(f)
        
    img_h, img_w = ann["size"]["height"], ann["size"]["width"]
    full_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    
    for obj in ann.get("objects", []):
        if obj["geometryType"] == "bitmap":
            origin = obj["bitmap"]["origin"] # [x, y]
            b64_data = obj["bitmap"]["data"]
            
            patch = base64_2_mask(b64_data)
            ph, pw = patch.shape
            
            x, y = origin[0], origin[1]
            full_mask[y:y+ph, x:x+pw] = np.maximum(full_mask[y:y+ph, x:x+pw], patch)
            
    # Bronze outputs
    out_img = os.path.join(bronze_dir, "images", item_id)
    out_mask = os.path.join(bronze_dir, "masks", item_id)
    
    shutil.copy2(img_path, out_img)
    cv2.imwrite(out_mask, full_mask)

print("Bronze data generated. Now preparing Silver Pseudo-Video...")

NUM_FRAMES = 5
manifest = []

images_bronze = os.listdir(os.path.join(bronze_dir, "images"))

# Process just 50 to match Kvasir's sample size and keep it fast
images_bronze = images_bronze[:50] 

for idx, img_file in enumerate(images_bronze):
    case_id = f"cvc_{os.path.splitext(img_file)[0]}"
    
    img_path = os.path.join(bronze_dir, "images", img_file)
    mask_path = os.path.join(bronze_dir, "masks", img_file)
    
    silver_case_path = os.path.join(silver_base, "clips", case_id)
    os.makedirs(silver_case_path, exist_ok=True)
    
    base_img = Image.open(img_path).convert("RGB")
    base_mask = Image.open(mask_path).convert("L")
    
    for i in range(1, NUM_FRAMES + 1):
        frame_name = f"frame_{i:04d}.png"
        mask_name = f"mask_{i:04d}.png"
        
        factor = random.uniform(0.95, 1.05)
        enhancer = ImageEnhance.Brightness(base_img)
        jittered_img = enhancer.enhance(factor)
        
        jittered_img.save(os.path.join(silver_case_path, frame_name))
        base_mask.save(os.path.join(silver_case_path, mask_name))
        
    manifest.append({
        "case_id": case_id,
        "num_frames": NUM_FRAMES,
        "pathology_class": "unknown",
        "split": "train"
    })

manifest_path = os.path.join(silver_base, "manifest.json")
with open(manifest_path, 'w') as f:
    json.dump(manifest, f, indent=2)

print(f"✅ Generated Pseudo-Video Silver dataset at {silver_base} with {len(images_bronze)} cases.")
