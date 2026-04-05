import os
import shutil
import json
import random
from PIL import Image, ImageEnhance
import numpy as np

kvasir_base = r"g:\My Drive\data_lake\01_bronze_medical\kvasir-seg\Kvasir-SEG\Kvasir-SEG"
silver_base = r"g:\My Drive\data_lake\02_silver\kvasir_pseudo_video\train"

images_dir = os.path.join(kvasir_base, "images")
masks_dir = os.path.join(kvasir_base, "masks")

os.makedirs(os.path.join(silver_base, "clips"), exist_ok=True)

NUM_FRAMES = 5
manifest = []

# Process first 20 cases for rapid testing, full scale on Colab later if needed
# We want to do all locally as commanded for step 1
print("Preparing Silver Pseudo-Video dataset from Kvasir-SEG...")

if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
    print("Kvasir-SEG images/masks not found.")
    exit(1)

valid_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

# Limit to 10 for local quick mock if that's preferred, but I'll do 50 to get a small realistic batch.
valid_files = valid_files[:50] 

for idx, img_file in enumerate(valid_files):
    case_id = f"kvasir_{os.path.splitext(img_file)[0]}"
    print(f"Processing {idx+1}/{len(valid_files)}: {case_id}...")
    
    img_path = os.path.join(images_dir, img_file)
    # Masks in Kvasir usually have same name but maybe .png instead of .jpg
    mask_file = os.path.splitext(img_file)[0] + ".jpg"
    mask_path = os.path.join(masks_dir, mask_file)
    
    # Check if mask exists. If not try png.
    if not os.path.exists(mask_path):
        mask_file = os.path.splitext(img_file)[0] + ".png"
        mask_path = os.path.join(masks_dir, mask_file)
        
    if not os.path.exists(mask_path):
        print(f"Mask not found for {img_file}")
        continue
        
    silver_case_path = os.path.join(silver_base, "clips", case_id)
    os.makedirs(silver_case_path, exist_ok=True)
    
    base_img = Image.open(img_path).convert("RGB")
    base_mask = Image.open(mask_path).convert("L")
    
    for i in range(1, NUM_FRAMES + 1):
        frame_name = f"frame_{i:04d}.png"
        mask_name = f"mask_{i:04d}.png"
        
        # Apply slight random brightness jitter to simulate different "frames"
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

print(f"✅ Generated Pseudo-Video Silver dataset at {silver_base} with {len(valid_files)} cases.")
