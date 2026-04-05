import os
import shutil
import json
import numpy as np
from PIL import Image

# Use one specific case to avoid running forever locally
target_cases = ['case15_1']

landing_base = r"g:\My Drive\data_lake\00_landing\polyp\sun-seg\SUN-SEG-Annotation\TrainDataset"
silver_base = r"g:\My Drive\data_lake\02_silver\sun_seg_vps\train"

os.makedirs(os.path.join(silver_base, "clips"), exist_ok=True)

manifest = []

print("Preparing Silver mock dataset for Video Polyp Segmentation...")

gt_base = os.path.join(landing_base, "GT")
if not os.path.exists(gt_base):
    print("GT folder not found.")
    exit(1)

for case_id in target_cases:
    print(f"Processing {case_id}...")
    case_path = os.path.join(gt_base, case_id)
    if not os.path.exists(case_path):
        continue
        
    silver_case_path = os.path.join(silver_base, "clips", case_id)
    os.makedirs(silver_case_path, exist_ok=True)
    
    masks = sorted(os.listdir(case_path))
    num_frames = 0
    
    for i, mask_file in enumerate(masks):
        if not mask_file.endswith('.png'):
            continue
            
        mask_src = os.path.join(case_path, mask_file)
        
        # Consistent sequential naming
        frame_name = f"frame_{i+1:04d}.png"
        mask_name = f"mask_{i+1:04d}.png"
        
        silver_mask = os.path.join(silver_case_path, mask_name)
        silver_frame = os.path.join(silver_case_path, frame_name)
        
        # Copy GT mask
        shutil.copy2(mask_src, silver_mask)
        
        # Create a mock video frame (uniform noise mimicking endoscopic texture loosely)
        img_array = np.random.randint(50, 150, (352, 352, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(silver_frame)
        
        num_frames += 1
        
    manifest.append({
        "case_id": case_id,
        "num_frames": num_frames,
        "pathology_class": "mock",
        "split": "train"
    })

manifest_path = os.path.join(silver_base, "manifest.json")
with open(manifest_path, 'w') as f:
    json.dump(manifest, f, indent=2)

print(f"✅ Generated dummy Silver dataset at {silver_base} with 1 case and {num_frames} frames.")
