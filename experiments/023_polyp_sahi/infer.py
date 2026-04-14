import os, glob
import torch
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from pathlib import Path

def run_sahi():
    p = r'/content/drive/MyDrive/models/trained/022_polyp_yolov12x/train/weights/best.pt'
    if not os.path.exists(p):
        p = r'/content/drive/MyDrive/models/trained/021_polyp_rfdetr/train/weights/best.pt'
    
    assert os.path.exists(p), f"Could not find model weights at {p}"
    print(f"Initializing SAHI with {p}")
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=p,
        confidence_threshold=0.3,
        device=device
    )
    
    test_img_paths = glob.glob('/content/drive/MyDrive/data_lake/03_gold/016_polyp_fast_diag_dataset/val/images/*.*')
    valid_exts = {'.jpg', '.jpeg', '.png'}
    test_img_paths = [img for img in test_img_paths if Path(img).suffix.lower() in valid_exts]
    
    if not test_img_paths:
        print("No test images found.")
        return

    out_dir = Path('/content/drive/MyDrive/models/trained/023_polyp_sahi')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Running SAHI inference on {len(test_img_paths)} images...")
    for i, test_img in enumerate(test_img_paths):
        result = get_sliced_prediction(
            test_img,
            model,
            slice_height=400,
            slice_width=400,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2
        )
        file_name = Path(test_img).stem
        result.export_visuals(export_dir=str(out_dir), file_name=file_name)
        
    print("SAHI inference complete! Outputs saved to drive.")

if __name__ == '__main__':
    run_sahi()
