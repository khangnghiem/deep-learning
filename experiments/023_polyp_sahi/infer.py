import os, glob
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from pathlib import Path

def run_sahi():
    # Attempt to load best model from 021 or 022
    p = r'/content/drive/MyDrive/models/trained/022_polyp_yolov12x/train/weights/best.pt'
    if not os.path.exists(p):
        p = r'/content/drive/MyDrive/models/trained/021_polyp_rfdetr/train/weights/best.pt'

    print(f"Initializing SAHI with {p}")
    model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=p,
        confidence_threshold=0.3,
        device="cuda:0"
    )

    test_img_path = glob.glob('/content/drive/MyDrive/data_lake/03_gold/016_polyp_fast_diag_dataset/val/images/*.jpg')
    if not test_img_path:
        print("No test images found.")
        return

    test_img = test_img_path[0]
    result = get_sliced_prediction(
        test_img,
        model,
        slice_height=400,
        slice_width=400,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2
    )

    out_dir = Path('/content/drive/MyDrive/models/trained/023_polyp_sahi')
    out_dir.mkdir(parents=True, exist_ok=True)
    result.export_visuals(export_dir=str(out_dir), file_name="sahi_eval")
    print("SAHI inference complete! Output saved to drive.")

if __name__ == '__main__':
    run_sahi()
