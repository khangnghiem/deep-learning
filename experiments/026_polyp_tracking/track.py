import os
import yaml
import glob
from ultralytics import YOLO

def track_videos(cfg_path='config.yaml'):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
        
    model = YOLO(cfg['model']['weights'])
    
    # In LDPolypVideo gold dataset, we have images named <video_name>_frame<id>.jpg
    # BUT wait, YOLO can track directories of images directly if they are sorted!
    # A better approach to see temporal smoothing is to run on the actual test videos in silver layer!
    # Bronze layer: /content/drive/MyDrive/data_lake/01_bronze/LDPolypVideo/Test.rar (extracted to silver)
    # Let's assume user wants to track original videos or the test frames. We can use the test frames directory.
    
    # We will just track the images in the test folder directly.
    # We can use predict() or track() over the directory.
    print(f"Tracking using {cfg['model']['tracker']}...")
    val_res = model.track(
        source=cfg['data']['test_videos_dir'],
        tracker=cfg['model']['tracker'],
        conf=cfg['tracking']['conf'],
        iou=cfg['tracking']['iou'],
        save=cfg['tracking']['save'],
        save_txt=cfg['tracking']['save_txt'],
        project=cfg['data']['output_dir'],
        name='test_tracking_results'
    )
    print("Tracking completed.")

if __name__ == '__main__':
    track_videos('config.yaml')
