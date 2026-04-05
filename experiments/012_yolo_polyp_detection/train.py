import argparse
import sys
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPOS_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(REPOS_ROOT / "shared_config"))

from shared_config.paths import TRAINED

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=TRAINED / "012_yolo_polyp_detection")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    yaml_path = args.dataset / "dataset.yaml"
    model = YOLO("yolo11n-seg.pt")

    results = model.train(
        data=str(yaml_path.absolute()),
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=640,
        project=str(args.output),
        name="run",
        exist_ok=True,
    )

    trained_pt = args.output / "run" / "weights" / "best.pt"
    best_model = YOLO(str(trained_pt))
    best_model.export(format="onnx", imgsz=640, dynamic=True)

if __name__ == "__main__":
    main()
