import json, os, glob, cv2, argparse
import numpy as np

def mask_to_polygons(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if len(contour) >= 3:
            polygons.append(contour.flatten().tolist())
    return polygons

def convert(mask_dir, output_json, dataset_name):
    images, annotations = [], []
    ann_id = 1
    files = sorted(glob.glob(os.path.join(mask_dir, "*.*")))
    for i, path in enumerate(files):
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None: continue
        h, w = mask.shape
        img_id = i + 1
        images.append({"id": img_id, "file_name": os.path.basename(path), "width": w, "height": h, "metadata": {"source_dataset": dataset_name}})
        for poly in mask_to_polygons(mask):
            area = float(cv2.contourArea(np.array(poly).reshape(-1, 2).astype(np.float32)))
            x, y, wb, hb = cv2.boundingRect(np.array(poly).reshape(-1, 2).astype(np.float32))
            annotations.append({"id": ann_id, "image_id": img_id, "category_id": 1, "bbox": [float(x), float(y), float(wb), float(hb)], "segmentation": [poly], "area": area, "iscrowd": 0, "attributes": {"is_ai_generated": False}})
            ann_id += 1
    with open(output_json, 'w') as f:
        json.dump({"images": images, "annotations": annotations, "categories": [{"id": 1, "name": "polyp"}]}, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    convert(args.mask_dir, args.out, args.dataset)
