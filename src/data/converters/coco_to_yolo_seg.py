import json, os, argparse
import numpy as np

def convert(json_path, output_dir):
    with open(json_path, 'r') as f:
        data = json.load(f)
    os.makedirs(output_dir, exist_ok=True)
    img_id_map = {img['id']: img for img in data['images']}
    for ann in data['annotations']:
        img = img_id_map[ann['image_id']]
        w, h = img['width'], img['height']
        txt_path = os.path.join(output_dir, os.path.splitext(img['file_name'])[0] + '.txt')
        cat_id = ann['category_id'] - 1
        if 'segmentation' in ann and ann['segmentation']:
            for seg in ann['segmentation']:
                poly = np.array(seg).reshape(-1, 2)
                poly[:, 0] /= w
                poly[:, 1] /= h
                line = f"{cat_id} " + " ".join([f"{x: .6f} {y: .6f}" for x, y in poly])
                with open(txt_path, 'a') as f:
                    f.write(line + "\n")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    convert(args.json, args.out)
