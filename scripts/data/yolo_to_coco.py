import json, os, glob, argparse
from PIL import Image
import numpy as np

def convert(label_dir, image_dir, output_json, dataset_name="polyp_yolo"):
    images, anns = [], []
    ann_id = 1
    for img_id, lp in enumerate(sorted(glob.glob(os.path.join(label_dir, "*.txt"))), 1):
        stem = os.path.splitext(os.path.basename(lp))[0]
        ip = None
        for ext in ['.jpg','.png','.jpeg']:
            c = os.path.join(image_dir, stem+ext)
            if os.path.exists(c): ip = c; break
        if not ip: continue
        img = Image.open(ip); w, h = img.size
        images.append({"id": img_id, "file_name": os.path.basename(ip), "width": w, "height": h,
            "metadata": {"source_dataset": dataset_name, "modality": "unknown"}})
        with open(lp) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5: continue
                cat = int(parts[0])+1; coords = [float(x) for x in parts[1:]]
                if len(coords) == 4:
                    cx,cy,bw,bh = coords
                    x=(cx-bw/2)*w; y=(cy-bh/2)*h; bw*=w; bh*=h
                    anns.append({"id":ann_id,"image_id":img_id,"category_id":cat,
                        "bbox":[round(x,1),round(y,1),round(bw,1),round(bh,1)],
                        "segmentation":[],"area":round(bw*bh,1),"iscrowd":0,
                        "attributes":{"is_ai_generated":False,"review_status":"unreviewed"}})
                else:
                    poly = []
                    for i in range(0,len(coords),2): poly.extend([coords[i]*w, coords[i+1]*h])
                    pts = np.array(poly).reshape(-1,2)
                    x1,y1 = pts.min(0); x2,y2 = pts.max(0)
                    anns.append({"id":ann_id,"image_id":img_id,"category_id":cat,
                        "bbox":[round(float(x1),1),round(float(y1),1),round(float(x2-x1),1),round(float(y2-y1),1)],
                        "segmentation":[[round(v,1) for v in poly]],"area":round(float((x2-x1)*(y2-y1)),1),
                        "iscrowd":0,"attributes":{"is_ai_generated":False,"review_status":"unreviewed"}})
                ann_id += 1
    coco = {"info":{"description":"YOLO to COCO","version":"1.0.0","schema_version":"1.0.0"},
        "licenses":[],"categories":[{"id":1,"name":"polyp","supercategory":"lesion"}],
        "images":images,"annotations":anns}
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json,'w') as f: json.dump(coco,f)
    print(f"Wrote {len(images)} images, {len(anns)} annotations")

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--label_dir",required=True); p.add_argument("--image_dir",required=True)
    p.add_argument("--out",required=True); p.add_argument("--dataset",default="polyp_yolo")
    a = p.parse_args(); convert(a.label_dir,a.image_dir,a.out,a.dataset)
