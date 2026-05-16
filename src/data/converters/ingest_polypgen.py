import logging

logger = logging.getLogger(__name__)
import json, os, glob, argparse
import xml.etree.ElementTree as ET

def convert(root, out):
    imgs, anns = [], []
    aid = iid = 1
    for cd in sorted(glob.glob(os.path.join(root, 'data_C*'))):
        cn = os.path.basename(cd).replace('data_','')
        bd = os.path.join(cd, 'bbox_' + cn)
        if not os.path.exists(bd): continue
        for xp in sorted(glob.glob(os.path.join(bd, '*.xml'))):
            s = os.path.splitext(os.path.basename(xp))[0]
            tree = ET.parse(xp)
            r = tree.getroot()
            sz = r.find('size')
            if sz is None: continue
            w = int(sz.find('width').text)
            h = int(sz.find('height').text)
            imgs.append({'id':iid,'file_name':cn+'/'+s+'.jpg','width':w,'height':h,
                'metadata':{'source_dataset':'polypgen','center_id':cn}})
            for obj in r.findall('object'):
                bb = obj.find('bndbox')
                x1=float(bb.find('xmin').text)
                y1=float(bb.find('ymin').text)
                x2=float(bb.find('xmax').text)
                y2=float(bb.find('ymax').text)
                anns.append({'id':aid,'image_id':iid,'category_id':1,
                    'bbox':[x1,y1,x2-x1,y2-y1],'segmentation':[],'area':(x2-x1)*(y2-y1),
                    'iscrowd':0,'attributes':{'is_ai_generated':False,'review_status':'unreviewed'}})
                aid += 1
            iid += 1
    coco = {'info':{'description':'PolypGen COCO','version':'1.0.0'},
        'categories':[{'id':1,'name':'polyp','supercategory':'lesion'}],
        'images':imgs,'annotations':anns}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out,'w') as f: json.dump(coco,f)
    logger.info(f'Wrote {len(imgs)} images, {len(anns)} annotations')

if __name__=='__main__':
    import sys
    from pathlib import Path
    
    # Try importing src.config to get standard paths and downloader
    try:
        from src.config.catalog import download_dataset
        from src.config.paths import get_bronze_path, SILVER
        
        logger.info("📥 Downloading PolypGen via src.config...")
        success = download_dataset("polypgen")
        if success is False:
            logger.error("❌ Download failed. Make sure ~/.kaggle/kaggle.json exists.")
            sys.exit(1)
            
        root = str(get_bronze_path('medical') / "polypgen")
        out = str(SILVER / "polypgen_coco" / "annotations.json")
        
    except ImportError:
        # Fallback to local parsing if src.config is missing
        p = argparse.ArgumentParser()
        p.add_argument('--root', required=True)
        p.add_argument('--out', required=True)
        a = p.parse_args()
        root, out = a.root, a.out

    logger.info(f"🔄 Converting {root} to COCO format at {out}...")
    convert(root, out)
