"""
Comprehensive Deep Learning Dataset Catalog
===========================================

100+ public datasets organized by category with ingestion support.

Usage:
    python ingest_catalog.py --list              # List all datasets
    python ingest_catalog.py --list vision       # List vision datasets
    python ingest_catalog.py cifar10             # Download specific dataset
    python ingest_catalog.py --all-small         # Download all <500MB datasets
"""

import sys
import os
import zipfile
import tarfile
import urllib.request
import time
import functools
from pathlib import Path
from src.data.kaggle import _validate_kaggle_id

from src.config.paths import LANDING, get_bronze_path


# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds, doubles each retry


def with_retry(max_retries=MAX_RETRIES, delay=RETRY_DELAY):
    """Decorator to retry a function on failure with exponential backoff."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        wait_time = delay * (2 ** attempt)
                        print(f"  ⚠️  Attempt {attempt + 1}/{max_retries} failed: {str(e)[:80]}")
                        print(f"     Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        print(f"  ❌ All {max_retries} attempts failed: {str(e)[:100]}")
            return None  # All retries failed
        return wrapper
    return decorator

# =============================================================================
# DATASET CATALOG - 100+ Datasets
# =============================================================================

DATASETS = {
    # =========================================================================
    # VISION - Image Classification (30+ datasets)
    # =========================================================================
    
    # Small (<100MB)
    "mnist": {"source": "torchvision", "size": "50MB", "category": "vision", "classes": 10},
    "fashion-mnist": {"source": "torchvision", "size": "50MB", "category": "vision", "classes": 10},
    "emnist": {"source": "torchvision", "size": "500MB", "category": "vision", "classes": 62},
    "kmnist": {"source": "torchvision", "size": "50MB", "category": "vision", "classes": 10},
    "qmnist": {"source": "torchvision", "size": "100MB", "category": "vision", "classes": 10},
    
    # Medium (100MB-1GB)
    "cifar10": {"source": "torchvision", "size": "170MB", "category": "vision", "classes": 10},
    "cifar100": {"source": "torchvision", "size": "170MB", "category": "vision", "classes": 100},
    "svhn": {"source": "torchvision", "size": "500MB", "category": "vision", "classes": 10},
    "stl10": {"source": "torchvision", "size": "700MB", "category": "vision", "classes": 10},
    "caltech101": {"source": "url", "url": "https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip", "size": "130MB", "category": "vision", "classes": 101},
    "caltech256": {"source": "url", "url": "https://data.caltech.edu/records/nyy15-4j048/files/256_ObjectCategories.tar", "size": "1.2GB", "category": "vision", "classes": 256},
    
    # Kaggle Vision
    "intel-image": {"source": "kaggle", "kaggle_id": "puneet6060/intel-image-classification", "size": "350MB", "category": "vision", "classes": 6},
    "flowers": {"source": "kaggle", "kaggle_id": "alxmamaev/flowers-recognition", "size": "200MB", "category": "vision", "classes": 5},
    "dogs-vs-cats": {"source": "kaggle", "kaggle_id": "competitions/dogs-vs-cats", "size": "800MB", "category": "vision", "classes": 2},
    "plant-seedlings": {"source": "kaggle", "kaggle_id": "competitions/plant-seedlings-classification", "size": "500MB", "category": "vision", "classes": 12},
    "stanford-dogs": {"source": "kaggle", "kaggle_id": "jessicali9530/stanford-dogs-dataset", "size": "800MB", "category": "vision", "classes": 120},
    "fruits360": {"source": "kaggle", "kaggle_id": "moltean/fruits", "size": "600MB", "category": "vision", "classes": 131},
    "birds400": {"source": "kaggle", "kaggle_id": "gpiosenka/100-bird-species", "size": "1.5GB", "category": "vision", "classes": 400},
    "butterflies": {"source": "kaggle", "kaggle_id": "gpiosenka/butterfly-images40-species", "size": "200MB", "category": "vision", "classes": 40},
    "food101": {"source": "kaggle", "kaggle_id": "dansbecker/food-101", "size": "5GB", "category": "vision", "classes": 101},
    "cars196": {"source": "kaggle", "kaggle_id": "jutrera/stanford-car-dataset-by-classes-folder", "size": "2GB", "category": "vision", "classes": 196},
    "aircraft": {"source": "kaggle", "kaggle_id": "seryouxblaster764/fgvc-aircraft", "size": "500MB", "category": "vision", "classes": 100},
    "eurosat": {"source": "kaggle", "kaggle_id": "apollo2506/eurosat-dataset", "size": "90MB", "category": "vision", "classes": 10},
    "garbage": {"source": "kaggle", "kaggle_id": "asdasdasasdas/garbage-classification", "size": "250MB", "category": "vision", "classes": 6},
    "weather": {"source": "kaggle", "kaggle_id": "jehanbhathena/weather-dataset", "size": "100MB", "category": "vision", "classes": 4},
    "landscape": {"source": "kaggle", "kaggle_id": "utkarshsaxenadn/landscape-recognition-image-dataset-12k-images", "size": "300MB", "category": "vision", "classes": 5},
    "sports": {"source": "kaggle", "kaggle_id": "gpiosenka/sports-classification", "size": "1GB", "category": "vision", "classes": 100},
    "animals10": {"source": "kaggle", "kaggle_id": "alessiocorrado99/animals10", "size": "400MB", "category": "vision", "classes": 10},
    "simpsons": {"source": "kaggle", "kaggle_id": "alexattia/the-simpsons-characters-dataset", "size": "1GB", "category": "vision", "classes": 42},
    
    # HuggingFace Vision
    "beans": {"source": "huggingface", "hf_id": "beans", "size": "200MB", "category": "vision", "classes": 3},
    "oxford-flowers": {"source": "huggingface", "hf_id": "nelorth/oxford-flowers", "size": "500MB", "category": "vision", "classes": 102},
    "cats-vs-dogs": {"source": "huggingface", "hf_id": "cats_vs_dogs", "size": "800MB", "category": "vision", "classes": 2},
    
    # =========================================================================
    # VISION - Object Detection (15+ datasets)
    # =========================================================================
    "coco": {"source": "url", "url": "http://images.cocodataset.org/zips/train2017.zip", "size": "18GB", "category": "detection"},
    "voc2007": {"source": "huggingface", "hf_id": "detection-datasets/coco", "size": "500MB", "category": "detection"},
    "voc2012": {"source": "huggingface", "hf_id": "detection-datasets/coco", "size": "2GB", "category": "detection"},
    "global-wheat": {"source": "kaggle", "kaggle_id": "competitions/global-wheat-detection", "size": "3GB", "category": "detection"},
    "open-images": {"source": "kaggle", "kaggle_id": "competitions/open-images-2019-object-detection", "size": "10GB", "category": "detection"},
    "vindr-cxr": {"source": "kaggle", "kaggle_id": "competitions/vinbigdata-chest-xray-abnormalities-detection", "size": "5GB", "category": "detection"},
    "sartorius": {"source": "kaggle", "kaggle_id": "competitions/sartorius-cell-instance-segmentation", "size": "2GB", "category": "detection"},
    "uw-madison": {"source": "kaggle", "kaggle_id": "competitions/uw-madison-gi-tract-image-segmentation", "size": "4GB", "category": "detection"},
    "hubmap": {"source": "kaggle", "kaggle_id": "competitions/hubmap-organ-segmentation", "size": "3GB", "category": "detection"},
    "rsna-breast": {"source": "kaggle", "kaggle_id": "competitions/rsna-breast-cancer-detection", "size": "8GB", "category": "detection"},
    "lvis": {"source": "huggingface", "hf_id": "lvis/lvis", "size": "15GB", "category": "detection"},
    
    # =========================================================================
    # MEDICAL IMAGING (40+ datasets) - Organized by Modality
    # =========================================================================
    
    # --- ULTRASOUND (30+ datasets) ---
    # HuggingFace datasets (no rule acceptance required)
    "hf-breast-ultrasound": {"source": "huggingface", "hf_id": "alkzar90/NIH-Chest-X-ray-dataset", "size": "200MB", "category": "medical", "classes": 3, "modality": "ultrasound"},
    "hf-busi": {"source": "huggingface", "hf_id": "gymprathap/Breast-Cancer-Ultrasound-Images-Dataset", "size": "150MB", "category": "medical", "classes": 3, "modality": "ultrasound"},
    "hf-busi-nnunet": {"source": "huggingface", "hf_id": "veyselozdemir/nnUNet-Breast-Cancer-Ultrasound", "size": "200MB", "category": "medical", "classes": 3, "modality": "ultrasound"},
    "hf-busi-shivam": {"source": "huggingface", "hf_id": "ShivamRaisharma/breastcancer", "size": "150MB", "category": "medical", "classes": 3, "modality": "ultrasound"},
    
    # Zenodo datasets (open access, no acceptance required)
    "zenodo-fetal-planes": {"source": "url", "url": "https://zenodo.org/records/3904280/files/FETAL_PLANES_ZENODO.zip?download=1", "size": "2.1GB", "category": "medical", "classes": 6, "modality": "ultrasound"},
    # Note: TNSCUI requires registration at grand-challenge.org - keeping as reference
    # "zenodo-thyroid-nodules": {"source": "url", "url": "https://tn-scui2020.grand-challenge.org/", "size": "500MB", "category": "medical", "classes": 2, "modality": "ultrasound"},
    
    # Zenodo datasets (open access, high quality - NEW Feb 2026)
    "zenodo-bus-bra": {"source": "url", "url": "https://zenodo.org/records/8231412/files/BUS-BRA.zip?download=1", "size": "1.5GB", "category": "medical", "classes": 2, "modality": "ultrasound"},
    "zenodo-focus-fetal-cardiac": {"source": "url", "url": "https://zenodo.org/records/14597550/files/FOCUS.zip?download=1", "size": "100MB", "category": "medical", "modality": "ultrasound"},
    
    # GitHub datasets (direct download)
    "github-breast-kidney-us": {"source": "url", "url": "https://github.com/CGPxy/Ultrasound-Dataset/archive/refs/heads/main.zip", "size": "100MB", "category": "medical", "modality": "ultrasound"},
    # Note: BUS-Synthetic repo appears to be removed, commenting out
    # "github-bus-synthetic": {"source": "url", "url": "https://github.com/udiat-group/BUS-Synthetic/archive/refs/heads/main.zip", "size": "200MB", "category": "medical", "classes": 2, "modality": "ultrasound"},
    
    # Kaggle datasets (verified to work without rule acceptance)
    "polypgen": {"source": "kaggle", "kaggle_id": "aliabozorgy/polypgen2021-segmentation-2", "size": "3GB", "category": "medical", "classes": 2, "modality": "endoscopy"},
    
    # --- ENDOSCOPY / POLYP DATASETS (Fast-Diag) ---
    "kvasir-seg": {"source": "kaggle", "kaggle_id": "debeshjha1/kvasirseg", "size": "44MB", "category": "medical", "classes": 2, "modality": "endoscopy"},
    "cvc-clinicdb": {"source": "kaggle", "kaggle_id": "balraj98/cvcclinicdb", "size": "50MB", "category": "medical", "classes": 2, "modality": "endoscopy"},
    "cvc-colondb": {"source": "kaggle", "kaggle_id": "hopmai/cvc-colondb", "size": "30MB", "category": "medical", "classes": 2, "modality": "endoscopy"},
    "bkai-igh": {"source": "kaggle", "kaggle_id": "phamsohanh/bkai-igh-neopolypsmall", "size": "100MB", "category": "medical", "classes": 3, "modality": "endoscopy"},
    "etis-larib": {"source": "kaggle", "kaggle_id": "nguyenvoquocduong/etis-laribpolypdb", "size": "50MB", "category": "medical", "classes": 2, "modality": "endoscopy"},
    "polypdb": {"source": "url", "url": "https://osf.io/pr7ms/download", "size": "500MB", "category": "medical", "classes": 2, "modality": "endoscopy"},
    "ldpolypvideo": {"source": "url", "url": "https://drive.google.com/drive/folders/13KwU_uZcxsl6dL-mqcs39Yb0gjU9vn3G", "size": "5GB", "category": "medical", "modality": "endoscopy"},
    
    "breast-ultrasound": {"source": "kaggle", "kaggle_id": "aryashah2k/breast-ultrasound-images-dataset", "size": "200MB", "category": "medical", "classes": 3, "modality": "ultrasound"},
    "bus-uclm-breast": {"source": "kaggle", "kaggle_id": "sabahesaraki/breast-ultrasound-images-dataset", "size": "150MB", "category": "medical", "modality": "ultrasound"},
    
    # Fetal/Obstetric
    "fetal-ultrasound": {"source": "kaggle", "kaggle_id": "andrewmvd/fetal-head-ultrasound", "size": "150MB", "category": "medical", "modality": "ultrasound"},
    "fetal-planes": {"source": "kaggle", "kaggle_id": "drajayshukla/fetus-ultrasound-images", "size": "100MB", "category": "medical", "modality": "ultrasound"},
    
    # Thyroid/Neck (using dataset uploads, not competitions)
    "thyroid-ultrasound": {"source": "kaggle", "kaggle_id": "dasmehdixtr/thyroid-disease-dataset", "size": "100MB", "category": "medical", "modality": "ultrasound"},
    "thyroid-nodules": {"source": "kaggle", "kaggle_id": "homayoonkhadivi/thyroid-nodules-ultrasound-images", "size": "100MB", "category": "medical", "modality": "ultrasound"},
    
    # Echocardiogram/Cardiac (using data uploads, not competitions)
    "echonet-dynamic": {"source": "kaggle", "kaggle_id": "xiaowenlimarketing/echonet-dynamic", "size": "1GB", "category": "medical", "modality": "ultrasound"},
    "camus-echo": {"source": "kaggle", "kaggle_id": "andrewmvd/camus-cardiac-ultrasound", "size": "300MB", "category": "medical", "modality": "ultrasound"},
    "echo-apical": {"source": "kaggle", "kaggle_id": "aysendegerli/echonet-dynamic-videos", "size": "500MB", "category": "medical", "modality": "ultrasound"},
    
    # Liver (verified to work)
    "liver-ultrasound": {"source": "kaggle", "kaggle_id": "kpnchan/annotated-ultrasound-liver-images-dataset", "size": "100MB", "category": "medical", "classes": 3, "modality": "ultrasound"},
    "fatty-liver-us": {"source": "kaggle", "kaggle_id": "wanghaifeng/dataset-of-bmode-fatty-liver-ultrasound-images", "size": "50MB", "category": "medical", "modality": "ultrasound"},
    
    # Kidney (verified to work)
    "kidney-ultrasound": {"source": "kaggle", "kaggle_id": "azmatsiddique/kidney-failure-ultrasound-clinical-images", "size": "100MB", "category": "medical", "modality": "ultrasound"},
    "kidney-stone-us": {"source": "kaggle", "kaggle_id": "nazmul0087/kidney-ultrasound-images-stone-and-no-stone", "size": "200MB", "category": "medical", "classes": 2, "modality": "ultrasound"},
    
    # Lung
    "lung-ultrasound": {"source": "kaggle", "kaggle_id": "patzeck/lung-ultrasound-images", "size": "100MB", "category": "medical", "modality": "ultrasound"},
    "covid-lung-us": {"source": "kaggle", "kaggle_id": "andyczhao/covid19-ultrasound-data", "size": "200MB", "category": "medical", "classes": 3, "modality": "ultrasound"},
    
    # Muscle/Musculoskeletal
    "muscle-ultrasound": {"source": "kaggle", "kaggle_id": "mursalin1199/muscle-ultrasound-images", "size": "100MB", "category": "medical", "modality": "ultrasound"},
    "nerve-ultrasound-v2": {"source": "kaggle", "kaggle_id": "paultimothymooney/brachial-plexus-nerve-segmentation", "size": "200MB", "category": "medical", "modality": "ultrasound"},
    
    # Abdominal/General
    "abdominal-organs-us": {"source": "kaggle", "kaggle_id": "ignaciorlando/ussimandseg", "size": "300MB", "category": "medical", "modality": "ultrasound"},
    "ovarian-ultrasound": {"source": "kaggle", "kaggle_id": "saurabhshahane/ovarian-ultrasound-images", "size": "200MB", "category": "medical", "modality": "ultrasound"},



    # --- X-RAY ---
    "chest-xray": {"source": "kaggle", "kaggle_id": "paultimothymooney/chest-xray-pneumonia", "size": "1.2GB", "category": "medical", "classes": 2, "modality": "xray"},
    "nih-chest-xray": {"source": "kaggle", "kaggle_id": "nih-chest-xrays/data", "size": "42GB", "category": "medical", "classes": 14, "modality": "xray"},
    "pediatric-xray": {"source": "kaggle", "kaggle_id": "andrewmvd/pediatric-pneumonia-chest-xray", "size": "300MB", "category": "medical", "classes": 2, "modality": "xray"},
    "mura-xray": {"source": "kaggle", "kaggle_id": "cjinny/mura-v11", "size": "3GB", "category": "medical", "classes": 2, "modality": "xray"},
    "rsna-pneumonia": {"source": "kaggle", "kaggle_id": "competitions/rsna-pneumonia-detection-challenge", "size": "2GB", "category": "medical", "classes": 2, "modality": "xray"},
    "covid-xray": {"source": "kaggle", "kaggle_id": "prashant268/chest-xray-covid19-pneumonia", "size": "100MB", "category": "medical", "classes": 3, "modality": "xray"},
    "dental-xray": {"source": "kaggle", "kaggle_id": "humansintheloop/teeth-segmentation-on-dental-x-ray-images", "size": "200MB", "category": "medical", "modality": "xray"},
    "bone-fracture": {"source": "kaggle", "kaggle_id": "bmadushanirodrigo/fracture-multi-region-x-ray-data", "size": "200MB", "category": "medical", "classes": 7, "modality": "xray"},
    "tuberculosis-xray": {"source": "kaggle", "kaggle_id": "tawsifurrahman/tuberculosis-tb-chest-xray-dataset", "size": "700MB", "category": "medical", "classes": 2, "modality": "xray"},
    
    # --- CT SCAN ---
    "covid-ct": {"source": "kaggle", "kaggle_id": "plameneduardo/sarscov2-ctscan-dataset", "size": "400MB", "category": "medical", "classes": 2, "modality": "ct"},
    "kidney-ct": {"source": "kaggle", "kaggle_id": "nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone", "size": "500MB", "category": "medical", "classes": 4, "modality": "ct"},
    "liver-ct": {"source": "kaggle", "kaggle_id": "andrewmvd/lits-liver-tumor-segmentation", "size": "500MB", "category": "medical", "modality": "ct"},
    "lung-ct": {"source": "kaggle", "kaggle_id": "mohamedhanyyy/chest-ctscan-images", "size": "200MB", "category": "medical", "classes": 4, "modality": "ct"},
    "head-ct": {"source": "kaggle", "kaggle_id": "felipekitamura/head-ct-hemorrhage", "size": "200MB", "category": "medical", "classes": 2, "modality": "ct"},
    "kits19-kidney-ct": {"source": "kaggle", "kaggle_id": "nazmul0087/kits19-dataset", "size": "30GB", "category": "medical", "modality": "ct"},
    "luna16-lung-ct": {"source": "kaggle", "kaggle_id": "competitions/data-science-bowl-2017", "size": "100GB", "category": "medical", "modality": "ct"},
    
    # --- MRI ---
    "brain-tumor": {"source": "kaggle", "kaggle_id": "sartajbhuvaji/brain-tumor-classification-mri", "size": "100MB", "category": "medical", "classes": 4, "modality": "mri"},
    "alzheimer-mri": {"source": "kaggle", "kaggle_id": "tourist55/alzheimers-dataset-4-class-of-images", "size": "100MB", "category": "medical", "classes": 4, "modality": "mri"},
    "prostate-mri": {"source": "kaggle", "kaggle_id": "competitions/prostate-cancer-grade-assessment", "size": "500MB", "category": "medical", "modality": "mri"},
    "knee-mri": {"source": "kaggle", "kaggle_id": "cburns/mrnet-mri", "size": "200MB", "category": "medical", "classes": 2, "modality": "mri"},
    "cardiac-mri": {"source": "kaggle", "kaggle_id": "competitions/second-annual-data-science-bowl", "size": "1GB", "category": "medical", "modality": "mri"},
    "brats-mri": {"source": "kaggle", "kaggle_id": "awsaf49/brats2020-training-data", "size": "5GB", "category": "medical", "classes": 4, "modality": "mri"},
    "oasis-brain-mri": {"source": "kaggle", "kaggle_id": "jboysen/mri-and-alzheimers", "size": "100MB", "category": "medical", "modality": "mri"},
    "fastmri-brain": {"source": "huggingface", "hf_id": "jldbc/fastmri-brain", "size": "3GB", "category": "medical", "modality": "mri"},
    
    # --- OTHER MEDICAL (Histopathology, Microscopy, etc.) ---
    "skin-cancer": {"source": "kaggle", "kaggle_id": "kmader/skin-cancer-mnist-ham10000", "size": "500MB", "category": "medical", "classes": 7},
    "diabetic-retinopathy": {"source": "kaggle", "kaggle_id": "competitions/diabetic-retinopathy-detection", "size": "80GB", "category": "medical", "classes": 5},
    "malaria": {"source": "kaggle", "kaggle_id": "iarunava/cell-images-for-detecting-malaria", "size": "350MB", "category": "medical", "classes": 2},
    "blood-cells": {"source": "kaggle", "kaggle_id": "paultimothymooney/blood-cells", "size": "350MB", "category": "medical", "classes": 4},
    "eye-disease": {"source": "kaggle", "kaggle_id": "gunavenkatdoddi/eye-diseases-classification", "size": "500MB", "category": "medical", "classes": 4},
    "lung-cancer": {"source": "kaggle", "kaggle_id": "adityamahimkar/iqothnccd-lung-cancer-dataset", "size": "100MB", "category": "medical", "classes": 3},
    "colon-cancer": {"source": "kaggle", "kaggle_id": "andrewmvd/lung-and-colon-cancer-histopathological-images", "size": "1.5GB", "category": "medical", "classes": 5},
    "leukemia": {"source": "kaggle", "kaggle_id": "andrewmvd/leukemia-classification", "size": "100MB", "category": "medical", "classes": 2},
    
    # =========================================================================
    # NLP & TEXT (15+ datasets)
    # =========================================================================
    "imdb": {"source": "huggingface", "hf_id": "imdb", "size": "80MB", "category": "nlp", "classes": 2},
    "squad": {"source": "huggingface", "hf_id": "squad", "size": "35MB", "category": "nlp"},
    "wikitext": {"source": "huggingface", "hf_id": "Salesforce/wikitext", "hf_config": "wikitext-103-v1", "size": "500MB", "category": "nlp"},
    "ag-news": {"source": "huggingface", "hf_id": "ag_news", "size": "30MB", "category": "nlp", "classes": 4},
    "yelp": {"source": "huggingface", "hf_id": "yelp_review_full", "size": "500MB", "category": "nlp", "classes": 5},
    "amazon-reviews": {"source": "huggingface", "hf_id": "amazon_polarity", "size": "2GB", "category": "nlp", "classes": 2},
    "sst2": {"source": "huggingface", "hf_id": "sst2", "size": "10MB", "category": "nlp", "classes": 2},
    "cola": {"source": "huggingface", "hf_id": "glue", "hf_config": "cola", "size": "1MB", "category": "nlp"},
    "mnli": {"source": "huggingface", "hf_id": "glue", "hf_config": "mnli", "size": "300MB", "category": "nlp"},
    "twitter-sentiment": {"source": "kaggle", "kaggle_id": "kazanova/sentiment140", "size": "300MB", "category": "nlp", "classes": 3},
    "fake-news": {"source": "kaggle", "kaggle_id": "jruvika/fake-news-detection", "size": "50MB", "category": "nlp", "classes": 2},
    "spam": {"source": "kaggle", "kaggle_id": "uciml/sms-spam-collection-dataset", "size": "1MB", "category": "nlp", "classes": 2},
    "toxic-comments": {"source": "kaggle", "kaggle_id": "competitions/jigsaw-toxic-comment-classification-challenge", "size": "100MB", "category": "nlp"},
    "quora-pairs": {"source": "kaggle", "kaggle_id": "competitions/quora-question-pairs", "size": "50MB", "category": "nlp"},
    
    # =========================================================================
    # AUDIO (10+ datasets)
    # =========================================================================
    "speech-commands": {"source": "torchvision", "size": "2GB", "category": "audio", "classes": 35},
    "librispeech": {"source": "huggingface", "hf_id": "openslr/librispeech_asr", "hf_config": "clean", "size": "60GB", "category": "audio"},
    "common-voice": {"source": "huggingface", "hf_id": "mozilla-foundation/common_voice_16_1", "hf_config": "en", "auth": True, "size": "70GB", "category": "audio"},
    "audioset": {"source": "url", "url": "https://research.google.com/audioset/", "size": "2TB", "category": "audio"},
    "gtzan": {"source": "kaggle", "kaggle_id": "andradaolteanu/gtzan-dataset-music-genre-classification", "size": "1.2GB", "category": "audio", "classes": 10},
    "urbansound8k": {"source": "kaggle", "kaggle_id": "chrisfilo/urbansound8k", "size": "6GB", "category": "audio", "classes": 10},
    "heartbeat": {"source": "kaggle", "kaggle_id": "kinguistics/heartbeat-sounds", "size": "300MB", "category": "audio", "classes": 5},
    "birdsong": {"source": "kaggle", "kaggle_id": "competitions/birdclef-2023", "size": "10GB", "category": "audio"},
    "emotion-speech": {"source": "kaggle", "kaggle_id": "ejlok1/toronto-emotional-speech-set-tess", "size": "200MB", "category": "audio", "classes": 7},
    
    # =========================================================================
    # TABULAR / STRUCTURED (10+ datasets)
    # =========================================================================
    "titanic": {"source": "kaggle", "kaggle_id": "competitions/titanic", "size": "100KB", "category": "tabular", "classes": 2},
    "house-prices": {"source": "kaggle", "kaggle_id": "competitions/house-prices-advanced-regression-techniques", "size": "1MB", "category": "tabular"},
    "credit-fraud": {"source": "kaggle", "kaggle_id": "mlg-ulb/creditcardfraud", "size": "150MB", "category": "tabular", "classes": 2},
    "walmart-sales": {"source": "kaggle", "kaggle_id": "competitions/walmart-recruiting-store-sales-forecasting", "size": "100MB", "category": "tabular"},
    "rossmann": {"source": "kaggle", "kaggle_id": "competitions/rossmann-store-sales", "size": "50MB", "category": "tabular"},
    "taxi-fare": {"source": "kaggle", "kaggle_id": "competitions/new-york-city-taxi-fare-prediction", "size": "5GB", "category": "tabular"},
    "santander": {"source": "kaggle", "kaggle_id": "competitions/santander-customer-transaction-prediction", "size": "500MB", "category": "tabular"},
    "ieee-fraud": {"source": "kaggle", "kaggle_id": "competitions/ieee-fraud-detection", "size": "1GB", "category": "tabular"},
    
    # =========================================================================
    # VIDEO (5+ datasets)
    # =========================================================================
    "ucf101": {"source": "url", "url": "https://www.crcv.ucf.edu/data/UCF101.php", "size": "7GB", "category": "video", "classes": 101},
    "hmdb51": {"source": "url", "url": "https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/", "size": "2GB", "category": "video", "classes": 51},
    "kinetics400": {"source": "url", "url": "https://github.com/cvdfoundation/kinetics-dataset", "size": "300GB", "category": "video", "classes": 400},
    "youtube8m": {"source": "url", "url": "https://research.google.com/youtube8m/", "size": "1.5TB", "category": "video"},
    "diving48": {"source": "kaggle", "kaggle_id": "benhamner/sf-bay-area-bike-share", "size": "5GB", "category": "video"},
    
    # =========================================================================
    # GENERATIVE / GAN (5+ datasets)
    # =========================================================================
    "celeba": {"source": "torchvision", "size": "1.5GB", "category": "generative"},
    "lsun": {"source": "torchvision", "size": "100GB", "category": "generative"},
    "anime-faces": {"source": "kaggle", "kaggle_id": "splcher/animefacedataset", "size": "500MB", "category": "generative"},
    "art-portraits": {"source": "kaggle", "kaggle_id": "karnikakapoor/art-movements-and-styles", "size": "2GB", "category": "generative"},
    "flickr-faces": {"source": "kaggle", "kaggle_id": "xhlulu/flickrfaceshq-dataset-nvidia-resized-256px", "size": "1GB", "category": "generative"},
    
    # =========================================================================
    # UCI MACHINE LEARNING REPOSITORY (10+ datasets)
    # =========================================================================
    "iris": {"source": "uci", "uci_id": 53, "size": "10KB", "category": "tabular", "classes": 3},
    "wine": {"source": "uci", "uci_id": 109, "size": "50KB", "category": "tabular", "classes": 3},
    "heart-disease": {"source": "uci", "uci_id": 45, "size": "100KB", "category": "tabular", "classes": 2},
    "breast-cancer-uci": {"source": "uci", "uci_id": 17, "size": "50KB", "category": "tabular", "classes": 2},
    "adult": {"source": "uci", "uci_id": 2, "size": "5MB", "category": "tabular", "classes": 2},
    "parkinsons": {"source": "uci", "uci_id": 174, "size": "50KB", "category": "tabular", "classes": 2},
    "abalone": {"source": "uci", "uci_id": 1, "size": "200KB", "category": "tabular"},
    "bank-marketing": {"source": "uci", "uci_id": 222, "size": "4MB", "category": "tabular", "classes": 2},
    "mushroom": {"source": "uci", "uci_id": 73, "size": "500KB", "category": "tabular", "classes": 2},
    "car-evaluation": {"source": "uci", "uci_id": 19, "size": "50KB", "category": "tabular", "classes": 4},
    
    # =========================================================================
    # TENSORFLOW DATASETS (10+ datasets)
    # =========================================================================
    "tf-flowers": {"source": "tfds", "tfds_id": "tf_flowers", "size": "200MB", "category": "vision", "classes": 5},
    "imagenette": {"source": "tfds", "tfds_id": "imagenette", "size": "1.5GB", "category": "vision", "classes": 10},
    "rock-paper-scissors": {"source": "tfds", "tfds_id": "rock_paper_scissors", "size": "100MB", "category": "vision", "classes": 3},
    "horses-or-humans": {"source": "tfds", "tfds_id": "horses_or_humans", "size": "150MB", "category": "vision", "classes": 2},
    "colorectal-histology": {"source": "tfds", "tfds_id": "colorectal_histology", "size": "100MB", "category": "medical", "classes": 8},
    "patch-camelyon": {"source": "tfds", "tfds_id": "patch_camelyon", "size": "6GB", "category": "medical", "classes": 2},
    "stanford-online-products": {"source": "tfds", "tfds_id": "stanford_online_products", "size": "3GB", "category": "vision"},
    "sun397": {"source": "tfds", "tfds_id": "sun397", "size": "36GB", "category": "vision", "classes": 397},
    "dtd": {"source": "tfds", "tfds_id": "dtd", "size": "600MB", "category": "vision", "classes": 47},
    "deep-weeds": {"source": "tfds", "tfds_id": "deep_weeds", "size": "500MB", "category": "vision", "classes": 9},
    
    # =========================================================================
    # OPENML (6+ datasets)
    # =========================================================================
    "openml-credit-g": {"source": "openml", "openml_id": 31, "size": "100KB", "category": "tabular", "classes": 2},
    "openml-electricity": {"source": "openml", "openml_id": 151, "size": "5MB", "category": "tabular", "classes": 2},
    "openml-covertype": {"source": "openml", "openml_id": 1596, "size": "70MB", "category": "tabular", "classes": 7},
    "openml-higgs": {"source": "openml", "openml_id": 23512, "size": "8GB", "category": "tabular", "classes": 2},
    "openml-phoneme": {"source": "openml", "openml_id": 1489, "size": "2MB", "category": "tabular", "classes": 2},
    "openml-spambase": {"source": "openml", "openml_id": 44, "size": "500KB", "category": "tabular", "classes": 2},
    
    # =========================================================================
    # SCIKIT-LEARN BUILT-IN (5 datasets)
    # =========================================================================
    "sklearn-digits": {"source": "sklearn", "sklearn_name": "load_digits", "size": "2MB", "category": "vision", "classes": 10},
    "sklearn-california": {"source": "sklearn", "sklearn_name": "fetch_california_housing", "size": "500KB", "category": "tabular"},
    "sklearn-olivetti": {"source": "sklearn", "sklearn_name": "fetch_olivetti_faces", "size": "2MB", "category": "vision", "classes": 40},
    "sklearn-20newsgroups": {"source": "sklearn", "sklearn_name": "fetch_20newsgroups", "size": "20MB", "category": "nlp", "classes": 20},
    "sklearn-rcv1": {"source": "sklearn", "sklearn_name": "fetch_rcv1", "size": "200MB", "category": "nlp"},
    
    # =========================================================================
    # TIME SERIES (NEW CATEGORY - 10 datasets)
    # =========================================================================
    "m4-hourly": {"source": "kaggle", "kaggle_id": "yogesh94/m4-forecasting-competition", "size": "50MB", "category": "timeseries"},
    "stock-market": {"source": "kaggle", "kaggle_id": "borismarjanovic/price-volume-data-for-all-us-stocks-etfs", "size": "500MB", "category": "timeseries"},
    "energy-consumption": {"source": "kaggle", "kaggle_id": "uciml/electric-power-consumption-data-set", "size": "100MB", "category": "timeseries"},
    "web-traffic": {"source": "kaggle", "kaggle_id": "competitions/web-traffic-time-series-forecasting", "size": "1GB", "category": "timeseries"},
    "air-quality": {"source": "uci", "uci_id": 360, "size": "2MB", "category": "timeseries"},
    "electricity-demand": {"source": "kaggle", "kaggle_id": "aramacus/electricity-demand-dataset", "size": "200MB", "category": "timeseries"},
    "covid-time-series": {"source": "kaggle", "kaggle_id": "sudalairajkumar/novel-corona-virus-2019-dataset", "size": "50MB", "category": "timeseries"},
    "weather-history": {"source": "kaggle", "kaggle_id": "selfishgene/historical-hourly-weather-data", "size": "500MB", "category": "timeseries"},
    "store-sales": {"source": "kaggle", "kaggle_id": "competitions/store-sales-time-series-forecasting", "size": "100MB", "category": "timeseries"},
    "crypto-prices": {"source": "kaggle", "kaggle_id": "jessevent/all-crypto-currencies", "size": "200MB", "category": "timeseries"},
    
    # =========================================================================
    # MORE KAGGLE DATASETS (15 popular)
    # =========================================================================
    "digit-recognizer": {"source": "kaggle", "kaggle_id": "competitions/digit-recognizer", "size": "100MB", "category": "vision", "classes": 10},
    "spaceship-titanic": {"source": "kaggle", "kaggle_id": "competitions/spaceship-titanic", "size": "10MB", "category": "tabular", "classes": 2},
    "otto-products": {"source": "kaggle", "kaggle_id": "competitions/otto-group-product-classification-challenge", "size": "500MB", "category": "tabular", "classes": 9},
    "predict-future-sales": {"source": "kaggle", "kaggle_id": "competitions/competitive-data-science-predict-future-sales", "size": "100MB", "category": "tabular"},
    "natural-images": {"source": "kaggle", "kaggle_id": "prasunroy/natural-images", "size": "150MB", "category": "vision", "classes": 8},
    "rice-images": {"source": "kaggle", "kaggle_id": "muratkokludataset/rice-image-dataset", "size": "200MB", "category": "vision", "classes": 5},
    "fer2013": {"source": "kaggle", "kaggle_id": "competitions/challenges-in-representation-learning-facial-expression-recognition-challenge", "size": "50MB", "category": "vision", "classes": 7},
    "gender-recognition": {"source": "kaggle", "kaggle_id": "cashutosh/gender-classification-dataset", "size": "200MB", "category": "vision", "classes": 2},
    "age-gender": {"source": "kaggle", "kaggle_id": "jangedoo/utkface-new", "size": "500MB", "category": "vision"},
    "sign-language": {"source": "kaggle", "kaggle_id": "datamunge/sign-language-mnist", "size": "100MB", "category": "vision", "classes": 24},
    "chest-ct": {"source": "kaggle", "kaggle_id": "mohamedhanyyy/chest-ctscan-images", "size": "200MB", "category": "medical", "classes": 4},
    "retinal-oct": {"source": "kaggle", "kaggle_id": "paultimothymooney/kermany2018", "size": "1GB", "category": "medical", "classes": 4},
    "melanoma": {"source": "kaggle", "kaggle_id": "competitions/siim-isic-melanoma-classification", "size": "15GB", "category": "medical", "classes": 2},
    "plant-disease": {"source": "kaggle", "kaggle_id": "vipoooool/new-plant-diseases-dataset", "size": "3GB", "category": "vision", "classes": 38},
    "ocean-plastics": {"source": "kaggle", "kaggle_id": "competitions/the-nature-conservancy-fisheries-monitoring", "size": "5GB", "category": "vision", "classes": 8},
    
    # =========================================================================
    # MORE HUGGINGFACE DATASETS (10)
    # =========================================================================
    "emotion": {"source": "huggingface", "hf_id": "emotion", "size": "5MB", "category": "nlp", "classes": 6},
    "rotten-tomatoes": {"source": "huggingface", "hf_id": "rotten_tomatoes", "size": "3MB", "category": "nlp", "classes": 2},
    "tweet-eval": {"source": "huggingface", "hf_id": "tweet_eval", "hf_config": "sentiment", "size": "20MB", "category": "nlp", "classes": 3},
    "xsum": {"source": "huggingface", "hf_id": "xsum", "size": "500MB", "category": "nlp"},
    "billsum": {"source": "huggingface", "hf_id": "billsum", "size": "100MB", "category": "nlp"},
    "conll2003": {"source": "huggingface", "hf_id": "tner/conll2003", "size": "10MB", "category": "nlp"},
    "hf-food101": {"source": "huggingface", "hf_id": "ethz/food101", "size": "5GB", "category": "vision", "classes": 101},
    "imagenet-sketch": {"source": "url", "url": "https://github.com/HaohanWang/ImageNet-Sketch", "size": "2GB", "category": "vision", "classes": 1000},
    "lj-speech": {"source": "huggingface", "hf_id": "keithito/lj_speech", "size": "2GB", "category": "audio"},
    "minds14": {"source": "huggingface", "hf_id": "PolyAI/minds14", "hf_config": "en-US", "size": "500MB", "category": "audio", "classes": 14},
    
    # =========================================================================
    # MORE UCI DATASETS (8)
    # =========================================================================
    "dermatology": {"source": "uci", "uci_id": 33, "size": "50KB", "category": "medical", "classes": 6},
    "glass": {"source": "uci", "uci_id": 42, "size": "20KB", "category": "tabular", "classes": 7},
    "hepatitis": {"source": "uci", "uci_id": 46, "size": "10KB", "category": "medical", "classes": 2},
    "letter-recognition": {"source": "uci", "uci_id": 59, "size": "800KB", "category": "vision", "classes": 26},
    "statlog-segment": {"source": "uci", "uci_id": 147, "size": "400KB", "category": "vision", "classes": 7},
    "optical-digits": {"source": "uci", "uci_id": 80, "size": "2MB", "category": "vision", "classes": 10},
    "pendigits": {"source": "uci", "uci_id": 81, "size": "500KB", "category": "vision", "classes": 10},
    "soybean": {"source": "uci", "uci_id": 90, "size": "100KB", "category": "tabular", "classes": 19},
    
    # =========================================================================
    # DIRECT URL DATASETS (7)
    # =========================================================================
    "tiny-imagenet": {"source": "url", "url": "http://cs231n.stanford.edu/tiny-imagenet-200.zip", "size": "400MB", "category": "vision", "classes": 200},
    "notmnist": {"source": "url", "url": "http://yaroslavvb.com/upload/notMNIST/notMNIST_large.tar.gz", "size": "700MB", "category": "vision", "classes": 10},
    "omniglot": {"source": "url", "url": "https://raw.githubusercontent.com/brendenlake/omniglot/master/python/images_background.zip", "size": "20MB", "category": "vision", "classes": 964},
    "lfw": {"source": "url", "url": "http://vis-www.cs.umass.edu/lfw/lfw.tgz", "size": "200MB", "category": "vision"},
    "fer-plus": {"source": "url", "url": "https://github.com/microsoft/FERPlus/raw/master/data/FER2013Train.zip", "size": "50MB", "category": "vision", "classes": 8},
    "quickdraw": {"source": "url", "url": "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/", "size": "37GB", "category": "vision", "classes": 345},
    "esc50": {"source": "url", "url": "https://github.com/karolpiczak/ESC-50/archive/master.zip", "size": "600MB", "category": "audio", "classes": 50},
    
    # =========================================================================
    # EDUCATION (20 datasets)
    # =========================================================================
    
    # UCI Education Datasets
    "student-performance": {"source": "uci", "uci_id": 320, "size": "1MB", "category": "education"},
    "student-academics": {"source": "uci", "uci_id": 856, "size": "500KB", "category": "education"},
    "higher-education-dropout": {"source": "uci", "uci_id": 697, "size": "2MB", "category": "education"},
    "national-poll-happiness": {"source": "uci", "uci_id": 826, "size": "100KB", "category": "education"},
    "open-university-learning": {"source": "uci", "uci_id": 349, "size": "5MB", "category": "education"},
    
    # Kaggle Education Datasets
    "students-exam-scores": {"source": "kaggle", "kaggle_id": "desalegngeb/students-exam-scores", "size": "5MB", "category": "education"},
    "student-study-performance": {"source": "kaggle", "kaggle_id": "nikhil7280/student-performance-multiple-linear-regression", "size": "1MB", "category": "education"},
    "world-university-rankings": {"source": "kaggle", "kaggle_id": "mylesoneill/world-university-rankings", "size": "5MB", "category": "education"},
    "pisa-test-scores": {"source": "kaggle", "kaggle_id": "econdata/pisa-test-score-mean-performance-on-the-mathematics-scale", "size": "1MB", "category": "education"},
    "us-college-scorecard": {"source": "kaggle", "kaggle_id": "kaggle/college-scorecard", "size": "200MB", "category": "education"},
    "online-education-survey": {"source": "kaggle", "kaggle_id": "nayakroshan/online-education-system-review-during-covid19", "size": "1MB", "category": "education"},
    "coursera-courses": {"source": "kaggle", "kaggle_id": "siddharthm1698/coursera-course-dataset", "size": "5MB", "category": "education"},
    "udemy-courses": {"source": "kaggle", "kaggle_id": "andrewmvd/udemy-courses", "size": "5MB", "category": "education"},
    "edx-courses": {"source": "kaggle", "kaggle_id": "khusheekapoor/edx-courses-dataset-2021", "size": "2MB", "category": "education"},
    "mooc-learner-dropout": {"source": "kaggle", "kaggle_id": "chi0726/mooc-offering-is-knowledge-half-its-battle", "size": "10MB", "category": "education"},
    "literacy-rates-world": {"source": "kaggle", "kaggle_id": "programmerrdai/literacy-rate", "size": "500KB", "category": "education"},
    "student-dropout-prediction": {"source": "kaggle", "kaggle_id": "nphantawee/predict-students-dropout-and-academic-success", "size": "2MB", "category": "education"},
    "learning-styles": {"source": "kaggle", "kaggle_id": "ashydv/students-learning-styles-dataset", "size": "1MB", "category": "education"},
    "school-performance": {"source": "kaggle", "kaggle_id": "aljarah/xAPI-Edu-Data", "size": "500KB", "category": "education"},
    "teachers-evaluation": {"source": "kaggle", "kaggle_id": "sangarshanan/teachers-evaluation", "size": "1MB", "category": "education"},
}

# Total count
TOTAL_DATASETS = len(DATASETS)


def list_datasets(category=None):
    """List all or filtered datasets."""
    print(f"\n{'='*70}")
    print(f"Deep Learning Dataset Catalog - {TOTAL_DATASETS} Datasets")
    print(f"{'='*70}\n")
    
    categories = {}
    for name, info in DATASETS.items():
        cat = info.get("category", "other")
        if cat not in categories:
            categories[cat] = []
        categories[cat].append((name, info))
    
    for cat in sorted(categories.keys()):
        if category and cat != category:
            continue
        datasets = categories[cat]
        print(f"\n## {cat.upper()} ({len(datasets)} datasets)")
        print("-" * 50)
        for name, info in sorted(datasets):
            size = info.get("size", "?")
            source = info.get("source", "?")
            classes = info.get("classes", "")
            cls_str = f" ({classes} classes)" if classes else ""
            print(f"  {name:25} {size:>8}  [{source}]{cls_str}")


def download_dataset(name: str):
    """Download a specific dataset."""
    if name not in DATASETS:
        print(f"Unknown dataset: {name}")
        print("Use --list to see available datasets")
        return False
    
    info = DATASETS[name]
    source = info["source"]
    category = info.get("category", "vision")  # Default to vision if not specified
    bronze_dir = get_bronze_path(category) / name
    landing_dir = LANDING / source / name
    
    # Skip if exists and has content
    if bronze_dir.exists() and any(bronze_dir.iterdir()):
        print(f"⏭️  {name} already exists in {category}, skipping...")
        return "skipped"
    
    # Only create landing dir upfront (bronze created after success)
    landing_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading {name} ({info.get('size', '?')}) via {source}...")
    
    # Track success for proper cleanup
    success = False
    try:
        if source == "torchvision":
            # Torchvision downloads directly to bronze
            bronze_dir.mkdir(parents=True, exist_ok=True)
            success = _download_torchvision(name, info, bronze_dir)
        elif source == "kaggle":
            success = _download_kaggle(name, info, landing_dir, bronze_dir)
        elif source == "huggingface":
            bronze_dir.mkdir(parents=True, exist_ok=True)
            success = _download_huggingface(name, info, bronze_dir)
        elif source == "url":
            success = _download_url(name, info, landing_dir, bronze_dir)
        elif source == "uci":
            bronze_dir.mkdir(parents=True, exist_ok=True)
            success = _download_uci(name, info, bronze_dir)
        elif source == "tfds":
            bronze_dir.mkdir(parents=True, exist_ok=True)
            success = _download_tfds(name, info, bronze_dir)
        elif source == "openml":
            bronze_dir.mkdir(parents=True, exist_ok=True)
            success = _download_openml(name, info, bronze_dir)
        elif source == "sklearn":
            bronze_dir.mkdir(parents=True, exist_ok=True)
            success = _download_sklearn(name, info, bronze_dir)
        else:
            print(f"⚠️  Unknown source: {source}")
            return False
    except Exception as e:
        print(f"❌ Error downloading {name}: {e}")
        success = False
    
    # Cleanup empty directories on failure
    if not success:
        _cleanup_empty_dir(bronze_dir)
        _cleanup_empty_dir(landing_dir)
        print(f"❌ {name} download failed")
        return False
    
    # Verify bronze has content
    if bronze_dir.exists() and any(bronze_dir.iterdir()):
        print(f"✅ {name} ingested to {bronze_dir}")
        # Keep MANIFEST.json in sync
        try:
            from src.config.manifest import update_manifest_entry
            update_manifest_entry(name, category, bronze_dir)
        except Exception as e:
            print(f"⚠️  MANIFEST update skipped: {e}")
        return True
    else:
        _cleanup_empty_dir(bronze_dir)
        print(f"⚠️  {name} completed but no files in bronze")
        return False


def _cleanup_empty_dir(path: Path):
    """Remove directory if empty."""
    try:
        if path.exists() and path.is_dir() and not any(path.iterdir()):
            path.rmdir()
    except OSError:
        pass  # Ignore cleanup errors




@with_retry()
def _download_torchvision(name, info, bronze_dir):
    from torchvision import datasets as tv_datasets
    
    dataset_map = {
        "mnist": tv_datasets.MNIST,
        "fashion-mnist": tv_datasets.FashionMNIST,
        "cifar10": tv_datasets.CIFAR10,
        "cifar100": tv_datasets.CIFAR100,
        "svhn": tv_datasets.SVHN,
        "stl10": tv_datasets.STL10,
        "emnist": tv_datasets.EMNIST,
        "kmnist": tv_datasets.KMNIST,
        "qmnist": tv_datasets.QMNIST,
        "celeba": tv_datasets.CelebA,
        "voc2007": lambda root, **kw: tv_datasets.VOCDetection(root, year='2007', **kw),
        "voc2012": lambda root, **kw: tv_datasets.VOCDetection(root, year='2012', **kw),
    }
    
    if name not in dataset_map:
        print(f"  ⚠️  No torchvision loader for {name}")
        return
    
    try:
        cls = dataset_map[name]
        if name == "svhn":
            cls(str(bronze_dir), split='train', download=True)
            cls(str(bronze_dir), split='test', download=True)
        elif name == "stl10":
            cls(str(bronze_dir), split='train', download=True)
            cls(str(bronze_dir), split='test', download=True)
        elif name == "emnist":
            cls(bronze_dir, split='balanced', train=True, download=True)
        elif name.startswith("voc"):
            # VOC datasets need special handling - server often unavailable
            cls(bronze_dir, download=True)
        elif callable(cls):
            cls(bronze_dir, download=True)
        else:
            cls(bronze_dir, train=True, download=True)
            cls(bronze_dir, train=False, download=True)
        return True
    except RuntimeError as e:
        if "File not found or corrupted" in str(e):
            print(f"  ❌ Download failed: Server unavailable or file corrupted")
            print(f"     This is a known issue with {name}. Try again later or download manually.")
            return False
        else:
            raise


def _download_kaggle(name, info, landing_dir, bronze_dir):
    """Download from Kaggle datasets or competitions."""
    import subprocess
    
    kaggle_id = info["kaggle_id"]
    if kaggle_id.startswith("competitions/"):
        _validate_kaggle_id(kaggle_id.replace("competitions/", ""))
    else:
        _validate_kaggle_id(kaggle_id)
    
    if kaggle_id.startswith("competitions/"):
        comp = kaggle_id.replace("competitions/", "")
        result = subprocess.run(
            ["kaggle", "competitions", "download", "-c", comp, "-p", str(landing_dir)],
            capture_output=True, text=True
        )
        output = result.stdout + result.stderr
        if "401" in output or "403" in output or "Unauthorized" in output or "Forbidden" in output:
            print(f"  ⚠️  Competition requires rule acceptance. Visit:")
            print(f"     https://www.kaggle.com/c/{comp}/rules")
            return False
        elif result.returncode != 0:
            print(f"  ⚠️  Kaggle download failed: {output[:200]}")
            return False
    else:
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", kaggle_id, "-p", str(landing_dir)],
            capture_output=True, text=True
        )
        output = result.stdout + result.stderr
        if "401" in output or "403" in output:
            print(f"  ⚠️  Dataset access denied: {output[:100]}")
            return False
        elif result.returncode != 0:
            print(f"  ⚠️  Kaggle download failed: {output[:200]}")
            return False
        # Show download progress from stdout
        if result.stdout:
            for line in result.stdout.strip().split('\n')[-3:]:
                print(f"  {line}")
    
    # Extract zips
    extracted = False
    bronze_dir.mkdir(parents=True, exist_ok=True)
    for zf in landing_dir.glob("*.zip"):
        with zipfile.ZipFile(zf, 'r') as z:
            z.extractall(bronze_dir)
            extracted = True
    return extracted


def _get_hf_token():
    """Get HuggingFace token from environment, Drive secrets, or cache file."""
    # 1. Check environment variable
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    
    # 2. Check Drive .secrets folder (copied in Colab setup)
    secrets_path = Path.home() / ".secrets" / "huggingface_token"
    if secrets_path.exists():
        return secrets_path.read_text().strip()
    
    # 3. Check standard HF cache (huggingface-cli login)
    cache_path = Path.home() / ".cache" / "huggingface" / "token"
    if cache_path.exists():
        return cache_path.read_text().strip()
    
    return None


@with_retry()
def _download_huggingface(name, info, bronze_dir):
    from datasets import load_dataset, get_dataset_config_names
    
    hf_id = info["hf_id"]
    config = info.get("hf_config")
    requires_auth = info.get("auth", False)
    
    # Get token for authenticated datasets
    token = _get_hf_token() if requires_auth else None
    if requires_auth and not token:
        print(f"⚠️  {name} requires HuggingFace authentication.")
        print("   Set HF_TOKEN env var or copy token to ~/.cache/huggingface/token")
        return False
    
    kwargs = {
        "cache_dir": str(bronze_dir),
        "trust_remote_code": True,
    }
    if token:
        kwargs["token"] = token
    
    try:
        if config:
            load_dataset(hf_id, config, **kwargs)
        else:
            load_dataset(hf_id, **kwargs)
    except ValueError as e:
        # Handle "Config name is missing" error
        if "Config name is missing" in str(e) or "config" in str(e).lower():
            print(f"  ⚠️  {hf_id} requires a config name, trying to find one...")
            try:
                configs = get_dataset_config_names(hf_id)
                if configs:
                    # Try first available config
                    default_config = configs[0]
                    print(f"  📋 Found configs: {configs[:5]}...")
                    print(f"  ➡️  Using config: {default_config}")
                    load_dataset(hf_id, default_config, **kwargs)
                else:
                    raise e
            except Exception as inner_e:
                print(f"  ❌ Could not find valid config: {inner_e}")
                raise e
        else:
            raise e
    return True


@with_retry()
def _download_url(name, info, landing_dir, bronze_dir):
    url = info["url"]
    filename = url.split("/")[-1]
    filepath = landing_dir / filename
    
    if not filepath.exists():
        print(f"  Downloading from {url}...")
        urllib.request.urlretrieve(url, filepath)
    
    # Extract
    bronze_dir.mkdir(parents=True, exist_ok=True)
    if filename.endswith(".zip"):
        with zipfile.ZipFile(filepath, 'r') as z:
            z.extractall(bronze_dir)
    elif filename.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(filepath, 'r:*') as t:
            t.extractall(bronze_dir)
    return True


@with_retry()
def _download_uci(name, info, bronze_dir):
    """Download from UCI ML Repository using official API."""
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError:
        print("  ⚠️  Install ucimlrepo: pip install ucimlrepo")
        return False
    
    import json
    
    dataset = fetch_ucirepo(id=info["uci_id"])
    
    # Save data as CSV
    if hasattr(dataset.data, 'original') and dataset.data.original is not None:
        df = dataset.data.original
    else:
        df = dataset.data.features.copy()
        if dataset.data.targets is not None:
            for col in dataset.data.targets.columns:
                df[col] = dataset.data.targets[col]
    
    df.to_csv(bronze_dir / f"{name}.csv", index=False)
    
    # Save metadata
    meta = {
        "name": dataset.metadata.name if hasattr(dataset.metadata, 'name') else name,
        "uci_id": info["uci_id"],
        "num_instances": dataset.metadata.num_instances if hasattr(dataset.metadata, 'num_instances') else len(df),
    }
    (bronze_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    return True


@with_retry()
def _download_tfds(name, info, bronze_dir):
    """Download from TensorFlow Datasets."""
    try:
        import tensorflow_datasets as tfds
    except ImportError:
        print("  ⚠️  Install tensorflow-datasets: pip install tensorflow-datasets")
        return False
    
    tfds.load(info["tfds_id"], data_dir=str(bronze_dir), download=True)
    return True


@with_retry()
def _download_openml(name, info, bronze_dir):
    """Download from OpenML."""
    try:
        import openml
    except ImportError:
        print("  ⚠️  Install openml: pip install openml")
        return
    
    dataset = openml.datasets.get_dataset(info["openml_id"])
    X, y, _, _ = dataset.get_data(dataset_format='dataframe')
    
    df = X.copy()
    if y is not None:
        df['target'] = y
    
    df.to_csv(bronze_dir / f"{name}.csv", index=False)
    return True


def _download_sklearn(name, info, bronze_dir):
    """Download from scikit-learn."""
    from sklearn import datasets as sk_datasets
    import pandas as pd
    
    loader_name = info["sklearn_name"]
    loader = getattr(sk_datasets, loader_name)
    
    # Handle different dataset types
    if loader_name == "fetch_20newsgroups":
        data = loader(subset='all')
        df = pd.DataFrame({'text': data.data, 'target': data.target})
    elif loader_name == "fetch_rcv1":
        data = loader()
        # Sparse matrix, save in npz format
        import scipy.sparse
        scipy.sparse.save_npz(bronze_dir / f"{name}_X.npz", data.data)
        pd.DataFrame({'target': data.target.toarray().flatten()}).to_csv(
            bronze_dir / f"{name}_y.csv", index=False
        )
        return True
    else:
        data = loader()
        feature_names = getattr(data, 'feature_names', [f'f{i}' for i in range(data.data.shape[1])])
        df = pd.DataFrame(data.data, columns=feature_names)
        if hasattr(data, 'target'):
            df['target'] = data.target
    
    df.to_csv(bronze_dir / f"{name}.csv", index=False)
    return True


def download_small():
    """Download all datasets under 500MB."""
    small = [n for n, i in DATASETS.items() 
             if _parse_size(i.get("size", "0")) < 500]
    print(f"Downloading {len(small)} small datasets (<500MB)...")
    for name in small:
        download_dataset(name)


def _parse_size(size_str):
    """Parse size string to MB."""
    size_str = size_str.upper().replace(" ", "")
    if "GB" in size_str:
        return float(size_str.replace("GB", "")) * 1024
    elif "MB" in size_str:
        return float(size_str.replace("MB", ""))
    elif "KB" in size_str:
        return float(size_str.replace("KB", "")) / 1024
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] == "--list":
        category = sys.argv[2] if len(sys.argv) > 2 else None
        list_datasets(category)
        print(f"\nTotal: {TOTAL_DATASETS} datasets")
    elif sys.argv[1] == "--all-small":
        download_small()
    else:
        download_dataset(sys.argv[1])
