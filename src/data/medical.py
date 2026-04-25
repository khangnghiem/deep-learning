import logging

logger = logging.getLogger(__name__)
"""
Medical and Genetic Datasets.

Popular free medical imaging and genomics datasets.

Usage:
    from src.data.medical import get_medical_datasets, download_medmnist
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.paths import get_bronze_path

BRONZE_MEDICAL = get_bronze_path("medical")


def get_medical_datasets() -> dict:
    """
    Return dictionary of popular free medical datasets with download info.
    """
    return {
        # === MEDICAL IMAGING ===
        "chest_xray": {
            "name": "Chest X-Ray Images (Pneumonia)",
            "source": "kaggle",
            "id": "paultimothymooney/chest-xray-pneumonia",
            "size": "2GB",
            "classes": ["NORMAL", "PNEUMONIA"],
            "modality": "X-Ray",
        },
        "skin_cancer": {
            "name": "Skin Cancer MNIST (HAM10000)",
            "source": "kaggle",
            "id": "kmader/skin-cancer-mnist-ham10000",
            "size": "3GB",
            "classes": 7,
            "modality": "Dermoscopy",
        },
        "brain_tumor": {
            "name": "Brain Tumor MRI Dataset",
            "source": "kaggle",
            "id": "masoudnickparvar/brain-tumor-mri-dataset",
            "size": "150MB",
            "classes": ["glioma", "meningioma", "notumor", "pituitary"],
            "modality": "MRI",
        },
        "retinal_oct": {
            "name": "Retinal OCT Images",
            "source": "kaggle",
            "id": "paultimothymooney/kermany2018",
            "size": "5GB",
            "classes": ["CNV", "DME", "DRUSEN", "NORMAL"],
            "modality": "OCT",
        },
        "covid19_xray": {
            "name": "COVID-19 Radiography Database",
            "source": "kaggle",
            "id": "tawsifurrahman/covid19-radiography-database",
            "size": "1.5GB",
            "classes": ["COVID", "Normal", "Viral Pneumonia", "Lung Opacity"],
            "modality": "X-Ray",
        },
        "diabetic_retinopathy": {
            "name": "Diabetic Retinopathy Detection",
            "source": "kaggle",
            "id": "competitions/diabetic-retinopathy-detection",
            "size": "80GB",
            "classes": 5,
            "modality": "Fundus",
        },
        
        # === MEDMNIST (Small standardized datasets) ===
        "pathmnist": {
            "name": "PathMNIST (Colorectal Cancer)",
            "source": "huggingface",
            "id": "albertvillanova/medmnist-v2",
            "subset": "pathmnist",
            "size": "28x28",
            "classes": 9,
        },
        "dermamnist": {
            "name": "DermaMNIST (Skin Lesion)",
            "source": "huggingface",
            "id": "albertvillanova/medmnist-v2",
            "subset": "dermamnist",
            "size": "28x28",
            "classes": 7,
        },
        "octmnist": {
            "name": "OCTMNIST (Retinal OCT)",
            "source": "huggingface",
            "id": "albertvillanova/medmnist-v2",
            "subset": "octmnist",
            "size": "28x28",
            "classes": 4,
        },
        "pneumoniamnist": {
            "name": "PneumoniaMNIST",
            "source": "huggingface",
            "id": "albertvillanova/medmnist-v2",
            "subset": "pneumoniamnist",
            "size": "28x28",
            "classes": 2,
        },
        "bloodmnist": {
            "name": "BloodMNIST (Blood Cell)",
            "source": "huggingface",
            "id": "albertvillanova/medmnist-v2",
            "subset": "bloodmnist",
            "size": "28x28",
            "classes": 8,
        },
        "organamnist": {
            "name": "OrganAMNIST (Abdominal CT)",
            "source": "huggingface",
            "id": "albertvillanova/medmnist-v2",
            "subset": "organamnist",
            "size": "28x28",
            "classes": 11,
        },
        
        # === GENOMICS / GENETICS ===
        "gene_expression": {
            "name": "Gene Expression Cancer RNA-Seq",
            "source": "kaggle",
            "id": "murats/gene-expression-cancer-rnaseq",
            "size": "10MB",
            "task": "Cancer type classification from gene expression",
        },
        "breast_cancer_wisconsin": {
            "name": "Breast Cancer Wisconsin",
            "source": "sklearn",
            "id": "load_breast_cancer",
            "size": "Small",
            "task": "Malignant vs Benign classification",
        },
        "tcga_pancan": {
            "name": "TCGA Pan-Cancer Atlas",
            "source": "url",
            "url": "https://www.cancer.gov/tcga",
            "note": "Requires GDC Data Portal access",
            "task": "Multi-omics cancer analysis",
        },
    }


def download_medical_dataset(name: str):
    """
    Download a medical dataset by name.
    
    Args:
        name: Dataset name from get_medical_datasets()
    """
    datasets = get_medical_datasets()
    
    if name not in datasets:
        available = list(datasets.keys())
        raise ValueError(f"Unknown dataset: {name}. Available: {available}")
    
    info = datasets[name]
    source = info["source"]
    
    if source == "kaggle":
        from .kaggle import download_dataset, download_competition
        
        dataset_id = info["id"]
        output_dir = BRONZE_MEDICAL / name
        
        if "competitions" in dataset_id:
            competition = dataset_id.split("/")[-1]
            download_competition(competition, output_dir, category="medical")
        else:
            download_dataset(dataset_id, output_dir, category="medical")
        
        logger.info(f"Downloaded {name} to {output_dir}")
        return output_dir
    
    elif source == "huggingface":
        from .huggingface import load_hf_dataset
        
        dataset = load_hf_dataset(info["id"], name=info.get("subset"))
        logger.info(f"Loaded {name} from HuggingFace")
        return dataset
    
    elif source == "sklearn":
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        logger.info(f"Loaded {name} from sklearn")
        return data
    
    else:
        logger.info(f"Manual download required. Info: {info}")
        return None


def list_medical_datasets():
    """Pretty print available medical datasets."""
    datasets = get_medical_datasets()
    
    logger.info("\n=== MEDICAL IMAGING ===")
    for name, info in datasets.items():
        if info.get("modality"):
            logger.info(f"  {name}: {info['name']} ({info['modality']})")
    
    logger.info("\n=== MEDMNIST (28x28 standardized) ===")
    for name, info in datasets.items():
        if "mnist" in name:
            logger.info(f"  {name}: {info['name']}")
    
    logger.info("\n=== GENOMICS / GENETICS ===")
    for name, info in datasets.items():
        if info.get("task"):
            logger.info(f"  {name}: {info['name']}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    list_medical_datasets()
