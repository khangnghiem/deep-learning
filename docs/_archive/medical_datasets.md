# Medical Imaging Datasets

This document describes the medical imaging datasets available in the deep learning project catalog, organized by imaging modality.

## Quick Start

Download a few small datasets to get started:

```bash
# Download priority medical datasets (~3GB total)
python scripts/batch_download.py --medical

# Or download specific modalities
python scripts/ingest_catalog.py breast-ultrasound   # 200MB Ultrasound
python scripts/ingest_catalog.py chest-xray         # 1.2GB X-ray
python scripts/ingest_catalog.py covid-ct           # 400MB CT
python scripts/ingest_catalog.py brain-tumor        # 100MB MRI
```

## Datasets by Modality

### 🔊 Ultrasound

| Dataset              | Size  | Classes | Description                              |
| -------------------- | ----- | ------- | ---------------------------------------- |
| `breast-ultrasound`  | 200MB | 3       | Normal, benign, malignant classification |
| `nerve-ultrasound`   | 200MB | -       | Brachial plexus nerve segmentation       |
| `fetal-ultrasound`   | 150MB | -       | Fetal head plane classification          |
| `thyroid-ultrasound` | 100MB | -       | Thyroid nodule detection                 |
| `carotid-ultrasound` | 500MB | -       | Carotid artery stenosis detection        |

### 📸 X-ray

| Dataset             | Size  | Classes | Description                               |
| ------------------- | ----- | ------- | ----------------------------------------- |
| `chest-xray`        | 1.2GB | 2       | Pneumonia detection (normal/pneumonia)    |
| `nih-chest-xray`    | 42GB  | 14      | NIH dataset with 112k images, 14 diseases |
| `pediatric-xray`    | 300MB | 2       | Pediatric pneumonia detection             |
| `mura-xray`         | 3GB   | 2       | Musculoskeletal radiograph abnormality    |
| `rsna-pneumonia`    | 2GB   | 2       | RSNA pneumonia detection challenge        |
| `covid-xray`        | 100MB | 3       | COVID-19/Pneumonia/Normal classification  |
| `dental-xray`       | 200MB | -       | Teeth segmentation                        |
| `bone-fracture`     | 200MB | 7       | Multi-region fracture detection           |
| `tuberculosis-xray` | 700MB | 2       | TB detection from chest X-rays            |

### 🔬 CT Scan

| Dataset            | Size  | Classes | Description                         |
| ------------------ | ----- | ------- | ----------------------------------- |
| `covid-ct`         | 400MB | 2       | COVID-19 CT scan classification     |
| `kidney-ct`        | 500MB | 4       | Kidney cyst/tumor/stone detection   |
| `liver-ct`         | 500MB | -       | Liver tumor segmentation            |
| `lung-ct`          | 200MB | 4       | Lung cancer classification          |
| `head-ct`          | 200MB | 2       | Intracranial hemorrhage detection   |
| `kits19-kidney-ct` | 30GB  | -       | Kidney tumor segmentation challenge |
| `luna16-lung-ct`   | 100GB | -       | Lung nodule analysis                |

### 🧠 MRI

| Dataset           | Size  | Classes | Description                     |
| ----------------- | ----- | ------- | ------------------------------- |
| `brain-tumor`     | 100MB | 4       | Brain tumor classification      |
| `alzheimer-mri`   | 100MB | 4       | Alzheimer's disease stages      |
| `prostate-mri`    | 500MB | -       | Prostate cancer grading         |
| `knee-mri`        | 200MB | 2       | ACL/meniscus tear detection     |
| `cardiac-mri`     | 1GB   | -       | Heart volume estimation         |
| `brats-mri`       | 5GB   | 4       | Brain tumor segmentation        |
| `oasis-brain-mri` | 100MB | -       | Brain aging/Alzheimer's dataset |
| `fastmri-brain`   | 3GB   | -       | FastMRI brain reconstruction    |

### 🔍 Other Medical

| Dataset                | Size  | Classes | Description                       |
| ---------------------- | ----- | ------- | --------------------------------- |
| `skin-cancer`          | 500MB | 7       | HAM10000 dermatoscopy images      |
| `diabetic-retinopathy` | 80GB  | 5       | Retinal fundus images             |
| `malaria`              | 350MB | 2       | Cell images for malaria detection |
| `blood-cells`          | 350MB | 4       | Blood cell classification         |
| `eye-disease`          | 500MB | 4       | Eye disease classification        |
| `lung-cancer`          | 100MB | 3       | Lung cancer histopathology        |
| `colon-cancer`         | 1.5GB | 5       | Colon cancer histopathology       |
| `leukemia`             | 100MB | 2       | Leukemia classification           |

## Data Storage

All datasets are stored in the medallion data lake under:

```
data_lake/01_bronze_medical/{dataset_name}/
```

## Usage in Training

```python
from config.paths import BRONZE_MEDICAL

# Load a dataset
dataset_path = BRONZE_MEDICAL / "chest-xray"

# Use with PyTorch ImageFolder
from torchvision.datasets import ImageFolder
train_data = ImageFolder(dataset_path / "train")
```

## Citation Requirements

Most datasets require proper citation. Check the original Kaggle/source page for specific citation requirements.

## Recommended Learning Path

1. **Start Small**: `brain-tumor` (100MB), `breast-ultrasound` (200MB)
2. **Classic Problems**: `chest-xray` (pneumonia), `skin-cancer` (dermatology)
3. **Competition Data**: `rsna-pneumonia`, `diabetic-retinopathy`
4. **Advanced**: `brats-mri` (segmentation), `nih-chest-xray` (multi-label)
