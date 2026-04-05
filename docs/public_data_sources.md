# Public Data Sources for Deep Learning

A curated list of datasets for learning deep learning, organized by difficulty and domain.

---

## 🟢 Beginner (Start Here)

### Image Classification

| Dataset | Size | Classes | Description | Link |
|---------|------|---------|-------------|------|
| **MNIST** | 70K images | 10 | Handwritten digits (28×28 grayscale) | `torchvision.datasets.MNIST` |
| **Fashion-MNIST** | 70K images | 10 | Clothing items (28×28 grayscale) | `torchvision.datasets.FashionMNIST` |
| **CIFAR-10** | 60K images | 10 | Objects (32×32 color) | `torchvision.datasets.CIFAR10` |
| **CIFAR-100** | 60K images | 100 | Fine-grained objects | `torchvision.datasets.CIFAR100` |

### Text Classification

| Dataset | Size | Classes | Description | Link |
|---------|------|---------|-------------|------|
| **IMDB Reviews** | 50K reviews | 2 | Movie sentiment | `datasets.load_dataset("imdb")` |
| **AG News** | 120K articles | 4 | News categories | `datasets.load_dataset("ag_news")` |

---

## 🟡 Intermediate

### Image Classification (Larger)

| Dataset | Size | Classes | Description | Link |
|---------|------|---------|-------------|------|
| **ImageNet-1K** | 1.2M images | 1000 | General objects | [image-net.org](https://image-net.org) |
| **Stanford Dogs** | 20K images | 120 | Dog breeds | `datasets.load_dataset("stanford_dogs")` |
| **Food-101** | 101K images | 101 | Food categories | `datasets.load_dataset("food101")` |
| **Oxford Pets** | 7K images | 37 | Cat/dog breeds | `datasets.load_dataset("oxford_iiit_pet")` |

### Object Detection

| Dataset | Size | Classes | Description | Link |
|---------|------|---------|-------------|------|
| **COCO** | 330K images | 80 | Common objects | [cocodataset.org](https://cocodataset.org) |
| **Pascal VOC** | 11K images | 20 | Classic detection | `torchvision.datasets.VOCDetection` |

### Semantic Segmentation

| Dataset | Size | Classes | Description | Link |
|---------|------|---------|-------------|------|
| **Cityscapes** | 5K images | 30 | Urban driving scenes | [cityscapes-dataset.com](https://www.cityscapes-dataset.com) |
| **ADE20K** | 25K images | 150 | Scene parsing | `datasets.load_dataset("scene_parse_150")` |

### NLP

| Dataset | Size | Task | Description | Link |
|---------|------|------|-------------|------|
| **SQuAD** | 100K+ QA pairs | QA | Question answering | `datasets.load_dataset("squad")` |
| **GLUE** | Various | Multiple | NLP benchmark | `datasets.load_dataset("glue", "mrpc")` |
| **WikiText** | 100M tokens | LM | Language modeling | `datasets.load_dataset("wikitext")` |

---

## 🔴 Advanced

### Medical Imaging

| Dataset | Size | Task | Description | Link |
|---------|------|------|-------------|------|
| **ChestX-ray14** | 112K images | Multi-label | Chest X-rays | [NIH](https://nihcc.app.box.com/v/ChestXray-NIHCC) |
| **ISIC Skin Cancer** | 25K images | Classification | Dermoscopy images | [isic-archive.com](https://www.isic-archive.com) |
| **BraTS** | 2K+ MRIs | Segmentation | Brain tumors | [synapse.org](https://www.synapse.org/brats) |

### Audio

| Dataset | Size | Task | Description | Link |
|---------|------|------|-------------|------|
| **Speech Commands** | 100K clips | Classification | Spoken words | `torchaudio.datasets.SPEECHCOMMANDS` |
| **LibriSpeech** | 1000h audio | ASR | Audiobook speech | [openslr.org](https://www.openslr.org/12) |

### Generative / Multimodal

| Dataset | Size | Task | Description | Link |
|---------|------|------|-------------|------|
| **CelebA** | 200K images | Generation | Celebrity faces with attributes | `torchvision.datasets.CelebA` |
| **LAION-400M** | 400M pairs | CLIP training | Image-text pairs | [laion.ai](https://laion.ai) |

---

## Quick Load Examples

```python
# PyTorch Vision
from torchvision import datasets
mnist = datasets.MNIST(root='./data', download=True)
cifar = datasets.CIFAR10(root='./data', download=True)

# HuggingFace Datasets
from datasets import load_dataset
imdb = load_dataset("imdb")
squad = load_dataset("squad")

# Kaggle (requires API key)
# kaggle datasets download -d <dataset-name>
```

---

## Recommended Learning Path

1. **Week 1-2**: MNIST, Fashion-MNIST (basic CNNs)
2. **Week 3-4**: CIFAR-10/100 (deeper networks, augmentation)
3. **Week 5-6**: ImageNet subset, transfer learning
4. **Week 7-8**: COCO (object detection)
5. **Week 9+**: Domain-specific (medical, audio, NLP)
