import json

path = 'experiments/016_polyp_fast_diag/016_polyp_fast_diag_pipeline.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_setup_source = [
    "# === 1. Standalone Environment Setup ===\n",
    "import sys, os\n",
    "from pathlib import Path\n",
    "\n",
    "try:\n",
    "    # Colab Environment\n",
    "    import google.colab\n",
    "    from google.colab import drive, runtime\n",
    "    drive.mount('/content/drive')\n",
    "    \n",
    "    # Clone repository if it doesn't exist\n",
    "    if not os.path.exists('/content/deep-learning'):\n",
    "        !git clone --depth 1 -q https://github.com/khangnghiem/deep-learning.git /content/deep-learning\n",
    "    \n",
    "    REPO_ROOT = Path('/content/deep-learning')\n",
    "    os.chdir(REPO_ROOT / 'experiments/016_polyp_fast_diag')\n",
    "    \n",
    "    if str(REPO_ROOT) not in sys.path:\n",
    "        sys.path.insert(0, str(REPO_ROOT))\n",
    "        \n",
    "except ImportError:\n",
    "    # Local Environment\n",
    "    cur = Path().resolve()\n",
    "    REPO_ROOT = cur.parent if cur.name in ('explorations', 'experiments') else cur.parents[1]\n",
    "    if str(REPO_ROOT) not in sys.path:\n",
    "        sys.path.insert(0, str(REPO_ROOT))\n",
    "\n",
    "from src.config.paths import GOLD, BRONZE, SILVER, DATA_LAKE, MODELS, PRETRAINED, TRAINED, OPS, MLFLOW_TRACKING_URI, REPOS\n"
]

if nb['cells'][0]['cell_type'] == 'code' and "REPO_ROOT" in "".join(nb['cells'][0]['source']):
    nb['cells'][0]['source'] = new_setup_source
    
    if nb['cells'][1]['cell_type'] == 'code' and "git clone" in "".join(nb['cells'][1]['source']):
        del nb['cells'][1]

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print("fixed")
