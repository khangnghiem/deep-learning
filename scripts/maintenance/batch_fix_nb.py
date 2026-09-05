import os
import json

exps = ["017_polyp_rtdetr", "018_polyp_mobilesam", "019_polyp_yolact", "020_polyp_ensemble"]
repo = 'g:/My Drive/repos/deep-learning'

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
    "        os.system('git clone --depth 1 -q https://github.com/khangnghiem/deep-learning.git /content/deep-learning')\n",
    "    \n",
    "    REPO_ROOT = Path('/content/deep-learning')\n",
    "    # os.chdir dynamically added below\n",
    "except ImportError:\n",
    "    # Local Environment\n",
    "    cur = Path().resolve()\n",
    "    REPO_ROOT = cur.parent if cur.name in ('explorations', 'experiments') else cur.parents[1]\n",
    "\n",
    "if str(REPO_ROOT) not in sys.path:\n",
    "    sys.path.insert(0, str(REPO_ROOT))\n",
    "\n",
    "from src.config.paths import GOLD, BRONZE, SILVER, DATA_LAKE, MODELS, PRETRAINED, TRAINED, OPS, MLFLOW_TRACKING_URI, REPOS\n"
]

for exp in exps:
    # Fix train.ipynb
    train_nb = os.path.join(repo, f"experiments/{exp}/{exp}_train.ipynb")
    if os.path.exists(train_nb):
        with open(train_nb, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        
        # Build specific cell
        exp_setup = new_setup_source.copy()
        # insert os.chdir
        for i, line in enumerate(exp_setup):
            if "os.chdir dynamically added" in line:
                exp_setup[i] = f"    os.chdir(REPO_ROOT / 'experiments/{exp}')\n"
                break
        
        # Replace first cell
        if nb['cells'][0]['cell_type'] == 'code':
            nb['cells'][0]['source'] = exp_setup
        
        # Remove duplicate git clone cell:
        for i in range(min(5, len(nb['cells'])-1), 0, -1):
            if nb['cells'][i].get('source') and "git clone" in "".join(nb['cells'][i].get('source', [])):
                del nb['cells'][i]
                
        with open(train_nb, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
            
    # Fix config.yaml
    conf_path = os.path.join(repo, f"experiments/{exp}/config.yaml")
    if os.path.exists(conf_path):
        with open(conf_path, 'r', encoding='utf-8') as f:
            text = f.read()
        text = text.replace("batch_size: 16", "batch_size: 32")
        text = text.replace("num_workers: 4", "num_workers: 8")
        text = text.replace("num_workers: 2", "num_workers: 8")
        with open(conf_path, 'w', encoding='utf-8') as f:
            f.write(text)
            
    # Fix explorations
    exp_nb = os.path.join(repo, f"explorations/{exp}.ipynb")
    if os.path.exists(exp_nb):
        with open(exp_nb, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        if nb['cells'][0]['cell_type'] == 'code':
            
            exp_setup2 = new_setup_source.copy()
            for i, line in enumerate(exp_setup2):
                if "os.chdir dynamically added" in line:
                    exp_setup2[i] = ""
                    break
                    
            nb['cells'][0]['source'] = exp_setup2
            
        with open(exp_nb, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)

print("Done fixing notebooks")
