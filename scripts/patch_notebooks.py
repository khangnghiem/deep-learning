import os
import glob
import json
import re

BOILERPLATE_SOURCE = [
    "# --- ENVIRONMENT SETUP: Environment-Aware Paths ---\n",
    "import sys, os\n",
    "from pathlib import Path\n",
    "\n",
    "try:\n",
    "    # Standard Colab setup\n",
    "    import google.colab\n",
    "    from google.colab import drive\n",
    "    drive.mount('/content/drive')\n",
    "    REPO_ROOT = Path('/content/drive/MyDrive/repos/deep-learning')\n",
    "except ImportError:\n",
    "    # Local fallback (assumes running from explorations/ or experiments/)\n",
    "    cur = Path().resolve()\n",
    "    REPO_ROOT = cur.parent if cur.name in ('explorations', 'experiments') else cur.parents[1]\n",
    "\n",
    "if str(REPO_ROOT) not in sys.path:\n",
    "    sys.path.insert(0, str(REPO_ROOT))\n",
    "\n",
    "from src.config.paths import GOLD, BRONZE, SILVER, DATA_LAKE, MODELS, PRETRAINED, TRAINED, OPS, MLFLOW_TRACKING_URI, REPOS\n"
]

BOILERPLATE_CELL = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": BOILERPLATE_SOURCE
}

def process_notebook(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            nb = json.load(f)
        except:
            return False

    has_boilerplate = False
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            src_str = ''.join(cell.get('source', []))
            if 'from src.config.paths import' in src_str or 'src.config.paths' in src_str:
                has_boilerplate = True
                break

    if not has_boilerplate:
        nb['cells'].insert(0, BOILERPLATE_CELL)

    modified = False
    for cell in nb.get('cells', []):
        if cell.get('cell_type') != 'code': continue
        new_source = []
        for line in cell.get('source', []):
            orig_line = line
            
            # Replaces
            line = re.sub(r"['\"]file:///content/drive/MyDrive/ops/mlflow/mlruns['\"]", "MLFLOW_TRACKING_URI", line)
            line = re.sub(r"['\"]/content/drive/MyDrive/data_lake/03_gold/([^'\"]+)['\"]", r"str(GOLD / '\1')", line)
            line = re.sub(r"['\"]/content/drive/MyDrive/data_lake/03_gold['\"]", "str(GOLD)", line)
            line = re.sub(r"['\"]/content/drive/MyDrive/models/trained/([^'\"]+)['\"]", r"str(TRAINED / '\1')", line)
            line = re.sub(r"['\"]/content/drive/MyDrive/models/trained['\"]", "str(TRAINED)", line)
            line = re.sub(r"['\"]/content/drive/MyDrive/models/tuning/([^'\"]+)['\"]", r"str(MODELS / 'tuning' / '\1')", line)
            line = re.sub(r"['\"]/content/drive/MyDrive/repos/deep-learning/([^'\"]+)['\"]", r"str(REPO_ROOT / '\1')", line)
            line = re.sub(r"['\"]/content/drive/MyDrive/repos/deep-learning['\"]", "str(REPO_ROOT)", line)
            
            if orig_line != line:
                modified = True
            new_source.append(line)
        cell['source'] = new_source

    if modified or not has_boilerplate:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
            # Add trailing newline for format standard
            f.write('\n')
        return True
    return False

if __name__ == '__main__':
    files = glob.glob('experiments/**/*.ipynb', recursive=True) + glob.glob('explorations/*.ipynb')
    count = 0
    for f in files:
        if process_notebook(f):
            print(f"Updated: {f}")
            count += 1
    print(f"Updated {count} notebooks.")
