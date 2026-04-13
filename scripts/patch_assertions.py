import os
import glob
import json
import re

def fix_assertions(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            nb = json.load(f)
        except:
            return False

    modified = False
    for cell in nb.get('cells', []):
        if cell.get('cell_type') != 'code': continue
        new_source = []
        skip_next = False
        for i, line in enumerate(cell.get('source', [])):
            if skip_next:
                skip_next = False
                continue
                
            orig_line = line
            
            if 'assert os.path.exists(DATASET_YAML)' in line:
                # Replace with the new nice error message
                replacement = (
                    "if not Path(DATASET_YAML).exists():\n"
                    "    raise FileNotFoundError(\n"
                    "        f\"❌ Dataset not found at: {DATASET_YAML}\\n\"\n"
                    "        \"We rely on the Medallion Data Lake architecture.\\n\"\n"
                    "        \"1. Create a `.env` file at the root of the repository.\\n\"\n"
                    "        \"2. Set `DATA_LAKE_DIR=/path/to/your/data_lake`.\\n\"\n"
                    "        \"3. Ensure the dataset is located at `{DATA_LAKE_DIR}/03_gold/...`\"\n"
                    "    )\n"
                )
                line = replacement
                modified = True
            
            new_source.append(line)
        cell['source'] = new_source

    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
            f.write('\n')
        return True
    return False

if __name__ == '__main__':
    files = glob.glob('experiments/**/*.ipynb', recursive=True) + glob.glob('explorations/*.ipynb')
    count = 0
    for f in files:
        if fix_assertions(f):
            print(f"Fixed: {f}")
            count += 1
    print(f"Fixed assertions in {count} notebooks.")
