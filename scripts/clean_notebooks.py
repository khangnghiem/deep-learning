import os
import json
import glob

def clean_notebook(path):
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    modified = False
    for cell in nb.get('cells', []):
        if cell.get('cell_type') != 'code':
            continue
            
        new_source = []
        for line in cell.get('source', []):
            if 'data-ingestion.git' in line or 'data-ingestion' in line:
                modified = True
                continue
            if 'shared-config' in line or 'shared_config' in line:
                modified = True
                continue
            new_source.append(line)
        
        if len(new_source) != len(cell.get('source', [])):
            cell['source'] = new_source
            
    if modified:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print(f"Cleaned {path}")

for ipynb in glob.glob("g:/My Drive/repos/deep-learning/**/*.ipynb", recursive=True):
    clean_notebook(ipynb)

for ipynb in glob.glob("g:/My Drive/data_lake/scripts/*.ipynb", recursive=True):
    clean_notebook(ipynb)

print("Scrub complete.")
