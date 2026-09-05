import json
import os

def update_notebook(path, find_replacements):
    if not os.path.exists(path):
        print(f"{path} not found.")
        return
        
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    modified = False
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            new_source = []
            for line in cell.get('source', []):
                new_line = line
                for f_str, r_str in find_replacements:
                    if f_str in new_line:
                        new_line = new_line.replace(f_str, r_str)
                        modified = True
                new_source.append(new_line)
            cell['source'] = new_source
            
    if modified:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print(f"Updated {path}")
    else:
        print(f"No changes made to {path}")

# Pipeline Notebook updates: increase batch size explicitly
pipeline_reps = [
    ("--batch-size 16", "--batch-size 32")
]
update_notebook('experiments/016_polyp_fast_diag/016_polyp_fast_diag_pipeline.ipynb', pipeline_reps)

# Explore Notebook updates: higher batch and workers
explore_reps = [
    ("    batch=16,\\n", "    batch=32,\\n    workers=8,  # L4 High RAM capacity\\n")
]
update_notebook('explorations/016_polyp_fast_diag.ipynb', explore_reps)
