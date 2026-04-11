import os
import glob

def clean_script(path):
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Simple find and replace for the deprecated imports
    new_content = content.replace('shared_config.paths', 'src.config.paths')
    new_content = new_content.replace('shared_config.catalog', 'src.config.catalog')
    new_content = new_content.replace('shared_config.manifest', 'src.config.manifest')
    new_content = new_content.replace('import shared_config', 'import src.config')
    new_content = new_content.replace('PROJECT_ROOT.parent / "shared_config"', 'PROJECT_ROOT')
    new_content = new_content.replace('PROJECT_ROOT.parent / "shared-config"', 'PROJECT_ROOT')
    new_content = new_content.replace('Path(__file__).resolve().parents[3] / "shared_config"', 'Path(__file__).resolve().parents[2]')
    
    if content != new_content:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Cleaned {path}")

# Target scripts
for py in glob.glob("g:/My Drive/repos/deep-learning/**/*.py", recursive=True):
    clean_script(py)

print("Python scrub complete.")
