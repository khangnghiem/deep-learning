import sys
import os

def setup():
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=True)
    except ImportError:
        pass
        
    repo_path = "/content/drive/MyDrive/repos/deep-learning"
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    print("✅ Deep Learning repo path added.")

if __name__ == "__main__":
    setup()
