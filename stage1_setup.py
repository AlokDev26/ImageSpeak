import sys
import os
import importlib

# ---------------------------------------------------------------------------
# 1. Check Python version
# ---------------------------------------------------------------------------
def check_python_version():
    print("Checking Python version...")
    major, minor = sys.version_info[0], sys.version_info[1]
    version_str = f"{major}.{minor}"
    if (major, minor) < (3, 9):
        print(f"  WARNING: Python {version_str} detected. Python 3.9+ is recommended.")
    else:
        print(f"  OK: Python {version_str}")


# ---------------------------------------------------------------------------
# 2. Check required libraries
# ---------------------------------------------------------------------------
REQUIRED_LIBRARIES = [
    "torch",
    "torchvision",
    "numpy",
    "pandas",
    "PIL",          # comes from the "pillow" package
    "matplotlib",
]

OPTIONAL_LIBRARIES = [
    "transformers",     # needed later if you fine-tune BLIP/GIT etc.
    "nltk",             # needed later for evaluation metrics
]


def check_libraries():
    print("\nChecking required libraries...")
    missing = []
    for lib in REQUIRED_LIBRARIES:
        try:
            importlib.import_module(lib)
            print(f"  OK: {lib}")
        except ImportError:
            print(f"  MISSING: {lib}")
            missing.append(lib)

    print("\nChecking optional libraries (needed in later stages)...")
    for lib in OPTIONAL_LIBRARIES:
        try:
            importlib.import_module(lib)
            print(f"  OK: {lib}")
        except ImportError:
            print(f"  NOT INSTALLED (ok for now): {lib}")

    if missing:
        print("\nTo install missing required libraries, run:")
        install_names = ["pillow" if lib == "PIL" else lib for lib in missing]
        print(f"  pip install {' '.join(install_names)}")

    return missing


# ---------------------------------------------------------------------------
# 3. Check GPU availability
# ---------------------------------------------------------------------------
def check_gpu():
    print("\nChecking GPU availability...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  GPU detected: {torch.cuda.get_device_name(0)}")
        else:
            print("  No GPU detected. Training will work but be much slower.")
            print("  Consider using Google Colab or Kaggle Kernels (free GPU access).")
    except ImportError:
        print("  Could not check (torch not installed yet).")


# ---------------------------------------------------------------------------
# 4. Create project folder structure
# ---------------------------------------------------------------------------
PROJECT_FOLDERS = [
    "data/raw",              # original downloaded dataset (images + captions file)
    "data/prepared",         # cleaned/tokenized captions, vocab, splits
    "data/features",         # cached CNN feature vectors per image
    "models/checkpoints",    # saved model weights during training
    "notebooks",             # exploration / experiment notebooks
    "src",                   # source code (data prep, model, train, inference)
    "outputs/sample_captions",  # generated captions on sample/test images for review
    "logs",                  # training logs, tensorboard runs, etc.
]


def create_folder_structure(base_dir="image_caption_project"):
    print(f"\nCreating project folder structure under ./{base_dir}/ ...")
    for folder in PROJECT_FOLDERS:
        path = os.path.join(base_dir, folder)
        os.makedirs(path, exist_ok=True)
        print(f"  Created: {path}")
    return base_dir


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    check_python_version()
    check_libraries()
    check_gpu()
    project_dir = create_folder_structure()

    print(f"\nStage 1 setup complete. Project root: ./{project_dir}/")
    print("Next: place your dataset (images + captions file) into 'data/raw/', then move to Stage 2 (Data Collection & Preparation).")


if __name__ == "__main__":
    main()