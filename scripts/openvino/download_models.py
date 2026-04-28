import argparse
from huggingface_hub import snapshot_download

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Download models from HuggingFace")
parser.add_argument("file", help="Path to file containing list of models (one per line)")
args = parser.parse_args()

# Read models from file
with open(args.file, "r") as f:
    models = [line.strip().strip('"') for line in f if line.strip() and not line.strip().startswith("#")]

# Base directory where models will be stored
base_dir = "./models"

for model in models:
    print(f"\nDownloading {model} ...")
    try:
        # Remove organization/username prefix (everything before first '/')
        folder_name = model.split('/', 1)[1]
        local_path = snapshot_download(
            repo_id=model,
            local_dir=f"{base_dir}/{folder_name}",
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print(f"Saved to: {local_path}")
    except Exception as e:
        print(f"Failed to download {model}: {e}")