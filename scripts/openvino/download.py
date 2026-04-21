from huggingface_hub import snapshot_download

# List of models to download
models = [
    "meta-llama/Llama-3.2-1B-Instruct"
]

# Base directory where models will be stored
base_dir = "./models"

for model in models:
    print(f"\nDownloading {model} ...")
    try:
        local_path = snapshot_download(
            repo_id=model,
            local_dir=f"{base_dir}/{model.replace('/', '_')}",
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print(f"Saved to: {local_path}")
    except Exception as e:
        print(f"Failed to download {model}: {e}")