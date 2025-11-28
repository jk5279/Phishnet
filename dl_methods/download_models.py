"""
Script to download pretrained transformer models and tokenizers for offline use.
Saves models to models/pretrained/ directory.
"""

import os
import shutil
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# =========================
# Configuration
# =========================

MODELS_TO_DOWNLOAD = {
    "bert-base-cased": "bert-base-cased",
    "roberta-base": "roberta-base",
    "distilbert-base-cased": "distilbert-base-cased"
}

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "pretrained")

def download_model(model_name: str, model_id: str):
    """Download model and tokenizer to local directory."""
    print(f"\nDownloading {model_name} ({model_id})...")
    
    save_path = os.path.join(OUTPUT_DIR, model_name)
    
    if os.path.exists(save_path):
        print(f"Directory {save_path} already exists. Skipping download (delete to force re-download).")
        return

    os.makedirs(save_path, exist_ok=True)
    
    try:
        # Download tokenizer
        print(" - Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.save_pretrained(save_path)
        
        # Download model
        print(" - Downloading model weights...")
        model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2) # Num labels doesn't matter for weights usually, but config needs it
        model.save_pretrained(save_path)
        
        print(f"✅ Successfully saved {model_name} to {save_path}")
        
    except Exception as e:
        print(f"❌ Failed to download {model_name}: {e}")
        # Cleanup partial download
        if os.path.exists(save_path):
            shutil.rmtree(save_path)

def main():
    print("=" * 60)
    print("Pretrained Model Downloader")
    print("=" * 60)
    print(f"Target directory: {OUTPUT_DIR}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for name, model_id in MODELS_TO_DOWNLOAD.items():
        download_model(name, model_id)
        
    print("\n" + "=" * 60)
    print("Download complete!")
    print("To transfer to compute cluster, copy the entire 'models/pretrained' directory.")
    print("=" * 60)

if __name__ == "__main__":
    main()

