import os
import logging
from huggingface_hub import snapshot_download

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELS_TO_DOWNLOAD = [
    "Qwen/Qwen3-VL-4B-Instruct-FP8",
]

def download_model(model_id):
    logger.info(f"Downloading model snapshot: {model_id}...")
    try:
        # Just download files to cache, don't load into memory/GPU
        path = snapshot_download(
            repo_id=model_id,
            ignore_patterns=["*.msgpack", "*.h5", "*.ot"] # Optional cleanup
        )
        logger.info(f"Successfully downloaded {model_id} to {path}")
        return True
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to download {model_id} - {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("Starting model download process...")
    
    success = False
    for model_id in MODELS_TO_DOWNLOAD:
        if download_model(model_id):
            success = True
            
    if success:
        logger.info("Model download complete. You can now run the shorts maker.")
    else:
        logger.error("All model downloads failed.")
