import os
import sys
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm

# Reconfigure stdout and stderr to use UTF-8 to prevent encoding crashes on Windows console.
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except AttributeError:
    pass

# We resolve the local target paths relative to this script
BASE_DIR = Path(__file__).resolve().parent
TARGET_PIPER_DIR = BASE_DIR / "tts" / "piper"

PIPER_ZIP_URL = "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_windows_amd64.zip"
VOICE_ONNX_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
VOICE_JSON_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"


def download_file(url: str, dest_path: Path):
    """Downloads a file from a URL to a local destination showing a progress bar."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}...")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192
    
    with open(dest_path, "wb") as f, tqdm(
        total=total_size, unit='iB', unit_scale=True, desc=dest_path.name
    ) as bar:
        for data in response.iter_content(block_size):
            size = f.write(data)
            bar.update(size)

def main():
    print("=== Piper TTS Windows Setup Script ===")
    
    # 1. Download Piper ZIP
    zip_temp_path = BASE_DIR / "piper_temp.zip"
    try:
        if not (TARGET_PIPER_DIR / "piper.exe").exists():
            download_file(PIPER_ZIP_URL, zip_temp_path)
            
            print("\nExtracting Piper binary...")
            TARGET_PIPER_DIR.mkdir(parents=True, exist_ok=True)
            
            with zipfile.ZipFile(zip_temp_path, 'r') as zip_ref:
                # The zip file contains a nested folder 'piper/'
                # We extract it to a temporary location or extract and move files
                zip_ref.extractall(BASE_DIR / "piper_extracted")
                
            # Move files from piper_extracted/piper/* to tts/piper/*
            extracted_folder = BASE_DIR / "piper_extracted" / "piper"
            for item in extracted_folder.iterdir():
                dest_item = TARGET_PIPER_DIR / item.name
                if dest_item.exists():
                    if dest_item.is_file():
                        os.remove(dest_item)
                    else:
                        import shutil
                        shutil.rmtree(dest_item)
                os.rename(item, dest_item)
                
            # Clean up
            import shutil
            shutil.rmtree(BASE_DIR / "piper_extracted")
            if zip_temp_path.exists():
                os.remove(zip_temp_path)
            print("✓ Piper binary extracted to tts/piper/")
        else:
            print("✓ Piper executable already exists. Skipping download.")
            
    except Exception as e:
        print(f"❌ Failed to setup Piper binary: {e}")
        if zip_temp_path.exists():
            os.remove(zip_temp_path)
        sys.exit(1)

    # 2. Download Voice ONNX
    voice_onnx_path = TARGET_PIPER_DIR / "en_US-lessac-medium.onnx"
    try:
        if not voice_onnx_path.exists():
            download_file(VOICE_ONNX_URL, voice_onnx_path)
            print("✓ Voice ONNX model downloaded.")
        else:
            print("✓ Voice ONNX model already exists. Skipping download.")
    except Exception as e:
        print(f"❌ Failed to download voice model: {e}")
        sys.exit(1)

    # 3. Download Voice Config JSON
    voice_json_path = TARGET_PIPER_DIR / "en_US-lessac-medium.onnx.json"
    try:
        if not voice_json_path.exists():
            download_file(VOICE_JSON_URL, voice_json_path)
            print("✓ Voice Config JSON downloaded.")
        else:
            print("✓ Voice Config JSON already exists. Skipping download.")
    except Exception as e:
        print(f"❌ Failed to download voice config JSON: {e}")
        sys.exit(1)

    print("\n🎉 Piper TTS is successfully set up locally under voice_assistant/tts/piper/")

if __name__ == "__main__":
    main()
