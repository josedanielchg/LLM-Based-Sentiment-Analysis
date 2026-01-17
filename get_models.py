import os
import shutil
import zipfile
import gdown
from pathlib import Path

# === CONFIGURATION ===
MODEL_DRIVE_ID = "1AEt76pbq5KZJb7xu5IzD2OUIB-smOw8v" 
BASE_DIR = Path(__file__).parent
folder_name = ""
def setup_models():
    
    print(f"[DOWNLOAD] Downloading models (ID: {MODEL_DRIVE_ID})...")
    url = f'https://drive.google.com/uc?id={MODEL_DRIVE_ID}'
    
    zip_filename = gdown.download(url, quiet=False)

    if not zip_filename:
        print("[ERROR] Download failed.")
        return

    zip_path = Path(zip_filename)
    folder_name = zip_path.stem + "s" 
    MODELS_DIR = BASE_DIR / folder_name

    if MODELS_DIR.exists():
        print(f"[RESET] Deleting existing folder '{MODELS_DIR.name}'...")
        shutil.rmtree(MODELS_DIR)
    
    MODELS_DIR.mkdir(exist_ok=True)

    print(f"[UNZIP] Extracting to '{MODELS_DIR.name}'...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(MODELS_DIR)
        print("[OK] Extraction complete.")
    except zipfile.BadZipFile:
        print("[ERROR] The downloaded file is not a valid zip.")
        return
    
    items = list(MODELS_DIR.iterdir()) 
    
    if len(items) == 1 and items[0].is_dir():
        nested_folder = items[0]

        for file_path in nested_folder.iterdir():
            shutil.move(str(file_path), str(MODELS_DIR))
        
        nested_folder.rmdir()

    if zip_path.exists():
        os.remove(zip_path)
        print(f"[CLEANUP] Removed temporary file: {zip_path.name}")

if __name__ == "__main__":
    try:
        setup_models()
        print(f"[SUCCESS] Todo listo en la carpeta: {folder_name}")
    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")