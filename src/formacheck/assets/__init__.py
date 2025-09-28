from pathlib import Path

ASSETS_DIR = Path(__file__).parent
MODELS_DIR = ASSETS_DIR / "models"

if __name__ == "__main__":
    print(ASSETS_DIR)
    print(MODELS_DIR)