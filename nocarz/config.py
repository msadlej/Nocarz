from pathlib import Path


PROJ_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJ_ROOT / "models"
RESULTS_DIR = PROJ_ROOT / "results"
LOGS_DIR = RESULTS_DIR / "logs"

HOST = "0.0.0.0"
PORT = 8080
