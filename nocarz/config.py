from pathlib import Path


PROJ_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJ_ROOT / "models"
LOGS_DIR = PROJ_ROOT / "logs"

ID_COLUMNS = ["id", "host_id"]
INPUT_COLUMNS = ["name", "description", "neighbourhood"]
NUMERICAL_TARGETS = ["accommodates", "bathrooms", "bedrooms", "beds", "price"]
CATEGORICAL_TARGETS = ["property_type", "room_type", "bathrooms_text"]

HOST = "0.0.0.0"
PORT = 8080
