from nocarz.config import MODELS_DIR, LOGS_DIR
from nocarz.api.schemas import ListingResponse
from nocarz.src.base_model import BaseModel
from datetime import datetime


def get_base_prediction(host_id: int) -> ListingResponse:
    base_model = BaseModel()
    base_model.load(MODELS_DIR / "base_model.pkl")

    pred, type = base_model.predict(host_id)

    with open(LOGS_DIR / "logs.csv", "a") as f:
        timestamp = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        f.write(f"{timestamp},base,{host_id},{pred},{type}\n")

    return ListingResponse(
        accommodates=pred["accommodates"],
        bathrooms=pred["bathrooms"],
        bedrooms=pred["bedrooms"],
        beds=pred["beds"],
        price=pred["price"],
        property_type=pred["property_type"],
        room_type=pred["room_type"],
        bathrooms_text=pred["bathrooms_text"],
    )
