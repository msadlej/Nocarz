from nocarz.config import MODELS_DIR, LOGS_DIR
from nocarz.api.schemas import ListingResponse
from nocarz.src.base_model import BaseModel
from datetime import datetime
import pandas as pd


def get_base_prediction(host_id: int) -> ListingResponse:
    base_model = BaseModel()
    base_model.load(MODELS_DIR / "base_model.pkl")

    pred, type = base_model.predict(host_id)
    result = ListingResponse(
        property_type=pred["property_type"],
        room_type=pred["room_type"],
        bathrooms_text=pred["bathrooms_text"],
        accommodates=pred["accommodates"],
        bathrooms=pred["bathrooms"],
        bedrooms=pred["bedrooms"],
        beds=pred["beds"],
        price=pred["price"],
    )

    log = pd.DataFrame({
        'timestamp': [datetime.now().strftime('%d/%m/%Y %H:%M:%S')],
        'model': ['base'],
        'type': [type],
        'host_id': [host_id],
        'property_type': [result.property_type],
        'room_type': [result.room_type],
        'bathrooms_text': [result.bathrooms_text],
        'accommodates': [result.accommodates],
        'bathrooms': [result.bathrooms],
        'bedrooms': [result.bedrooms],
        'beds': [result.beds],
        'price': [result.price]
    })
    log.to_csv(LOGS_DIR / "logs.csv", mode='a', header=False, index=False)

    return result
