from datetime import datetime
import pandas as pd

from nocarz.api.schemas import ListingResponse, ListingRequest
from nocarz.src.advanced_model import AdvancedModel
from nocarz.config import MODELS_DIR, LOGS_DIR
from nocarz.src.base_model import BaseModel


def get_base_prediction(listing: ListingRequest) -> ListingResponse:
    base_model = BaseModel()
    base_model.load(MODELS_DIR / "base_model.pkl")

    pred, type = base_model.predict(listing.host_id)
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

    log = pd.DataFrame(
        {
            "timestamp": [datetime.now().strftime("%d/%m/%Y %H:%M:%S")],
            "model": ["base"],
            "type": [type],
            "host_id": [listing.host_id],
            "property_type": [result.property_type],
            "room_type": [result.room_type],
            "bathrooms_text": [result.bathrooms_text],
            "accommodates": [result.accommodates],
            "bathrooms": [result.bathrooms],
            "bedrooms": [result.bedrooms],
            "beds": [result.beds],
            "price": [result.price],
        }
    )
    log.to_csv(LOGS_DIR / "logs.csv", mode="a", header=False, index=False)

    return result


def get_advanced_prediction(listing: ListingRequest) -> ListingResponse:
    advanced_model = AdvancedModel()
    advanced_model.load(MODELS_DIR / "advanced_model.pkl")

    df = pd.DataFrame(
        {
            "name": [listing.name],
            "description": [listing.description],
            "neighbourhood": [listing.neighbourhood],
        }
    )
    pred = advanced_model.predict(df)

    result = ListingResponse(
        property_type=pred["property_type"],
        room_type=pred["room_type"],
        bathrooms_text=pred["bathrooms_text"],
        accommodates=int(round(pred["accommodates"])),
        bathrooms=int(round(pred["bathrooms"])),
        bedrooms=int(round(pred["bedrooms"])),
        beds=int(round(pred["beds"])),
        price=float(round(pred["price"], 2)),
    )

    log = pd.DataFrame(
        {
            "timestamp": [datetime.now().strftime("%d/%m/%Y %H:%M:%S")],
            "model": ["advanced"],
            "type": ["features"],
            "host_id": [listing.host_id],
            "property_type": [result.property_type],
            "room_type": [result.room_type],
            "bathrooms_text": [result.bathrooms_text],
            "accommodates": [result.accommodates],
            "bathrooms": [result.bathrooms],
            "bedrooms": [result.bedrooms],
            "beds": [result.beds],
            "price": [result.price],
        }
    )
    log.to_csv(LOGS_DIR / "logs.csv", mode="a", header=False, index=False)

    return result
