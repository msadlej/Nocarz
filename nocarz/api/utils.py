from nocarz.config import MODELS_DIR, LOGS_DIR
from nocarz.api.schemas import ListingResponse, AdvancedListingRequest
from nocarz.src.base_model import BaseModel
from datetime import datetime
import pandas as pd
import pickle


regressor_path = MODELS_DIR / "random_forest_regressor.pkl"
classifier_path = MODELS_DIR / "random_forest_classifier.pkl"
vectorizer_path = MODELS_DIR / "tfidf_vectorizer.pkl"
label_encoders_path = MODELS_DIR / "label_encoders.pkl"


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

    log = pd.DataFrame(
        {
            "timestamp": [datetime.now().strftime("%d/%m/%Y %H:%M:%S")],
            "model": ["base"],
            "type": [type],
            "host_id": [host_id],
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


def get_advanced_prediction(listing: AdvancedListingRequest) -> ListingResponse:
    with open(regressor_path, "rb") as f:
        regressor = pickle.load(f)
    with open(classifier_path, "rb") as f:
        classifier = pickle.load(f)
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    with open(label_encoders_path, "rb") as f:
        le_dict = pickle.load(f)

    text_input = " ".join([listing.name, listing.description, listing.neighbourhood or ""])
    X_text = vectorizer.transform([text_input])

    y_pred_reg = regressor.predict(X_text)[0]
    y_pred_cls = classifier.predict(X_text)[0]

    categorical_cols = ["property_type", "room_type", "bathrooms_text"]
    decoded_cls = {col: le_dict[col].inverse_transform([y_pred_cls[i]])[0] for i, col in enumerate(categorical_cols)}

    result = ListingResponse(
        property_type=decoded_cls["property_type"],
        room_type=decoded_cls["room_type"],
        bathrooms_text=decoded_cls["bathrooms_text"],
        accommodates=int(round(y_pred_reg[0])),
        bathrooms=int(round(y_pred_reg[1])),
        bedrooms=int(round(y_pred_reg[2])),
        beds=int(round(y_pred_reg[3])),
        price=float(round(y_pred_reg[4], 2)),
    )

    log = pd.DataFrame(
        {
            "timestamp": [datetime.now().strftime("%d/%m/%Y %H:%M:%S")],
            "model": ["advanced"],
            "type": ["features"],
            "host_id": [-1],
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
