from datetime import datetime
import pandas as pd
import subprocess
import contextlib
import requests
import time

from nocarz.api.schemas import ListingResponse, ListingRequest
from nocarz.config import MODELS_DIR, LOGS_DIR, HOST, PORT
from nocarz.src.advanced_model import AdvancedModel
from nocarz.src.base_model import BaseModel


def get_base_prediction(listing: ListingRequest) -> ListingResponse:
    """
    Get prediction using the base model.

    Args:
        listing (ListingRequest): The listing request containing host_id.

    Returns:
        ListingResponse: The predicted listing response.
    """

    base_model = BaseModel()
    base_model.load(MODELS_DIR / "base_model.pkl")

    pred, type = base_model.predict(listing.host_id)
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
            "model": ["base"],
            "type": [type],
            "id": [listing.id],
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
    """
    Get prediction using the advanced model.

    Args:
        listing (ListingRequest): The listing request containing name, description, and neighbourhood.

    Returns:
        ListingResponse: The predicted listing response.
    """

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
            "id": [listing.id],
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


@contextlib.contextmanager
def get_microservice():
    """
    Context manager to start the microservice for testing.
    """

    process = subprocess.Popen(
        [
            "python",
            "-m",
            "uvicorn",
            "nocarz.api.main:app",
            "--host",
            HOST,
            "--port",
            str(PORT),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(5)

    try:
        yield process
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def test_connection(host: str = HOST, port: int = PORT) -> None:
    """
    Test the connection to the microservice.

    Args:
        host (str): The host where the microservice is running.
        port (int): The port where the microservice is running.
    """

    response = requests.get(f"http://{host}:{port}/")
    if response.text.strip() == "Server is running.":
        print("Server is running.")
    else:
        raise RuntimeError(f"Unexpected response: {response.text}")


def safe_str(value) -> str:
    """
    Convert a value to a string, handling NaN values.

    Args:
        value: The value to convert.

    Returns:
        str: The string representation of the value, or an empty string if the value is NaN.
    """

    return "" if pd.isna(value) else str(value)


def create_listing_request(row: pd.Series) -> dict:
    """
    Jsonify the listing request from a DataFrame row.

    Args:
        row (pd.Series): A row from a DataFrame containing listing information.

    Returns:
        dict: A dictionary representation of the listing request.
    """

    return {
        "id": int(row["id"]),
        "host_id": int(row["host_id"]),
        "name": safe_str(row.get("name")),
        "description": safe_str(row.get("description")),
        "neighbourhood": safe_str(row.get("neighbourhood")),
    }
