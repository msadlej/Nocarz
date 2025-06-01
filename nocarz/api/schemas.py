from pydantic import BaseModel


class ListingResponse(BaseModel):
    accommodates: int
    bathrooms: int
    bedrooms: int
    beds: int
    price: float
    property_type: str
    room_type: str
    bathrooms_text: str
