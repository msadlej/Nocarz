from pydantic import BaseModel


class ListingResponse(BaseModel):
    property_type: str
    room_type: str
    bathrooms_text: str
    accommodates: int
    bathrooms: int
    bedrooms: int
    beds: int
    price: float
