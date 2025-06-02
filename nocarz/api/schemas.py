from pydantic import BaseModel


class ListingRequest(BaseModel):
    host_id: int
    property_type: str
    room_type: str
    bathrooms_text: str
    accommodates: int
    bathrooms: float
    bedrooms: int
    beds: int
    price: float


class ListingResponse(BaseModel):
    property_type: str
    room_type: str
    bathrooms_text: str
    accommodates: int
    bathrooms: int
    bedrooms: int
    beds: int
    price: float
