from pydantic import BaseModel, Field


class ListingRequest(BaseModel):
    host_id: int
    name: str
    description: str
    neighbourhood: str = Field(default="")


class ListingResponse(BaseModel):
    property_type: str
    room_type: str
    bathrooms_text: str
    accommodates: int
    bathrooms: int
    bedrooms: int
    beds: int
    price: float
