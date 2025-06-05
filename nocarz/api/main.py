from fastapi import FastAPI, Response, Form
import uvicorn

from nocarz.api.utils import get_base_prediction, get_advanced_prediction
from nocarz.api.schemas import ListingResponse, ListingRequest
from nocarz.config import HOST, PORT


app = FastAPI()


@app.get("/")
def read_root():
    return Response("Server is running.")


@app.post("/predict/base", response_model=ListingResponse)
async def predict_base(host_id: ListingRequest = Form(...)) -> ListingResponse:
    return get_base_prediction(host_id)


@app.post("/predict/advanced", response_model=ListingResponse)
async def predict_advanced(listing: ListingRequest = Form(...)) -> ListingResponse:
    return get_advanced_prediction(listing)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=True,
    )
