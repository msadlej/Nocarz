from fastapi import FastAPI, Response
import uvicorn

from nocarz.api.utils import get_base_prediction
from nocarz.api.schemas import ListingResponse
from nocarz.config import HOST, PORT


app = FastAPI()


@app.get("/")
def read_root():
    return Response("Server is running.")


@app.post("/predict/base", response_model=ListingResponse)
async def predict_base(host_id: int) -> ListingResponse:
    return get_base_prediction(host_id)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=True,
    )
