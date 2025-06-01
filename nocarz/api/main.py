from fastapi import FastAPI, Response
import uvicorn

from nocarz.config import HOST, PORT


app = FastAPI()


@app.get("/")
def read_root():
    return Response("Server is running.")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=True,
    )
