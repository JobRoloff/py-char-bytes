from collections.abc import AsyncIterable
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.services.readability import extract_readability_metrics

app = FastAPI()

@app.get("/")
def read_root():
    print("hello!")
    return "SOLID!"

class TextPayload(BaseModel):
  text: str

@app.post("/nlp/analyze")
async def get_analysis(payload: TextPayload):
    return extract_readability_metrics(payload.text)