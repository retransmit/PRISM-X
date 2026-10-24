"""A stand-in service for the HTTP latency measurement's subprocess mode:
answers at once and lingers afterwards, like the real service does."""

from __future__ import annotations

import time

from fastapi import BackgroundTasks, FastAPI
from pydantic import BaseModel

app = FastAPI()


class Row(BaseModel):
    i: int


def _linger(i: int) -> None:
    time.sleep(0.2)


@app.get("/api/v1/health")
def health() -> dict:
    return {"status": "ready"}


@app.post("/api/v1/predict")
def predict(body: Row, background: BackgroundTasks) -> dict:
    background.add_task(_linger, body.i)
    return {"flagged": body.i % 2 == 0}
