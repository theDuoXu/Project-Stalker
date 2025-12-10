from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

class PollutionDetection(BaseModel):
    river_km: float
    concentration: float
    simulation_id: str

# Endpoint de salud para Docker Healthcheck
@app.get("/health")
def health():
    return {"status": "ok"}

# Endpoint Dummy para la simulación
@app.post("/predict/source")
def predict(detection: PollutionDetection):
    # Simulamos que siempre encontramos un culpable cerca
    return [
        {
            "id": "EXP-999",
            "name": "Industrias Dummy S.L. (MOCK)",
            "distance_to_event": 0.5,
            "activity": "Simulacro",
            "risk_score": 0.99
        }
    ]