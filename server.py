from fastapi import FastAPI, Request
from typing import List, Dict
from pydantic import BaseModel
import uvicorn
from evaluate import predict

app = FastAPI()

class PredictionInput(BaseModel):
    records: List[Dict]
    

@app.post("/predict")
async def predict_api(input_data: PredictionInput):
    prediction_result = predict(input_data.records)
    return {"prediction": str(prediction_result.to_dict())}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)