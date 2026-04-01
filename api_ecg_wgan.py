from fastapi import FastAPI
import numpy as np
from inference_ecg import load_model, generate_ecg

app = FastAPI()
model = load_model("checkpoints/G_epoch_180.pt")

@app.get("/health")
def health():
    return {"status": "OK"}

@app.post("/generate_ecg")
def generate(n_samples: int = 5):
    samples = generate_ecg(model, n_samples)
    return {"samples": samples.tolist()}