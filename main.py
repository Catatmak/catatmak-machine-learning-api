# import library yang bakal dipake
from fastapi import FastAPI
from tensorflow import keras
import numpy as np
from PIL import Image
import uvicorn

app = FastAPI()

# load models
model_path = "/path/models"
model = keras.models.load_model(model_path)

@app.get("/", status_code=200)
async def root():
    return {
        "code": 200,
        "success": "true",
        "message": "Service OK"
    }

# tambahkan endpoint pada aplikasi FastAPI untuk menerima file gambar dan melakukan prediksi
@app.post("/categorize", status_code=201)
async def categorize(payload):
    return

# jalankan server FastAPI
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8080)
