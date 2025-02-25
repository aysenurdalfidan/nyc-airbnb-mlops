from fastapi import FastAPI, HTTPException
import tensorflow as tf
import numpy as np
from pydantic import BaseModel

app = FastAPI()

# 📌 Modeli yükle
model = tf.keras.models.load_model("airbnb_price_model.h5", compile=False)

# 📌 Modelin beklediği giriş değişkeni sayısını al
EXPECTED_FEATURE_COUNT = model.input_shape[1]

class PredictionInput(BaseModel):
    features: list

@app.post("/predict")
def predict_price(input_data: PredictionInput):
    try:
        # 📌 Gelen feature sayısını kontrol et
        if len(input_data.features) != EXPECTED_FEATURE_COUNT:
            raise HTTPException(status_code=400, detail=f"Expected {EXPECTED_FEATURE_COUNT} features, but got {len(input_data.features)}")

        # 📌 Veriyi numpy array'e çevir
        input_array = np.array([input_data.features])

        # 📌 Modelle tahmin yap
        prediction = model.predict(input_array)

        return {"predicted_price": float(prediction[0][0])}
    
    except Exception as e:
        return {"error": str(e)}
