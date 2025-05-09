import base64

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from loguru import logger

from contracts import GenericResponse, PredictParams
from predict import Searcher
from settings import UvicornSettings

app = FastAPI()
searcher = Searcher()


@app.post("/api/predict")
async def predict(params: PredictParams) -> GenericResponse:
    try:
        image_bytes = base64.b64decode(params.image)
        image_array = np.asarray(bytearray(image_bytes), dtype="uint8")
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        model_result = searcher.predict(image)
        logger.info(f"Результат: {model_result}")

        return GenericResponse(result=model_result)
    except Exception as e:
        logger.error(f"Произошла ошибка: {str(e)}")
        raise HTTPException(status_code=500, detail="Ошибка при обработке изображения.")


if __name__ == "__main__":
    uvicorn_settings = UvicornSettings()
    uvicorn.run("app:app", host=uvicorn_settings.host, port=uvicorn_settings.port, reload=True)
