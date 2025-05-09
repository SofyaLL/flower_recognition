from typing import Dict

from pydantic import BaseModel


class PredictParams(BaseModel):
    """
    Параметры для поиска изображений

    image: Base64 строка изображения
    """

    image: str


class GenericResponse(BaseModel):
    """Результат поиска в формате: Dict"""

    result: Dict
