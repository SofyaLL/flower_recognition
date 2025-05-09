# flower_recognition

## Запуск

```bash
docker build -t flower_search .
docker run -p 8100:8100 flower_search
```

```python
import base64
import cv2
import requests

img_path = "test.jpg"
api_url = "http://127.0.0.1:8100/api/predict"

frame = cv2.imread(img_path)
_, img_encoded = cv2.imencode(".jpg", frame)
img_base64 = base64.b64encode(img_encoded).decode("utf-8")


payload = {
    "image": img_base64,
}
response = requests.post(api_url, json=payload)
result = response.json()['result'] # словарь вида похожее изображение: показатель сходства 

```

## Структура

- `dataset.ipynb` ⏤ ноутбук с разделением датасета на библиотеку изображений и тестовую часть для поиска с примерами работы модели 
- `test.ipynb` ⏤ проверка работы API


## Пояснения к коду

- Я использовала модель DinoV2 для получения векторов изображений. Мне не хотелось обучать отдельную модель, так как тестовое задание на 12 часов.

- Dinov2 может быть тяжела или не удобна для продакшена (из-за использованя pytorch как минимум).

- Я не добавляла веб-интерфейс в веб-сервис.

- Также весь код рассчитан на запуск на CPU.

- На данный момент веса подгружаются во время запуска докер-контейнера. В дальнейшем это можно было бы исправить

- В результатах используется косинусное сходство ⏤ чем оно больше, тем более похожие изображения.
