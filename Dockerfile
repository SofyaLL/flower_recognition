FROM continuumio/miniconda3:24.1.2-0

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

RUN pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install faiss-cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ .
COPY data/ ./data

CMD ["python", "app.py"]