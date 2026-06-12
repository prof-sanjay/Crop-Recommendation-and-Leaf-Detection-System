FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download VGG16 ImageNet weights at build time so the first
# request doesn't have to download ~530 MB and time out.
RUN python -c "from tensorflow.keras.applications import VGG16; VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))"

COPY . .

RUN mkdir -p static/uploads

EXPOSE 10000

CMD gunicorn --bind 0.0.0.0:${PORT:-10000} --timeout 300 --workers 1 app:app
