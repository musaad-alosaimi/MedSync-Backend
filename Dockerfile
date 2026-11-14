FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements_2.txt ./
COPY server/requirements.txt ./server/
RUN pip install --no-cache-dir -r server/requirements.txt
COPY *.py *.xlsx ./
COPY server ./server
CMD uvicorn server.app:app --host 0.0.0.0 --port ${PORT:-8000}
