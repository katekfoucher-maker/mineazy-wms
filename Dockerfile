FROM python:3.12-slim

# tesseract-ocr: the OCR engine binary for photo/scan document uploads (the
# pytesseract pip package is only a thin wrapper around it, see requirements.txt).
# libgomp1: OpenMP runtime xgboost/lightgbm need at import time, not present on
# the slim base image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "wms.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
