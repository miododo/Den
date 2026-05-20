FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libgomp1 tesseract-ocr tesseract-ocr-chi-sim \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-web.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p integrated_runtime/uploads integrated_runtime/exports

ENV PYTHONUTF8=1
ENV PYTHONIOENCODING=utf-8
ENV FLAGS_use_mkldnn=0
ENV OCR_MAX_PAGES=all
ENV OCR_FORCE_LOCAL=0
# Build v20260520 — AI config panel fix
ENV PORT=8080

EXPOSE 8080

CMD python -m uvicorn integrated_test_app:app --host 0.0.0.0 --port ${PORT:-8080}
