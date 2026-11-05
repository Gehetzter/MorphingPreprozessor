
FROM python:3.11-slim

ARG BUILD_NUMBER=dev
ARG GIT_SHA=unknown
ARG SEGMENTATION_MODEL=birefnet-general

ENV BUILD_NUMBER=${BUILD_NUMBER} \
    GIT_SHA=${GIT_SHA} \
    SEGMENTATION_MODEL=${SEGMENTATION_MODEL} \ PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    U2NET_HOME=/models \
    HOME=/app

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgomp1 \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Modell bereits beim Image-Build herunterladen
RUN mkdir -p /models && \
    python -c "import os; from rembg import new_session; new_session(os.environ['SEGMENTATION_MODEL'])"

# Python-Anwendung kopieren
COPY app/ /app/

RUN mkdir -p /data/input /data/output

EXPOSE 8080

CMD ["streamlit", "run", "main.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8080", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
