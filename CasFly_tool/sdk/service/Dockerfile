FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Each container = one device. Pass identity via environment variables:
#   DEVICE_ID, LOOKUP_CSV, TPHG_DIR, FILTERED_PATIENTS_CSV,
#   DEVICE_IP, DEVICE_PORT, METRICS_DIR, MAX_DEPTH
#
# Example:
#   docker run --rm -p 8001:8000 \
#     -e DEVICE_ID=BPCuff \
#     -e LOOKUP_CSV=/data/All_Device_Lookup_with_Probabilities.csv \
#     -e TPHG_DIR=/data/tphg_cache/BPCuff \
#     -e FILTERED_PATIENTS_CSV=/data/filtered_patients.csv \
#     -e DEVICE_IP=0.0.0.0 \
#     -e DEVICE_PORT=5001 \
#     -v /your/data:/data \
#     casfly-service:0.1.0

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY service/requirements.txt ./service/requirements.txt
COPY service/casfly_service ./service/casfly_service

RUN pip install --no-cache-dir -r service/requirements.txt && \
    pip install --no-cache-dir .

EXPOSE 8000

# One CasFlyDevice per container. The device is initialised at runtime
# via POST /start — raw patient data stays inside the container volume.
CMD ["uvicorn", "casfly_service.app:app", "--app-dir", "/app/service", "--host", "0.0.0.0", "--port", "8000"]
