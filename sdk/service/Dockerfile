FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY service/requirements.txt ./service/requirements.txt
COPY service/casfly_service ./service/casfly_service

RUN pip install --no-cache-dir -r service/requirements.txt && \
    pip install --no-cache-dir .

EXPOSE 8000

CMD ["uvicorn", "casfly_service.app:app", "--app-dir", "/app/service", "--host", "0.0.0.0", "--port", "8000"]
