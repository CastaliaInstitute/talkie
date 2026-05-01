# Castalia CPU Cloud Run: circa-1931 chat UI, /health, optional proxy to GPU (/v1/chat/completions).
FROM python:3.12-slim-bookworm

WORKDIR /app
ENV PYTHONUNBUFFERED=1
COPY web/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY web/ .

ENV PORT=8080
CMD exec uvicorn main:app --host 0.0.0.0 --port "${PORT}"
