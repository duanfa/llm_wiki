FROM node:22-bookworm-slim AS frontend

WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build:web

FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    LLM_WIKI_WEB_DIST=/app/dist \
    LLM_WIKI_DATA_ROOT=/data \
    LLM_WIKI_CORS_ORIGINS=*

WORKDIR /app
COPY backend/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY backend ./backend
COPY --from=frontend /app/dist ./dist

EXPOSE 8000
VOLUME ["/data"]

CMD ["python", "-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
