FROM node:22-bookworm-slim AS frontend
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/index.html frontend/tsconfig.json frontend/vite.config.ts ./
COPY frontend/src/ src/
COPY frontend/public/ public/
RUN npm run build

FROM python:3.11-slim-bookworm
WORKDIR /app
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt
COPY backend/app/*.py backend/app/
COPY --from=frontend /build/dist/ frontend/dist/
CMD ["sh", "-c", "exec python -m uvicorn app.staging:app --app-dir backend --host 0.0.0.0 --port ${PORT:-8000}"]
