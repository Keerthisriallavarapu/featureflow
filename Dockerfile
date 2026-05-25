FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir -e ".[ml]"

COPY featureflow/ ./featureflow/

EXPOSE 8080 9100

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
  CMD curl -fsS http://localhost:8080/health || exit 1

CMD ["python", "-m", "featureflow.cli", "serve", "--host", "0.0.0.0", "--port", "8080"]
