FROM python:3.11-slim

# build-essential + curl: prophet's install compiles cmdstan from source via cmdstanpy.
# libgomp1: required at runtime by lightgbm (OpenMP).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -e .

COPY . .

EXPOSE 8000

CMD ["uvicorn", "epicast.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
