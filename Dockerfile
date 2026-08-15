# Anistroph Dockerfile
# Python runtime with dependencies pre-installed.
# Application code and data are bind-mounted via docker-compose.yml.

FROM python:3.11-slim

# Install system dependencies required by XGBoost (libomp) and Polars
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy only the dependency manifest first for layer caching
COPY pyproject.toml .

# Install the package in editable mode (deps only; code is bind-mounted)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[dev]"

# Copy the rest of the application code (also bind-mounted in dev)
COPY . /app

# Expose the port used by Uvicorn
EXPOSE 9500
