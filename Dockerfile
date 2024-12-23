# Use Python 3.12 as base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    poppler-utils \
    libmagic1 \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Configure pip to use BFSU mirror
RUN pip config set global.index-url https://mirrors.bfsu.edu.cn/pypi/web/simple

# Copy requirements files
COPY requirements.txt requirements-dev.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt -r requirements-dev.txt

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Command will be specified in docker-compose.yml
CMD ["celery", "-A", "app.tasks", "worker", "--loglevel=info"]
