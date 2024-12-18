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
    && rm -rf /var/lib/apt/lists/*

# Install Poetry and add to PATH
ENV POETRY_HOME=/opt/poetry
ENV PATH="/opt/poetry/bin:$PATH"
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    poetry config virtualenvs.create false

# Copy project files
COPY pyproject.toml poetry.lock ./
COPY app ./app/

# Install dependencies
RUN poetry install --no-interaction --no-ansi

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Command will be specified in docker-compose.yml
CMD ["celery", "-A", "app.tasks", "worker", "--loglevel=info"]
