"""Task queue module for Ananke2."""

from celery import Celery
from ..config import Settings

settings = Settings()
celery_app = Celery(
    "ananke2",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND
)

# Optional configurations can be added here if needed
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
)
