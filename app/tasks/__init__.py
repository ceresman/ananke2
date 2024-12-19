"""Task queue module for Ananke2."""

from celery import Celery
from ..config import Settings

settings = Settings()

# Configure Celery with Redis backend
celery_app = Celery(
    'ananke2',
    broker=f'redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/{settings.REDIS_DB}',
    backend=f'redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/{settings.REDIS_DB}',
    include=['app.tasks.document', 'app.tasks.workflow']
)

# Configure Celery settings
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_ignore_result=False,
    task_store_errors_even_if_ignored=True,
    worker_prefetch_multiplier=1,
    # Database configurations
    neo4j_uri=settings.NEO4J_URI,
    neo4j_user=settings.NEO4J_USER,
    neo4j_password=settings.NEO4J_PASSWORD,
    chroma_host=settings.CHROMA_HOST,
    chroma_port=settings.CHROMA_PORT,
    chroma_collection=settings.CHROMA_COLLECTION,
    mysql_host=settings.MYSQL_HOST,
    mysql_port=settings.MYSQL_PORT,
    mysql_user=settings.MYSQL_USER,
    mysql_password=settings.MYSQL_PASSWORD,
    mysql_database=settings.MYSQL_DATABASE
)

# Import tasks after celery app is configured
from . import document  # noqa
from . import workflow  # noqa
