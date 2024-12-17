from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import psycopg
from .routers import tasks

app = FastAPI()

# Disable CORS. Do not remove this for full-stack development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Include task management router
app.include_router(tasks.router, prefix="/api/v1")

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}
