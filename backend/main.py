import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import router

from backend.database.database import (
    Base,
    engine
)

from backend.database import models


Base.metadata.create_all(
    bind=engine
)


app = FastAPI(
    title="AI Sports Coach",
    description=(
        "AI-powered sports talent assessment "
        "and performance improvement system"
    ),
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.getenv(
            "CORS_ORIGINS",
            "https://arijitpal01.github.io,http://localhost:5000,http://127.0.0.1:5000",
        ).split(",")
        if origin.strip()
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "message": "AI Sports Coach Backend is running!",
        "status": "success"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


app.include_router(
    router,
    prefix="/api"
)