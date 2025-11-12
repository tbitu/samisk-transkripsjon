"""Application entrypoint for the Samisk transcription service."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .routes.transcription import router as transcription_router

app = FastAPI(
    title="Samisk Transkribering",
    description="Speech-to-text service powered by NbAiLab Whisper large Northern Sámi model.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

app.include_router(transcription_router)

static_directory = Path(__file__).resolve().parent.parent / "static"
app.mount("/static", StaticFiles(directory=static_directory), name="static")


@app.get("/")
async def index() -> FileResponse:
    index_path = static_directory / "index.html"
    return FileResponse(index_path)
