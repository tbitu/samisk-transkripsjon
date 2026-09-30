"""Application entrypoint for the Samisk transcription service."""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .routes.transcription import router as transcription_router
from pathlib import Path
import os
from fastapi import HTTPException

# Configure logging for the entire application
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

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


@app.middleware("http")
async def forwarded_prefix_middleware(request, call_next):
    """Honor X-Forwarded-Prefix so docs/OpenAPI work behind a reverse proxy path prefix."""
    prefix = request.headers.get("x-forwarded-prefix")
    if prefix:
        normalized = "/" + prefix.lstrip("/")
        request.scope["root_path"] = normalized.rstrip("/")
    return await call_next(request)


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint for reverse proxies and monitors."""
    return {"status": "ok"}


app.include_router(transcription_router)

static_directory = Path(__file__).resolve().parent.parent / "static"
app.mount("/static", StaticFiles(directory=static_directory), name="static")


@app.get("/")
async def index() -> FileResponse:
    index_path = static_directory / "index.html"
    return FileResponse(index_path)


@app.on_event("startup")
def ensure_diarization_token_available() -> None:
    """Ensure a Hugging Face token is available for pyannote diarization."""
    from .utils.sentence_segmenter import _get_hf_token

    token = _get_hf_token()
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["PYANNOTE_AUTH_TOKEN"] = token
        return

    raise RuntimeError(
        "Missing Hugging Face token required for speaker diarization. "
        "Store your token on disk in one of the following locations: "
        "1) .env file in project directory (HF_TOKEN=hf_...); "
        "2) ./hf_token file; "
        "3) /data/hf/token or ~/.cache/huggingface/token; "
        "or pass it via environment variable HF_TOKEN."
    )
