"""Application entrypoint for the Samisk transcription service."""
from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.types import ASGIApp, Receive, Scope, Send

from .routes.transcription import router as transcription_router

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


class ForwardedPrefixMiddleware:
    """Honor X-Forwarded-Prefix for reverse proxies (e.g. Apache/Nginx subpaths)."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] in ("http", "websocket"):
            headers = dict(scope.get("headers", []))
            prefix = headers.get(b"x-forwarded-prefix")
            if prefix:
                prefix_str = "/" + prefix.decode("latin1").strip("/")
                scope["root_path"] = prefix_str
                # Ensure scope["path"] matches scope["root_path"] prefix so Starlette Mount/StaticFiles
                # and sub-routers calculate correct relative lookup paths.
                if not scope["path"].startswith(prefix_str):
                    scope["path"] = prefix_str + scope["path"]
                    scope["raw_path"] = scope["path"].encode("latin1")
        await self.app(scope, receive, send)


app.add_middleware(ForwardedPrefixMiddleware)


@app.api_route("/health", methods=["GET", "HEAD"])
async def health() -> dict[str, str]:
    """Health check endpoint for reverse proxies and monitors."""
    return {"status": "ok"}


app.include_router(transcription_router)

static_directory = Path(__file__).resolve().parent.parent / "static"
app.mount("/static", StaticFiles(directory=static_directory), name="static")


@app.api_route("/", methods=["GET", "HEAD"])
@app.api_route("/index.html", methods=["GET", "HEAD"])
async def index(request: Request) -> Response:
    """Serve web application with dynamically injected base href for proxy subpaths."""
    index_path = static_directory / "index.html"
    prefix = request.headers.get("x-forwarded-prefix") or request.scope.get("root_path") or ""
    prefix = prefix.strip("/")
    base_href = f"/{prefix}/" if prefix else "./"

    content = index_path.read_text(encoding="utf-8")
    content = content.replace("<head>", f'<head>\n  <base href="{base_href}">', 1)
    return HTMLResponse(content=content)


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
