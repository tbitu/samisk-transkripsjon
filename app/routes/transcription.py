"""HTTP endpoints for transcription workflow."""
from __future__ import annotations

import mimetypes
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse

from ..services.transcription_service import TranscriptionManager, initialise_manager
from ..utils.pdf import build_transcript_pdf

router = APIRouter(prefix="/api/transcriptions", tags=["transcriptions"])

UPLOAD_DIR = Path("storage") / "incoming"
JOBS_DIR = Path("storage") / "jobs"
EXPORT_DIR = Path("storage") / "exports"


def get_manager() -> TranscriptionManager:
    return initialise_manager(JOBS_DIR)


@router.post("", status_code=status.HTTP_202_ACCEPTED)
async def create_transcription(
    file: UploadFile,
    manager: TranscriptionManager = Depends(get_manager),
) -> JSONResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    suffix = Path(file.filename).suffix.lower()
    if not suffix:
        raise HTTPException(status_code=400, detail="File must include an extension")

    upload_path = UPLOAD_DIR
    upload_path.mkdir(parents=True, exist_ok=True)
    temp_path = upload_path / f"{uuid.uuid4().hex}{suffix}"

    with temp_path.open("wb") as fh:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)
    await file.close()

    job = manager.submit(temp_path)
    return JSONResponse({"job_id": job.job_id, "status": job.status})


@router.get("/{job_id}")
async def get_transcription(job_id: str, manager: TranscriptionManager = Depends(get_manager)):
    job = manager.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.as_dict()


@router.post("/{job_id}/finalise")
async def finalise_transcription(
    job_id: str,
    payload: dict,
    manager: TranscriptionManager = Depends(get_manager),
):
    job = manager.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")

    text = payload.get("text")
    if not isinstance(text, str):
        raise HTTPException(status_code=400, detail="Missing corrected text")

    job.text = text
    job.updated_at = time.time()
    return job.as_dict()


@router.get("/{job_id}/download")
async def download_transcription(job_id: str, manager: TranscriptionManager = Depends(get_manager)):
    job = manager.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "completed" or not job.text:
        raise HTTPException(status_code=400, detail="Transcript not ready")

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = EXPORT_DIR / f"{job.job_id}.pdf"

    pdf_bytes = build_transcript_pdf(job.text)
    pdf_path.write_bytes(pdf_bytes)

    media_type, _ = mimetypes.guess_type(pdf_path.name)
    return FileResponse(
        path=pdf_path,
        media_type=media_type or "application/pdf",
        filename=f"transcript-{job.job_id}.pdf",
    )
