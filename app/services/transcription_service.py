"""Background transcription job management."""
from __future__ import annotations

import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue, Empty
from typing import Dict, Optional

from .model_service import transcribe


@dataclass
class TranscriptionJob:
    job_id: str
    source_path: Path
    status: str = "pending"
    text: Optional[str] = None
    segments: Optional[list] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    diarization_progress: float = 0.0
    transcription_progress: float = 0.0
    overall_progress: float = 0.0
    current_task: str = "pending"

    def as_dict(self) -> Dict[str, object]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "text": self.text,
            "segments": self.segments,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "progress": {
                "annotation": self.diarization_progress,
                "transcription": self.transcription_progress,
                "overall": self.overall_progress,
            },
            "current_task": self.current_task,
        }


class TranscriptionManager:
    """Manages background workers for transcription jobs."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.jobs: Dict[str, TranscriptionJob] = {}
        self._queue: Queue[TranscriptionJob] = Queue()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def submit(self, uploaded_path: Path) -> TranscriptionJob:
        job_id = uuid.uuid4().hex
        job_storage = self.workspace / f"{job_id}{uploaded_path.suffix}"
        shutil.move(str(uploaded_path), job_storage)

        job = TranscriptionJob(job_id=job_id, source_path=job_storage)
        self.jobs[job_id] = job
        self._queue.put(job)
        return job

    def get(self, job_id: str) -> Optional[TranscriptionJob]:
        return self.jobs.get(job_id)

    def list_jobs(self) -> Dict[str, Dict[str, object]]:
        return {job_id: job.as_dict() for job_id, job in self.jobs.items()}

    def _worker_loop(self) -> None:
        while True:
            try:
                job = self._queue.get(timeout=0.1)
            except Empty:
                continue

            job.status = "diarizing"
            job.current_task = "annotation"
            job.diarization_progress = 0.0
            job.transcription_progress = 0.0
            job.overall_progress = 0.0
            job.updated_at = time.time()

            def _update_progress(stage: str, fraction: float, info: Optional[Dict[str, object]] = None) -> None:
                bounded = max(0.0, min(1.0, fraction))
                if stage == "diarization":
                    job.status = "diarizing"
                    job.current_task = "annotation"
                    job.diarization_progress = round(bounded * 100.0, 2)
                elif stage == "transcription":
                    job.status = "transcribing"
                    job.current_task = "transcription"
                    job.transcription_progress = round(bounded * 100.0, 2)

                job.overall_progress = round(
                    (job.diarization_progress + job.transcription_progress) / 2.0,
                    2,
                )
                job.updated_at = time.time()

            try:
                result = transcribe(job.source_path, progress=_update_progress)
                job.text = result.get("text")
                job.segments = result.get("chunks") or result.get("segments")
                job.status = "completed"
                job.current_task = "completed"
                job.diarization_progress = 100.0
                job.transcription_progress = 100.0
                job.overall_progress = 100.0
            except Exception as exc:  # pragma: no cover - defensive logging hook
                job.error = str(exc)
                job.status = "failed"
                job.current_task = "failed"
            finally:
                job.updated_at = time.time()
                self._queue.task_done()


manager: Optional[TranscriptionManager] = None


def initialise_manager(storage_dir: Path) -> TranscriptionManager:
    global manager
    if manager is None:
        manager = TranscriptionManager(storage_dir)
    return manager
