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
    overall_progress: float = 0.0
    current_step: str = "pending"
    step_progress: float = 0.0
    session_id: Optional[str] = None

    def as_dict(self, include_details: bool = True) -> Dict[str, object]:
        """
        Return job data as dictionary.
        
        Args:
            include_details: If False, only returns basic progress info (for non-owners)
        """
        if not include_details:
            # Minimal info for non-owner clients
            return {
                "job_id": self.job_id,
                "status": self.status,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
                "progress": {
                    "overall": self.overall_progress,
                    "step": self.step_progress,
                },
                "current_step": self.current_step,
            }
        
        return {
            "job_id": self.job_id,
            "status": self.status,
            "text": self.text,
            "segments": self.segments,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "progress": {
                "overall": self.overall_progress,
                "step": self.step_progress,
            },
            "current_step": self.current_step,
        }
    
    def estimate_time_remaining(self) -> Optional[float]:
        """
        Estimate remaining time in seconds based on progress.
        Returns None if not enough data to estimate.
        """
        if self.status not in ["processing", "queued", "pending"]:
            return None
        
        if self.overall_progress <= 0:
            return None
        
        elapsed = time.time() - self.created_at
        if elapsed <= 0:
            return None
        
        # Estimate total time based on progress so far
        estimated_total = elapsed / (self.overall_progress / 100.0)
        remaining = estimated_total - elapsed
        return max(0, remaining)


class TranscriptionManager:
    """Manages background workers for transcription jobs."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.jobs: Dict[str, TranscriptionJob] = {}
        self._queue: Queue[TranscriptionJob] = Queue()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()
        self._active_job_id: Optional[str] = None
        self._lock = threading.Lock()

    def submit(self, uploaded_path: Path, session_id: str) -> TranscriptionJob:
        job_id = uuid.uuid4().hex
        job_storage = self.workspace / f"{job_id}{uploaded_path.suffix}"
        shutil.move(str(uploaded_path), job_storage)

        job = TranscriptionJob(job_id=job_id, source_path=job_storage, session_id=session_id)
        self.jobs[job_id] = job
        self._queue.put(job)
        return job

    def get(self, job_id: str) -> Optional[TranscriptionJob]:
        return self.jobs.get(job_id)
    
    def get_active_job(self) -> Optional[TranscriptionJob]:
        """Get the currently processing job, if any."""
        with self._lock:
            if self._active_job_id:
                return self.jobs.get(self._active_job_id)
        return None
    
    def is_busy(self) -> bool:
        """Check if there's currently a job being processed."""
        return self.get_active_job() is not None

    def list_jobs(self) -> Dict[str, Dict[str, object]]:
        return {job_id: job.as_dict() for job_id, job in self.jobs.items()}

    def _worker_loop(self) -> None:
        # Step weights for overall progress calculation
        # Four distinct phases with equal weight for simplicity
        STEP_WEIGHTS = {
            "Speaker Diarization": 0.25,  # pyannote - 25%
            "VAD Processing": 0.25,       # vad - 25%
            "Transcription": 0.25,        # whisper - 25%
            "Punctuation": 0.25,          # stanza - 25%
        }
        
        while True:
            try:
                job = self._queue.get(timeout=0.1)
            except Empty:
                continue

            # Mark this job as active
            with self._lock:
                self._active_job_id = job.job_id

            job.status = "processing"
            job.current_step = "Starting"
            job.step_progress = 0.0
            job.overall_progress = 0.0
            job.updated_at = time.time()

            def _update_progress(stage: str, fraction: float, info: Optional[Dict[str, object]] = None) -> None:
                bounded = max(0.0, min(1.0, fraction))
                details = info or {}
                
                from logging import getLogger
                progress_logger = getLogger(__name__)
                progress_logger.debug(
                    "TranscriptionService: stage=%s, fraction=%.3f, phase=%s",
                    stage, fraction, details.get("phase", "")
                )
                
                # Map stages/phases to user-friendly step names
                if stage == "diarization":
                    # Model loading is instantaneous and doesn't report progress
                    # All diarization progress is from actual inference
                    raw_diarization = max(0.0, min(1.0, float(details.get("diarization_fraction", bounded))))
                    step_name = "Speaker Diarization"
                    job.step_progress = round(raw_diarization * 100.0, 2)
                    job.overall_progress = round(raw_diarization * STEP_WEIGHTS["Speaker Diarization"] * 100.0, 2)

                    job.status = "processing"
                    job.current_step = step_name

                elif stage == "vad":
                    # After completing Speaker Diarization (25%), we're in VAD (25%)
                    base_progress = STEP_WEIGHTS["Speaker Diarization"]
                    step_name = "VAD Processing"
                    job.step_progress = round(bounded * 100.0, 2)
                    job.overall_progress = round((base_progress + bounded * STEP_WEIGHTS["VAD Processing"]) * 100.0, 2)
                    job.current_step = step_name
                        
                elif stage == "transcription":
                    # Phase 3: Whisper Transcription (25% of total)
                    job.status = "processing"
                    job.current_step = "Transcription"
                    job.step_progress = round(bounded * 100.0, 2)
                    
                    # We've completed Pyannote (25%) + VAD (25%), now doing Whisper (25%)
                    base_progress = STEP_WEIGHTS["Speaker Diarization"] + STEP_WEIGHTS["VAD Processing"]
                    job.overall_progress = round((base_progress + bounded * STEP_WEIGHTS["Transcription"]) * 100.0, 2)
                    
                elif stage == "punctuation":
                    # Phase 4: Stanza Punctuation (25% of total)
                    job.status = "processing"
                    job.current_step = "Punctuation"
                    job.step_progress = round(bounded * 100.0, 2)
                    
                    # We've completed Pyannote (25%) + VAD (25%) + Whisper (25%), now doing Stanza (25%)
                    base_progress = (STEP_WEIGHTS["Speaker Diarization"] + 
                                   STEP_WEIGHTS["VAD Processing"] + 
                                   STEP_WEIGHTS["Transcription"])
                    job.overall_progress = round((base_progress + bounded * STEP_WEIGHTS["Punctuation"]) * 100.0, 2)

                job.updated_at = time.time()

            try:
                result = transcribe(job.source_path, progress=_update_progress)
                job.text = result.get("text")
                job.segments = result.get("chunks") or result.get("segments")
                job.status = "completed"
                job.current_step = "Completed"
                job.step_progress = 100.0
                job.overall_progress = 100.0
            except Exception as exc:  # pragma: no cover - defensive logging hook
                job.error = str(exc)
                job.status = "failed"
                job.current_step = "Failed"
            finally:
                job.updated_at = time.time()
                # Clear active job when done
                with self._lock:
                    self._active_job_id = None
                self._queue.task_done()


manager: Optional[TranscriptionManager] = None


def initialise_manager(storage_dir: Path) -> TranscriptionManager:
    global manager
    if manager is None:
        manager = TranscriptionManager(storage_dir)
    return manager
