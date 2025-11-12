from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.routes import transcription
from app.services.transcription_service import TranscriptionJob


class DummyManager:
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self.jobs: dict[str, TranscriptionJob] = {}

    def submit(self, uploaded_path: Path) -> TranscriptionJob:
        job_id = "test-job"
        stored_path = self.workspace / f"{job_id}{uploaded_path.suffix}"
        stored_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(uploaded_path, stored_path)
        job = TranscriptionJob(
            job_id=job_id,
            source_path=stored_path,
            status="completed",
            text="stub transcription",
            segments=[{"text": "stub"}],
        )
        self.jobs[job_id] = job
        return job

    def get(self, job_id: str):
        return self.jobs.get(job_id)


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    uploads = tmp_path / "incoming"
    jobs = tmp_path / "jobs"
    exports = tmp_path / "exports"
    transcription.UPLOAD_DIR = uploads
    transcription.JOBS_DIR = jobs
    transcription.EXPORT_DIR = exports

    dummy_manager = DummyManager(jobs)
    app.dependency_overrides[transcription.get_manager] = lambda: dummy_manager

    yield TestClient(app)

    app.dependency_overrides.clear()


def test_transcription_flow(client: TestClient):
    files = {"file": ("sample.wav", b"fake-data", "audio/wav")}

    create_resp = client.post("/api/transcriptions", files=files)
    assert create_resp.status_code == 202
    payload = create_resp.json()
    assert payload["job_id"] == "test-job"

    status_resp = client.get(f"/api/transcriptions/{payload['job_id']}")
    assert status_resp.status_code == 200
    assert status_resp.json()["status"] == "completed"

    final_resp = client.post(
        f"/api/transcriptions/{payload['job_id']}/finalise",
        json={"text": "corrected text"},
    )
    assert final_resp.status_code == 200
    assert final_resp.json()["text"] == "corrected text"

    download_resp = client.get(f"/api/transcriptions/{payload['job_id']}/download")
    assert download_resp.status_code == 200
    assert download_resp.headers["content-type"].startswith("application/pdf")
    assert download_resp.content.startswith(b"%PDF")
