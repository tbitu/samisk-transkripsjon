# Samisk Transkribering

A FastAPI-based web application that performs speech-to-text transcription using NBAiLab's Whisper large Northern Sámi model (`whisper-large-sme`).

## Features

- Upload audio or video files for transcription
- On-device decoding with GPU acceleration when available (CUDA)
- Streaming-ready architecture with background job queue
- Web frontend to review, edit, and download transcripts as PDF

## Requirements

- Python 3.10+
- CUDA-capable GPU with drivers and CUDA toolkit installed (for GPU acceleration)
- FFmpeg available on the system path
- Sufficient disk space and memory for the large Whisper model (~3 GB)
- Optional: Download `DejaVuSans.ttf` (or any Unicode-capable font) into `static/fonts` for proper PDF export of Sámi characters

PDF font handling
-----------------

The application will attempt to use `static/fonts/DejaVuSans.ttf` for PDF
exports. If that file is missing the app will try to download DejaVu Sans
automatically on-demand. If the download fails (offline environment or
network blocked) the PDF exporter falls back to a core PDF font (Helvetica)
and replaces unsupported glyphs.

You can manually install a Unicode-capable font by placing a TTF file at
`static/fonts/DejaVuSans.ttf` (or edit `app/utils/pdf.py` to point to a
different font).

## Getting Started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

If you want GPU-accelerated PyTorch with CUDA 12.9 in the same venv, run:

```bash
# in the already-activated virtualenv
python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url https://download.pytorch.org/whl/cu129 \
	--extra-index-url https://download.pytorch.org/whl/torch_stable.html \
	torch torchaudio --upgrade
```

To run the API server:

```bash
uvicorn app.main:app --reload
```

When running inside the project's virtual environment, it's often more reliable to run the server using the venv's Python interpreter so the correct site-packages are used. The following is recommended and is what worked in the repo environment:

```bash
python -m uvicorn app.main:app --reload
```

Then open `http://localhost:8000` in your browser.

## Testing

```bash
pytest
```

## Notes

- The first request will trigger a model download from Hugging Face (requires an internet connection and a configured token if the model is gated).
- To force GPU usage, ensure CUDA is available. The application automatically falls back to CPU if CUDA is not detected.
- Speaker diarization relies on the `pyannote/speaker-diarization-3.1` pipeline. Export `PYANNOTE_AUTH_TOKEN`, `HUGGINGFACE_TOKEN`, or `HF_TOKEN` with a valid Hugging Face access token before starting the API server to enable speaker labels.

Diarization requirement
----------------------

Speaker diarization is mandatory for the application. The server will refuse to start unless a valid Hugging Face token is provided for the `pyannote` diarization pipeline.

- **Provide token via environment:** set one of `PYANNOTE_AUTH_TOKEN`, `HUGGINGFACE_TOKEN`, or `HF_TOKEN` in your environment before starting the server.
- **Or provide token via file:** place the token (plain text) in a file named `hf_token` at the project root.

If no token is available the server will raise an error on startup. The application does not silently fall back to silence-based segmentation anymore; diarization must succeed and produce segments for transcription to proceed.

## Run

Preferred ways to run the server from the project root:

- Use the venv's Python to ensure the correct site-packages are used (recommended):

```bash
# when your virtualenv is activated
python -m uvicorn app.main:app --reload
```

- Or use the convenience script `run.sh` which will create `.venv` (if missing), install `requirements.txt`,
	and start the server. To run it:

```bash
chmod +x run.sh
./run.sh
```

- If you want to install CUDA 12.9 torch automatically as part of the script, set `TORCH_CU=129`:

```bash
TORCH_CU=129 ./run.sh
```

Troubleshooting note:

- If you see `ModuleNotFoundError: No module named 'uvicorn'` when running `uvicorn ...`, it's likely
	you're invoking a globally-installed `uvicorn` launcher while your venv does not have `uvicorn` installed.
	Prefer `python -m uvicorn ...` or install `uvicorn` into the venv with `python -m pip install "uvicorn[standard]"`.
