# Samisk Transkribering

A FastAPI-based web application that performs speech-to-text transcription using NBAiLab's Whisper large Northern Sámi model (`whisper-large-sme`).

## Features

- Upload audio or video files for transcription
- **4-stage intelligent processing pipeline:**
  1. **Speaker Diarization** (Pyannote) - identifies speakers and speaker changes
  2. **Voice Activity Detection** (Silero VAD) - refines speaker turns into precise speech segments
  3. **Speech Recognition** (Whisper) - transcribes individual segments
  4. **Punctuation Restoration** (Stanza NLP) - adds proper punctuation and formatting
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

## Processing Pipeline

The transcription system uses a sophisticated 4-stage pipeline:

### 1. Speaker Diarization (Pyannote) - 25% of processing
- **Purpose:** Identifies who is speaking and when speaker changes occur
- **Model:** `pyannote/speaker-diarization-3.1`
- **Output:** Speaker turns with timestamps (e.g., "Speaker A: 0.0s-5.2s, Speaker B: 5.2s-10.1s")
- **GPU Memory:** Models are moved to CPU after this stage to free GPU for VAD

### 2. Voice Activity Detection (Silero VAD) - 25% of processing
- **Purpose:** Refines speaker turns into precise sentence-level speech segments
- **Model:** Silero VAD (via PyTorch Hub)
- **How it works:**
  - Loads the full audio file once (optimization)
  - For each speaker turn, runs VAD to detect actual speech
  - Merges short adjacent segments within the same speaker
  - Enforces minimum (1.2s) and maximum (20s) segment durations
  - Filters out silence, pauses, and non-speech audio
- **Output:** Precise speech segments with speaker labels
- **GPU Memory:** VAD model is cleared from GPU after this stage

### 3. Speech Recognition (Whisper) - 25% of processing
- **Purpose:** Transcribes each speech segment to text
- **Model:** `NbAiLab/whisper-large-sme` (Northern Sámi)
- **How it works:**
  - Processes sentence-level audio chunks (not entire speaker turns)
  - Uses word-level timestamps for sub-sentence alignment
  - Batches chunks for efficiency (batch size 4 on GPU, 1 on CPU)
  - Returns raw transcription without proper punctuation
- **GPU Memory:** Whisper is cleared from GPU after this stage to make room for Stanza

### 4. Punctuation Restoration (Stanza NLP) - 25% of processing
- **Purpose:** Adds proper punctuation, capitalization, and formatting
- **Model:** Stanza NLP (Norwegian pipeline used as fallback for Sámi)
- **How it works:**
  - Uses linguistic analysis (POS tagging, lemmatization)
  - Identifies question structures (question words, inverted verb-subject)
  - Adds periods, commas, question marks appropriately
  - Can split Whisper output into multiple sentences using word timestamps
- **Question Detection:** Combines audio pitch analysis and linguistic features
- **Output:** Properly formatted, punctuated transcription with speaker labels

### Pipeline Flow

```
Audio File → [Pyannote] → Speaker Turns → [VAD] → Speech Segments 
  → [Whisper] → Raw Text → [Stanza] → Formatted Transcript
```

**Key Optimizations:**
- Sequential GPU memory management: Each model is offloaded before loading the next
- Audio is loaded once and reused across VAD operations
- Parallel processing of audio chunks where possible
- Word-level timestamp tracking for precise segment alignment

## Notes

- The first request will trigger model downloads from Hugging Face (requires internet connection)
- To force GPU usage, ensure CUDA is available. The application automatically falls back to CPU if CUDA is not detected.
- GPU memory is managed carefully: models are sequentially loaded, used, and cleared to avoid OOM errors

### Diarization Requirement

Speaker diarization is **mandatory** for the application. The server will refuse to start unless a valid Hugging Face token is provided for the `pyannote` diarization pipeline.

- **Provide token via environment:** set one of `PYANNOTE_AUTH_TOKEN`, `HUGGINGFACE_TOKEN`, or `HF_TOKEN` in your environment before starting the server.
- **Or provide token via file:** place the token (plain text) in a file named `hf_token` at the project root.

If no token is available the server will raise an error on startup.

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
