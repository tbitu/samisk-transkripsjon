"""Utilities for producing PDF exports of transcripts.

This module will attempt to use a Unicode-capable TTF font (DejaVu Sans)
located at `static/fonts/DejaVuSans.ttf`. If the file is missing the code
tries to download it automatically. If download fails, a safe fallback to
the core Helvetica font is used and unsupported glyphs are replaced.
"""
from __future__ import annotations

import logging
import urllib.request
from typing import Final
from pathlib import Path

from fpdf import FPDF
from fpdf.errors import FPDFException

LOGGER = logging.getLogger(__name__)

FONT_DIR: Final[Path] = Path("static/fonts")
FONT_PATH: Final[Path] = FONT_DIR / "DejaVuSans.ttf"
# Try multiple reliable raw URLs for DejaVu Sans. The first one that
# succeeds will be used.
DEJAVU_URLS = [
    "https://github.com/dejavu-fonts/dejavu-fonts/raw/master/ttf/DejaVuSans.ttf",
    "https://raw.githubusercontent.com/dejavu-fonts/dejavu-fonts/master/ttf/DejaVuSans.ttf",
    "https://github.com/dejavu-fonts/dejavu-fonts/blob/master/ttf/DejaVuSans.ttf?raw=true",
]


def _ensure_dejavu_font(timeout: int = 30) -> bool:
    """Ensure the DejaVu Sans TTF exists under `static/fonts`.

    Returns True if the font is available (already present or successfully
    downloaded). Returns False on any error, leaving the caller to fall back
    to a core PDF font.
    """
    try:
        FONT_DIR.mkdir(parents=True, exist_ok=True)
        if FONT_PATH.exists():
            return True
        LOGGER.info("Attempting to download DejaVuSans.ttf to %s", FONT_PATH)
        last_exc: Exception | None = None
        for url in DEJAVU_URLS:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "python-urllib/3"})
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    data = resp.read()
                if data:
                    FONT_PATH.write_bytes(data)
                    LOGGER.info("Successfully downloaded DejaVuSans.ttf from %s", url)
                    return True
            except Exception as exc:  # try next URL
                last_exc = exc
                LOGGER.debug("Download attempt from %s failed: %s", url, exc)

        # If we reach here we tried all URLs and failed.
        if last_exc is not None:
            raise last_exc
    except Exception as exc:  # pragma: no cover - network/IO guards
        LOGGER.warning("Could not ensure DejaVuSans.ttf: %s", exc)
        return False


def build_transcript_pdf(text: str) -> bytes:
    """Render the provided text into a simple multi-page PDF.

    The function will try to load DejaVu Sans (Unicode-capable). If not
    available it falls back to Helvetica and replaces unsupported glyphs.
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_title("Transcription")
    pdf.set_author("Samisk Transkribering")
    pdf.add_page()

    font_loaded = False
    if _ensure_dejavu_font():
        try:
            pdf.add_font("DejaVu", fname=str(FONT_PATH), uni=True)
            pdf.set_font("DejaVu", size=12)
            font_loaded = True
        except Exception as exc:  # pragma: no cover - runtime font errors
            LOGGER.warning("Failed to register DejaVu font: %s", exc)

    if not font_loaded:
        # Fall back to core font, replacing unsupported glyphs.
        LOGGER.info("Using fallback core font (Helvetica); some glyphs may be lost")
        pdf.set_font("Helvetica", size=12)
        text = text.encode("latin-1", "replace").decode("latin-1")

    for line in text.splitlines() or [text]:
        _write_line_with_wrap(pdf, line)

    output = pdf.output(dest="S")
    if isinstance(output, str):
        output = output.encode("latin-1")
    return output


def _write_line_with_wrap(pdf: FPDF, line: str) -> None:
    """Write a line to the PDF, wrapping overly long tokens with hyphenation."""
    safe_line = line if line else " "
    max_width = pdf.w - pdf.l_margin - pdf.r_margin
    max_text_width = max(max_width - 2 * pdf.c_margin, 1)
    pdf.set_x(pdf.l_margin)
    try:
        pdf.multi_cell(max_width, 8, safe_line)
    except FPDFException:
        for wrapped in _wrap_line(pdf, safe_line, max_text_width):
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(max_width, 8, wrapped)


def _wrap_line(pdf: FPDF, line: str, max_text_width: float) -> list[str]:
    """Break a line into chunks that fit the available width."""
    chunks: list[str] = []
    current = ""

    for word in line.split(" "):
        if not word:
            candidate = (current + " ") if current else " "
            if pdf.get_string_width(candidate) <= max_text_width:
                current = candidate
            else:
                if current:
                    chunks.append(current.rstrip())
                current = " "
            continue

        for segment in _split_word(pdf, word, max_text_width):
            if not current:
                current = segment
                continue

            candidate = f"{current} {segment}" if not current.endswith(" ") else current + segment
            if pdf.get_string_width(candidate) <= max_text_width:
                current = candidate
            else:
                chunks.append(current.rstrip())
                current = segment

    if current:
        chunks.append(current.rstrip())

    return chunks or [" "]


def _split_word(pdf: FPDF, word: str, max_text_width: float) -> list[str]:
    """Split a word so that each part fits within the usable text width."""
    if pdf.get_string_width(word) <= max_text_width:
        return [word]

    segments: list[str] = []
    remaining = word

    while remaining:
        best = None
        for length in range(len(remaining), 0, -1):
            chunk = remaining[:length]
            is_last = length == len(remaining)
            candidate = f"{chunk}-" if not is_last else chunk
            if pdf.get_string_width(candidate) <= max_text_width or length == 1:
                best = candidate if not is_last else chunk
                remaining = remaining[length:]
                break
        if best is None:
            # Fallback: take single character to avoid infinite loop
            best = remaining[0] + ("-" if len(remaining) > 1 else "")
            remaining = remaining[1:]

        segments.append(best)

    return segments
