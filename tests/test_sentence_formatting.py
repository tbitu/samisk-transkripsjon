from __future__ import annotations

import pytest

from app.services.model_service import _format_sentence


@pytest.mark.parametrize(
    "raw, audio_flag, expected_suffix",
    [
        ("mii lea", False, "?"),
        ("det er bra", False, "."),
        ("kan du", False, "?"),
        ("this is fine", True, "?"),
        ("already capitalized!", False, "!"),
    ],
)
def test_format_sentence_punctuation(raw: str, audio_flag: bool, expected_suffix: str):
    formatted, is_question = _format_sentence(raw, audio_flag)

    assert formatted.endswith(expected_suffix)
    if expected_suffix == "?":
        assert is_question is True


def test_format_sentence_capitalization():
    formatted, _ = _format_sentence("hello world", False)
    assert formatted[0].isupper()


def test_format_sentence_handles_empty():
    formatted, is_question = _format_sentence("   ", False)
    assert formatted == ""
    assert is_question is False


def test_format_sentence_preserves_existing_punctuation():
    """Test that existing punctuation is preserved."""
    formatted, _ = _format_sentence("This is good!", False)
    assert formatted == "This is good!"


def test_format_sentence_with_comma():
    """Test that commas are preserved."""
    formatted, _ = _format_sentence("Hello, world", False)
    assert "," in formatted
    assert formatted.endswith(".")


def test_multiple_questions_in_sequence():
    """Test processing multiple questions."""
    questions = ["kan du", "hvor er", "mii lea"]
    for q in questions:
        formatted, is_question = _format_sentence(q, False)
        assert is_question is True
        assert formatted.endswith("?")
