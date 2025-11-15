"""Tests for utils.sanitize helpers."""

import logging

import pytest

from utils.sanitize import sanitize_agent_output


def test_collapse_repeated_sentences_shrinks_to_two() -> None:
    text = "Hello. Hello. Hello."
    assert sanitize_agent_output(text) == "Hello. Hello."


def test_collapse_repeated_sentences_logs(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    text = "Hi! Hi! Hi! Hi!"
    sanitize_agent_output(text)
    assert any(
        record.levelno == logging.INFO
        and "Collapsed repeated sentences" in record.getMessage()
        for record in caplog.records
    )
