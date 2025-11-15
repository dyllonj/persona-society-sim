from __future__ import annotations

import sys
import types

try:  # pragma: no cover - only hit in minimal test environments
    from orchestrator.console_logger import ConsoleLogger
except ModuleNotFoundError as exc:  # pragma: no cover
    if exc.name != "pydantic":
        raise

    schemas_module = types.ModuleType("schemas")
    sys.modules.setdefault("schemas", schemas_module)

    logs_module = types.ModuleType("schemas.logs")

    class _ActionLog:  # minimal shim for import-time type references
        def __init__(self):
            self.action_type = "talk"
            self.agent_id = "agent"
            self.outcome = "success"
            self.params = {}
            self.prompt_hash = None

    class _MsgLog:
        def __init__(self):
            self.from_agent = "agent"
            self.room_id = "room"
            self.content = ""
            self.steering_snapshot = {}
            self.tokens_in = 0
            self.tokens_out = 0

    logs_module.ActionLog = _ActionLog
    logs_module.MsgLog = _MsgLog
    sys.modules["schemas.logs"] = logs_module

    from orchestrator.console_logger import ConsoleLogger


def test_truncate_enabled_by_default():
    logger = ConsoleLogger(enabled=True, use_colors=False)
    text = "a" * 150

    truncated = logger._truncate(text, max_len=120)

    assert truncated.endswith("...")
    assert len(truncated) == 120


def test_full_messages_when_truncation_disabled():
    logger = ConsoleLogger(enabled=True, use_colors=False, truncate=False)
    text = "b" * 150

    full_text = logger._truncate(text, max_len=120)

    assert full_text == text
