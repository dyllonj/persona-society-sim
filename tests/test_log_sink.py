from __future__ import annotations

from pathlib import Path

from schemas.logs import ActionLog
from storage.log_sink import LogSink


def test_log_sink_flush(tmp_path: Path):
    sink = LogSink(run_id="r1", db_url=None, parquet_dir=None)
    action = ActionLog(
        action_id="a1",
        run_id="r1",
        tick=0,
        agent_id="agent-1",
        action_type="talk",
        params={"topic": "test"},
        outcome="success",
        info={"utterance": "hi"},
    )
    sink.log_action(action)
    assert sink.action_buffer
    sink.flush(tick=0)
    assert not sink.action_buffer
