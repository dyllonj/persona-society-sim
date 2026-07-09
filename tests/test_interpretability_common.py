from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq

from interpretability.common import (
    load_prompts,
    parse_layers,
    read_inference_events,
    resolve_event_paths,
    sha256_json,
)


def test_common_loads_prompts_and_normalizes_parquet_events(tmp_path):
    prompt_path = tmp_path / "prompts.jsonl"
    prompt_path.write_text(
        '{"text":"first prompt"}\n{"prompt":"second prompt"}\n',
        encoding="utf-8",
    )
    assert load_prompts(prompt_path) == ["first prompt", "second prompt"]
    assert parse_layers("4,2,4") == [2, 4]

    event_path = tmp_path / "events" / "inference_t00001.parquet"
    event_path.parent.mkdir()
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "trace_id": "trace-1",
                    "input_ids": json.dumps([1, 2]),
                    "attention_mask": json.dumps([1, 1]),
                    "generated_ids": json.dumps([3]),
                    "effective_alphas": json.dumps({"E": 0.5}),
                    "steering_vector_ids": json.dumps({"E": "E-v1"}),
                    "steering_vector_hashes": json.dumps({"E": {"4": "abc"}}),
                }
            ]
        ),
        event_path,
    )

    paths = resolve_event_paths(event_path.parent)
    events = read_inference_events(paths)

    assert events[0]["input_ids"] == [1, 2]
    assert events[0]["effective_alphas"] == {"E": 0.5}
    assert sha256_json({"b": 2, "a": 1}) == sha256_json({"a": 1, "b": 2})
