from pathlib import Path

from orchestrator.cli import load_config


def test_load_config_merges_template(tmp_path: Path):
    template = tmp_path / "base.yaml"
    template.write_text(
        """
max_events_per_tick: 12
inference:
  temperature: 0.5
  top_p: 0.95
  max_new_tokens: 64
        """.strip()
    )

    child = tmp_path / "child.yaml"
    child.write_text(
        """
template: base.yaml
run_id: sample
inference:
  max_new_tokens: 20
        """.strip()
    )

    config = load_config(child)

    assert config["max_events_per_tick"] == 12
    assert config["inference"]["temperature"] == 0.5
    assert config["inference"]["top_p"] == 0.95
    assert config["inference"]["max_new_tokens"] == 20
    assert config["run_id"] == "sample"
