from pathlib import Path
import xml.etree.ElementTree as ET

from scripts.plot_distributed_clues import conditions, effect, guesses

BATCH = Path(__file__).resolve().parents[1] / "artifacts" / "local-v1" / "distributed-clues-003"


def _titles(path):
    root = ET.parse(path).getroot()
    ns = {"s": "http://www.w3.org/2000/svg"}
    return root.find("s:title", ns).text, root.find("s:desc", ns).text


def test_figures_render_valid_svg_with_findings_and_text_alternatives(tmp_path):
    effect([BATCH], tmp_path / "effect.svg")
    title, desc = _titles(tmp_path / "effect.svg")
    assert title == "Sharing evidence lifts accuracy from 11% to 100%"
    assert "3 task instances" in desc and "Uniform-guess reference" in desc

    guesses([BATCH], tmp_path / "guesses.svg")
    title, desc = _titles(tmp_path / "guesses.svg")
    assert title.startswith("Without the missing clues")
    assert "Uniform reference 25%" in desc

    conditions([f"Talk={BATCH}:communication", f"Alone={BATCH}:no_communication"],
               tmp_path / "conditions.svg", "Talking helps", "development batch")
    title, desc = _titles(tmp_path / "conditions.svg")
    assert title == "Talking helps" and "Alone: mean 11% over 3 instances" in desc
