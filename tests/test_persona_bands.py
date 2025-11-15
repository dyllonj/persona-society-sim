from metrics.persona_bands import BAND_METADATA, trait_band_key


def test_trait_band_key_uses_dominant_trait():
    persona = {"E": 0.1, "A": -2.0, "C": 0.0, "O": 0.0, "N": 0.0}
    key = trait_band_key(persona, {"A": 0.2})
    assert key == "A:low"
    assert "bands" in BAND_METADATA
