"""Keep personality instruments and strict model-output parsing explicit."""
import json
import math


DIMENSION_SETS = {
    "society-big-five-v1": ("O", "C", "E", "A", "N"),
    "published-comparison-v1": (
        "O", "C", "E", "A", "N", "curiosity", "assertiveness", "warmth", "formality", "creativity"),
    "lindr-current-code-v1": (
        "O", "C", "E", "A", "N", "assertiveness", "ambition", "resilience", "integrity", "curiosity"),
}
# Dimension inventories only: these do not reproduce or validate the original scorers.


def strict_json(text):
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f"non-finite JSON value: {value}")

    def finite_float(value):
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError("JSON number overflows")
        return parsed

    return json.loads(text, object_pairs_hook=object_pairs,
                      parse_constant=reject_constant, parse_float=finite_float)


def validate_persona(values):
    if not isinstance(values, dict) or set(values) - set("OCEAN"):
        raise ValueError("persona coefficients must map Big Five letters to numbers")
    for value in values.values():
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not -1 <= value <= 1:
            raise ValueError("persona coefficients must be finite and within [-1, 1]")
    return dict(values)
