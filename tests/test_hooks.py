from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from steering.hooks import SteeringController  # noqa: E402


class DummyLayer(nn.Module):
    def forward(self, hidden_states, *args, **kwargs):  # noqa: D401
        return hidden_states + 0


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([DummyLayer() for _ in range(2)])

    def forward(self, inputs_embeds):
        x = inputs_embeds
        for layer in self.model.layers:
            x = layer(x)
        return x


def test_steering_controller_adds_vector():
    model = DummyModel()
    hidden = torch.zeros(1, 1, 4)
    controller = SteeringController(model, {"E": {0: torch.ones(4)}})
    controller.register()
    controller.set_alphas({"E": 1.0})
    out = model(inputs_embeds=hidden)
    controller.remove()
    assert torch.allclose(out, torch.ones_like(out))


def test_steering_controller_batched_mode():
    model = DummyModel()
    hidden = torch.zeros(2, 1, 4)
    controller = SteeringController(model, {"E": {0: torch.ones(4)}})
    controller.register()
    controller.set_batched_alphas([{"E": 1.0}, {"E": 2.0}])
    out = model(inputs_embeds=hidden)
    controller.clear_batched_alphas()
    controller.remove()
    expected = torch.tensor([[[1.0, 1.0, 1.0, 1.0]], [[2.0, 2.0, 2.0, 2.0]]])
    assert torch.allclose(out, expected)


def test_batched_matches_sequential_outputs():
    model = DummyModel()
    hidden = torch.zeros(2, 1, 4)
    vectors = {
        "E": {0: torch.ones(4)},
        "F": {1: torch.tensor([1.0, 2.0, 3.0, 4.0])},
    }
    controller = SteeringController(model, vectors)
    controller.register()

    batched_alphas = [{"E": 0.5, "F": 1.0}, {"E": -0.5, "F": 0.0}]

    sequential_outputs = []
    for sample_alphas in batched_alphas:
        controller.set_alphas(sample_alphas)
        sequential_outputs.append(model(inputs_embeds=hidden[0:1]))
    sequential = torch.cat(sequential_outputs, dim=0)

    controller.set_batched_alphas(batched_alphas)
    batched = model(inputs_embeds=hidden)
    controller.clear_batched_alphas()
    controller.remove()

    assert torch.allclose(sequential, batched)


def test_prompt_mask_steers_only_continuation_positions():
    model = DummyModel()
    hidden = torch.zeros(1, 4, 4)
    controller = SteeringController(model, {"E": {0: torch.ones(4)}})
    controller.register()
    controller.set_alphas(
        {"E": 1.0}, prompt_mask=torch.tensor([True, True, False, False])
    )

    out = model(inputs_embeds=hidden)

    controller.remove()
    assert torch.allclose(out[:, :2], torch.zeros(1, 2, 4))
    assert torch.allclose(out[:, 2:], torch.ones(1, 2, 4))


def test_batched_prompt_masks_support_different_padding_layouts():
    model = DummyModel()
    hidden = torch.zeros(2, 4, 4)
    controller = SteeringController(model, {"E": {0: torch.ones(4)}})
    controller.register()
    controller.set_batched_alphas(
        [{"E": 1.0}, {"E": 2.0}],
        prompt_masks=torch.tensor(
            [
                [True, True, False, False],
                [True, False, True, False],
            ]
        ),
    )

    out = model(inputs_embeds=hidden)

    controller.remove()
    assert torch.allclose(out[0, :2], torch.zeros(2, 4))
    assert torch.allclose(out[0, 2:], torch.ones(2, 4))
    assert torch.allclose(out[1, 0], torch.zeros(4))
    assert torch.allclose(out[1, 1], torch.full((4,), 2.0))
    assert torch.allclose(out[1, 2], torch.zeros(4))
    assert torch.allclose(out[1, 3], torch.full((4,), 2.0))


def test_runtime_delta_smoke_test_measures_each_trait_layer():
    model = DummyModel()
    controller = SteeringController(
        model,
        {
            "E": {0: torch.ones(4)},
            "A": {1: torch.full((4,), 2.0)},
        },
    )
    controller.register()

    measurements = controller.measure_runtime_deltas(
        lambda: model(inputs_embeds=torch.zeros(1, 2, 4))
    )

    controller.remove()
    assert measurements["E"][0] > 0
    assert measurements["A"][1] > 0


def test_runtime_delta_smoke_test_rejects_zero_effect_hook():
    model = DummyModel()
    controller = SteeringController(model, {"E": {0: torch.zeros(4)}})
    controller.register()

    with pytest.raises(RuntimeError, match="trait=E layer=0"):
        controller.measure_runtime_deltas(
            lambda: model(inputs_embeds=torch.zeros(1, 1, 4))
        )

    controller.remove()
