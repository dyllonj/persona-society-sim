from __future__ import annotations

import torch
from torch import nn

from steering.hooks import SteeringController


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
