"""Action primitives for the town environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from env.world import World


@dataclass
class ActionResult:
    action_type: str
    success: bool
    info: Dict[str, str]


def move(world: World, agent_id: str, destination: str) -> ActionResult:
    world.move_agent(agent_id, destination)
    return ActionResult("move", True, {"destination": destination})


def talk(world: World, agent_id: str, utterance: str) -> ActionResult:
    world.broadcast(f"{agent_id}: {utterance}")
    return ActionResult("talk", True, {"utterance": utterance})


def trade(world: World, agent_id: str, item: str, qty: str) -> ActionResult:
    qty_int = int(qty)
    note = f"{agent_id} offers {qty_int} {item} at tick {world.tick}"
    world.broadcast(note)
    return ActionResult("trade", True, {"item": item, "qty": str(qty_int)})


ACTION_ROUTER = {
    "move": move,
    "talk": talk,
    "trade": trade,
}


def execute(world: World, agent_id: str, action_type: str, params: Dict[str, str]) -> ActionResult:
    handler = ACTION_ROUTER.get(action_type)
    if not handler:
        return ActionResult(action_type, False, {"error": "unsupported"})
    return handler(world, agent_id, **params)
