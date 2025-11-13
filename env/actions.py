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


MAX_BROADCAST_CHARS = 280


def move(world: World, agent_id: str, destination: str) -> ActionResult:
    world.move_agent(agent_id, destination)
    return ActionResult("move", True, {"destination": destination})


def talk(world: World, agent_id: str, utterance: str) -> ActionResult:
    location = world.agent_location(agent_id)
    truncated = utterance[:MAX_BROADCAST_CHARS]
    room_id = location if location != "unknown" else None
    world.broadcast(f"{agent_id}: {truncated}", room_id=room_id)
    return ActionResult("talk", True, {"utterance": truncated})


def trade(world: World, agent_id: str, item: str, qty: str) -> ActionResult:
    qty_int = int(qty)
    note = f"{agent_id} offers {qty_int} {item} at tick {world.tick}"
    location = world.agent_location(agent_id)
    room_id = location if location != "unknown" else None
    world.broadcast(note[:MAX_BROADCAST_CHARS], room_id=room_id)
    return ActionResult("trade", True, {"item": item, "qty": str(qty_int)})


ACTION_ROUTER = {
    "move": move,
    "talk": talk,
    "trade": trade,
    "research": None,  # patched below
    "cite": None,
    "submit_report": None,
}


def execute(world: World, agent_id: str, action_type: str, params: Dict[str, str]) -> ActionResult:
    handler = ACTION_ROUTER.get(action_type)
    if not handler:
        return ActionResult(action_type, False, {"error": "unsupported"})
    return handler(world, agent_id, **params)


# ---- Research Sprint actions ----

def research(world: World, agent_id: str, query: str | None = None, doc_id: str | None = None) -> ActionResult:
    location = world.agent_location(agent_id)
    if location != "library":
        note = "research outside library"
    else:
        note = "ok"
    info = world.research_access(agent_id, doc_id=doc_id, query=query)
    info.update({"note": note})
    room_id = location if location != "unknown" else None
    world.broadcast(f"{agent_id} researched {info.get('doc_id') or 'unknown'}", room_id=room_id)
    return ActionResult("research", True, info)


def cite(world: World, agent_id: str, doc_id: str | None = None) -> ActionResult:
    location = world.agent_location(agent_id)
    # If doc_id not provided, cite a recently accessed doc or fall back to any known doc
    if not doc_id:
        state = world._ensure_agent_research(agent_id)
        accessed = list(state.get("accessed_docs", []))  # type: ignore
        if accessed:
            doc_id = accessed[-1]
        elif world.corpus:
            doc_id = next(iter(world.corpus.keys()))
        else:
            doc_id = "unknown"
    world.add_citation(agent_id, doc_id)
    room_id = location if location != "unknown" else None
    world.broadcast(f"{agent_id} cited {doc_id}", room_id=room_id)
    return ActionResult("cite", True, {"doc_id": doc_id})


def submit_report(world: World, agent_id: str) -> ActionResult:
    result = world.grade_report(agent_id)
    location = world.agent_location(agent_id)
    room_id = location if location != "unknown" else None
    world.broadcast(
        f"{agent_id} submitted report: {result['facts_correct']}/{result['targets_total']} facts, {result['citations_valid']} cites",
        room_id=room_id,
    )
    return ActionResult("submit_report", True, {"grading": result})


# Attach dynamically to the router now that functions are defined
ACTION_ROUTER.update({
    "research": research,
    "cite": cite,
    "submit_report": submit_report,
})
