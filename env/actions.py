"""Action primitives for the town environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from env.world import World


@dataclass
class ActionResult:
    action_type: str
    success: bool
    info: Dict[str, Any]


MAX_BROADCAST_CHARS = 280


def move(world: World, agent_id: str, destination: str) -> ActionResult:
    # Suppress no-op moves
    current = world.agent_location(agent_id)
    if current == destination:
        return ActionResult("move", True, {"destination": destination, "note": "no_op"})
    world.move_agent(agent_id, destination)
    return ActionResult("move", True, {"destination": destination})


def talk(world: World, agent_id: str, utterance: str) -> ActionResult:
    location = world.agent_location(agent_id)
    truncated = utterance[:MAX_BROADCAST_CHARS]
    room_id = location if location != "unknown" else None
    world.broadcast(
        f"{agent_id}: {truncated}",
        room_id=room_id,
        speaker=agent_id,
        utterance=truncated,
    )
    return ActionResult("talk", True, {"utterance": truncated})


def trade(
    world: World,
    agent_id: str,
    item: str,
    qty: str,
    price: str = "1",
    side: str = "buy",
) -> ActionResult:
    try:
        qty_int = max(1, int(qty))
    except ValueError:
        return ActionResult("trade", False, {"error": "invalid_qty"})
    try:
        unit_price = float(price)
    except ValueError:
        unit_price = 1.0
    location = world.agent_location(agent_id)
    if location == "unknown":
        return ActionResult("trade", False, {"error": "no_location"})
    success, info = world.trade_with_location(
        agent_id=agent_id,
        location_id=location,
        item=item,
        qty=qty_int,
        price=unit_price,
        side=side,
    )
    room_id = location
    action_note = (
        f"{agent_id} {info.get('note', 'traded')} {qty_int} {item} ({side}) at {location}"
        if success
        else f"{agent_id} trade failed: {info.get('error', 'unknown')}"
    )
    world.broadcast(action_note[:MAX_BROADCAST_CHARS], room_id=room_id, speaker=agent_id, utterance=action_note)
    info.update({"item": item, "qty": str(qty_int), "side": side})
    return ActionResult("trade", success, info)


# ---- Town economy + policy actions ----

def work(world: World, agent_id: str, task: str = "town project") -> ActionResult:
    location = world.agent_location(agent_id)
    new_balance = world.adjust_resource(agent_id, "credits", 1)
    note = f"{agent_id} worked on {task} at {location}"
    room_id = location if location != "unknown" else None
    world.broadcast(note[:MAX_BROADCAST_CHARS], room_id=room_id, speaker=agent_id, utterance=note)
    return ActionResult(
        "work",
        True,
        {"task": task, "resource": "credits", "balance": str(new_balance)},
    )


def gift(
    world: World, agent_id: str, recipient: str, item: str = "credits", qty: str = "1"
) -> ActionResult:
    qty_int = max(1, int(qty))
    balance = world.resource_balance(agent_id, item)
    if balance < qty_int:
        return ActionResult("gift", False, {"error": "insufficient", "item": item})
    world.adjust_resource(agent_id, item, -qty_int)
    world.adjust_resource(recipient, item, qty_int)
    location = world.agent_location(agent_id)
    room_id = location if location != "unknown" else None
    note = f"{agent_id} gifted {qty_int} {item} to {recipient}"
    world.broadcast(note[:MAX_BROADCAST_CHARS], room_id=room_id, speaker=agent_id, utterance=note)
    return ActionResult("gift", True, {"item": item, "qty": str(qty_int), "to": recipient})


def scan(world: World, agent_id: str) -> ActionResult:
    location = world.agent_location(agent_id)
    token = world.acquire_scan_token(agent_id, location)
    if not token:
        return ActionResult("scan", False, {"note": "no_tokens"})
    room_id = location if location != "unknown" else None
    note = f"{agent_id} scanned {location} and found {token}"
    world.broadcast(note[:MAX_BROADCAST_CHARS], room_id=room_id, speaker=agent_id, utterance=note)
    return ActionResult("scan", True, {"token": token, "token_acquired": "1"})


def fill_field(world: World, agent_id: str, field_name: str, value: str) -> ActionResult:
    if not field_name:
        field_name = f"field_{world.tick}"
    saved = world.record_checklist_field(agent_id, field_name, value)
    location = world.agent_location(agent_id)
    room_id = location if location != "unknown" else None
    note = f"{agent_id} updated checklist field {field_name}"
    world.broadcast(note[:MAX_BROADCAST_CHARS], room_id=room_id, speaker=agent_id, utterance=note)
    if not saved:
        return ActionResult(
            "fill_field",
            False,
            {"field": field_name, "unique": "0", "note": "already_completed"},
        )
    return ActionResult("fill_field", True, {"field": field_name, "unique": "1"})


def propose_plan(world: World, agent_id: str, summary: str = "") -> ActionResult:
    plan = {
        "summary": summary or "Coordinate civic improvements",
        "fields": str(world.checklist_fields_completed(agent_id)),
    }
    world.agent_policy_plans[agent_id] = plan
    location = world.agent_location(agent_id)
    room_id = location if location != "unknown" else None
    note = f"{agent_id} drafted a plan proposal"
    world.broadcast(note[:MAX_BROADCAST_CHARS], room_id=room_id, speaker=agent_id, utterance=note)
    return ActionResult("propose_plan", True, plan)


def submit_plan(world: World, agent_id: str) -> ActionResult:
    ready = world.policy_plan_ready(agent_id)
    location = world.agent_location(agent_id)
    room_id = location if location != "unknown" else None
    fields_completed = world.checklist_fields_completed(agent_id)
    status = "submitted" if ready else "incomplete"
    note = (
        f"{agent_id} attempted to submit a plan ({fields_completed}/{world.policy_required_fields} fields)"
    )
    world.broadcast(note[:MAX_BROADCAST_CHARS], room_id=room_id, speaker=agent_id, utterance=note)
    info = {
        "fields_completed": str(fields_completed),
        "required": str(world.policy_required_fields),
        "status": status,
    }
    if ready:
        rule = world.enact_plan_rule(agent_id)
        if rule:
            info.update({"rule_id": rule.rule_id, "rule_text": rule.text})
    return ActionResult("submit_plan", ready, info)


ACTION_ROUTER = {
    "move": move,
    "talk": talk,
    "trade": trade,
    "work": None,
    "gift": None,
    "scan": None,
    "fill_field": None,
    "propose_plan": None,
    "submit_plan": None,
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
    raw_info = world.research_access(agent_id, doc_id=doc_id, query=query)
    info: Dict[str, Any] = {
        "note": note,
        "doc_id": raw_info.get("doc_id") or "",
        "facts_found": raw_info.get("facts_found", []),
    }
    room_id = location if location != "unknown" else None
    research_note = f"{agent_id} researched {raw_info.get('doc_id') or 'unknown'}"
    world.broadcast(
        research_note,
        room_id=room_id,
        speaker=agent_id,
        utterance=research_note,
    )
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
    cite_note = f"{agent_id} cited {doc_id}"
    world.broadcast(cite_note, room_id=room_id, speaker=agent_id, utterance=cite_note)
    return ActionResult("cite", True, {"doc_id": doc_id})


def submit_report(world: World, agent_id: str) -> ActionResult:
    result = world.grade_report(agent_id)
    location = world.agent_location(agent_id)
    room_id = location if location != "unknown" else None
    report_note = (
        f"{agent_id} submitted report: {result['facts_correct']}/{result['targets_total']} facts, "
        f"{result['citations_valid']} cites"
    )
    world.broadcast(
        report_note,
        room_id=room_id,
        speaker=agent_id,
        utterance=report_note,
    )
    return ActionResult("submit_report", True, result)


# Attach dynamically to the router now that functions are defined
ACTION_ROUTER.update({
    "work": work,
    "gift": gift,
    "scan": scan,
    "fill_field": fill_field,
    "propose_plan": propose_plan,
    "submit_plan": submit_plan,
    "research": research,
    "cite": cite,
    "submit_report": submit_report,
})
